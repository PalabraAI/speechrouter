#!/usr/bin/env python3
"""Capture the Gemini Live transcription wire and answer the open questions
in docs/providers/gemini.md.

Talks DIRECTLY to generativelanguage.googleapis.com, not through the gateway
— the point is the untouched vendor payloads that become test fixtures.

Usage:
    export GEMINI_API_KEY=...           # never pass the key as an argv
    uv run --project gateway python scripts/gemini_probe.py speech.wav --exp all

Each experiment writes <out>/<exp>.jsonl (one frame per line, with a
relative timestamp) plus <out>/summary.json with the verdicts.

Experiments, and the brief's open question each one closes:

    capture     baseline stream to clean end — frame shapes, interim vs
                final cadence, whether inputTranscription arrives whole or
                in fragments                                            (1)
    resume      ask for sessionResumption — are handles actually issued
                for this model, and does reconnecting with one work?    (2)
    flush       audioStreamEnd mid-stream, then keep sending — does the
                tail final arrive, and is the session still usable?     (3)
    rate        the same audio declared at 8000 in the mimeType — is a
                non-16k rate accepted?                                  (4)
    usage       read usageMetadata and compare real token spend against
                the blended $0.009/min estimate                         (5)
    smart       mode=SMART next to VERBATIM on the same audio
    auth        deliberately bad key — where the rejection surfaces
                (failed upgrade vs. close after setup)

`resume` and `flush` are the ones that matter most: rotation around the
~10-minute connection cap depends on both.
"""

import argparse
import asyncio
import base64
import contextlib
import json
import os
import sys
import time
import wave
from pathlib import Path

import websockets

WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
MODEL = "gemini-3.5-transcribe-live"
CHUNK_MS = 100  # docs recommend ~100ms chunks
SETUP_TIMEOUT = 15.0
EXPERIMENTS = ("capture", "resume", "flush", "rate", "usage", "smart", "auth")


def load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise SystemExit("expected 16-bit mono PCM wav")
        return f.readframes(f.getnframes()), f.getframerate()


def setup_message(*, mode: str = "", handle: str | None = None,
                  resumption: bool = False) -> str:
    transcription: dict = {"languageCodes": []}
    if mode:
        transcription["mode"] = mode
    setup: dict = {
        "model": f"models/{MODEL}",
        "generationConfig": {"responseModalities": ["TEXT"]},
        "inputAudioTranscription": transcription,
    }
    if resumption:
        setup["sessionResumption"] = {"handle": handle} if handle else {}
    return json.dumps({"setup": setup})


def audio_message(chunk: bytes, rate: int) -> str:
    return json.dumps(
        {"realtimeInput": {"audio": {"data": base64.b64encode(chunk).decode("ascii"),
                                     "mimeType": f"audio/pcm;rate={rate}"}}}
    )


AUDIO_STREAM_END = json.dumps({"realtimeInput": {"audioStreamEnd": True}})


class Recorder:
    """Frames + wall-clock offsets for one socket."""

    def __init__(self, out: Path, name: str):
        self._fh = out.joinpath(f"{name}.jsonl").open("w")
        self._t0 = time.monotonic()
        self.frames: list[dict] = []

    def at(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def record(self, msg: dict) -> dict:
        self.frames.append(msg)
        self._fh.write(json.dumps({"at": self.at(), "frame": msg}) + "\n")
        self._fh.flush()
        return msg

    def note(self, **fields) -> None:
        self._fh.write(json.dumps({"at": self.at(), **fields}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    # ---- convenience views over what came back -------------------------

    def texts(self, key: str) -> list[str]:
        return [
            (f.get("serverContent", {}).get(key) or {}).get("text", "")
            for f in self.frames
            if (f.get("serverContent", {}).get(key) or {}).get("text")
        ]

    def handles(self) -> list[str]:
        return [
            f["sessionResumptionUpdate"]["newHandle"]
            for f in self.frames
            if (f.get("sessionResumptionUpdate") or {}).get("newHandle")
        ]

    def usage(self) -> dict:
        seen = [f["usageMetadata"] for f in self.frames if f.get("usageMetadata")]
        return seen[-1] if seen else {}


async def dial(key: str, setup: str, rec: Recorder) -> websockets.ClientConnection:
    """Open, send setup, return once setupComplete arrives."""
    ws = await websockets.connect(f"{WS_BASE}?key={key}", open_timeout=15, max_size=None)
    await ws.send(setup)
    deadline = time.monotonic() + SETUP_TIMEOUT
    while time.monotonic() < deadline:
        msg = rec.record(json.loads(await asyncio.wait_for(ws.recv(), timeout=SETUP_TIMEOUT)))
        if "setupComplete" in msg:
            return ws
    raise SystemExit("no setupComplete within the timeout")


async def read_for(ws, rec: Recorder, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            rec.record(json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining)))
        except TimeoutError:
            return
        except websockets.exceptions.ConnectionClosed as exc:
            rec.note(event="closed", code=exc.code, reason=exc.reason)
            return


async def stream(ws, rec: Recorder, audio: bytes, rate: int, *,
                 declared_rate: int = 0, realtime: bool = True) -> None:
    """Feed the file while a reader drains frames, paced to realtime."""
    declared_rate = declared_rate or rate
    step = int(rate * 2 * CHUNK_MS / 1000)

    async def reader():
        try:
            while True:
                rec.record(json.loads(await ws.recv()))
        except websockets.exceptions.ConnectionClosed as exc:
            rec.note(event="closed", code=exc.code, reason=exc.reason)
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(reader())
    try:
        for i in range(0, len(audio), step):
            await ws.send(audio_message(audio[i : i + step], declared_rate))
            if realtime:
                await asyncio.sleep(CHUNK_MS / 1000)
    finally:
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------- experiments


async def exp_capture(key: str, audio: bytes, rate: int, out: Path) -> dict:
    rec = Recorder(out, "capture")
    ws = await dial(key, setup_message(), rec)
    await stream(ws, rec, audio, rate)
    await ws.send(AUDIO_STREAM_END)
    await read_for(ws, rec, 10.0)
    await ws.close()
    rec.close()
    finals = rec.texts("inputTranscription")
    return {
        "interims": len(rec.texts("interimInputTranscription")),
        "finals": len(finals),
        "final_texts": finals,
        # (1) fragments vs whole utterances: many short finals per utterance
        # means fragments, which clients must concatenate.
        "final_lengths": [len(t) for t in finals],
        "turn_completes": sum(
            1 for f in rec.frames if (f.get("serverContent") or {}).get("turnComplete")
        ),
    }


async def exp_resume(key: str, audio: bytes, rate: int, out: Path) -> dict:
    """(2) Are handles issued, and does reconnecting with one work?"""
    rec = Recorder(out, "resume")
    ws = await dial(key, setup_message(resumption=True), rec)
    half = len(audio) // 2
    await stream(ws, rec, audio[:half], rate)
    handles = rec.handles()
    rec.note(event="handles_seen", count=len(handles))
    await ws.close()
    if not handles:
        rec.close()
        return {"handles_issued": False, "resumed": False}
    rec.note(event="reconnecting", handle=handles[-1][:12] + "...")
    try:
        ws = await dial(key, setup_message(resumption=True, handle=handles[-1]), rec)
    except Exception as exc:  # noqa: BLE001 - the verdict IS the failure
        rec.note(event="resume_failed", error=str(exc))
        rec.close()
        return {"handles_issued": True, "resumed": False, "error": str(exc)}
    await stream(ws, rec, audio[half:], rate)
    await ws.send(AUDIO_STREAM_END)
    await read_for(ws, rec, 10.0)
    await ws.close()
    rec.close()
    return {"handles_issued": True, "resumed": True,
            "finals_after_resume": rec.texts("inputTranscription")}


async def exp_flush(key: str, audio: bytes, rate: int, out: Path) -> dict:
    """(3) Does audioStreamEnd flush the tail, and stay usable afterwards?"""
    rec = Recorder(out, "flush")
    ws = await dial(key, setup_message(), rec)
    half = len(audio) // 2
    await stream(ws, rec, audio[:half], rate)
    before = len(rec.texts("inputTranscription"))
    t0 = rec.at()
    await ws.send(AUDIO_STREAM_END)
    await read_for(ws, rec, 5.0)
    after_flush = len(rec.texts("inputTranscription"))
    flushed_at = rec.at() - t0
    # Still alive? Keep sending and see whether transcripts resume.
    rec.note(event="resending_after_audio_stream_end")
    try:
        await stream(ws, rec, audio[half:], rate)
        await ws.send(AUDIO_STREAM_END)
        await read_for(ws, rec, 10.0)
        reusable = len(rec.texts("inputTranscription")) > after_flush
    except websockets.exceptions.ConnectionClosed as exc:
        rec.note(event="closed_after_flush", code=exc.code, reason=exc.reason)
        reusable = False
    with contextlib.suppress(Exception):
        await ws.close()
    rec.close()
    return {
        "final_flushed_by_audio_stream_end": after_flush > before,
        "flush_latency_s": round(flushed_at, 2),
        "session_reusable_after_flush": reusable,
    }


async def exp_rate(key: str, audio: bytes, rate: int, out: Path) -> dict:
    """(4) Is a non-16k rate in the mimeType accepted?"""
    rec = Recorder(out, "rate")
    other = 8000 if rate != 8000 else 24000
    try:
        ws = await dial(key, setup_message(), rec)
        await stream(ws, rec, audio, rate, declared_rate=other)
        await ws.send(AUDIO_STREAM_END)
        await read_for(ws, rec, 10.0)
        await ws.close()
    except Exception as exc:  # noqa: BLE001 - the rejection IS the verdict
        rec.note(event="rejected", error=str(exc).replace(key, "***"))
        rec.close()
        return {"declared_rate": other, "accepted": False, "error": str(exc).replace(key, "***")}
    rec.close()
    texts = rec.texts("inputTranscription")
    return {"declared_rate": other, "accepted": bool(texts), "final_texts": texts}


async def exp_usage(key: str, audio: bytes, rate: int, out: Path) -> dict:
    """(5) Real token spend vs. the blended $0.009/min estimate."""
    rec = Recorder(out, "usage")
    ws = await dial(key, setup_message(), rec)
    await stream(ws, rec, audio, rate)
    await ws.send(AUDIO_STREAM_END)
    await read_for(ws, rec, 10.0)
    await ws.close()
    rec.close()
    usage = rec.usage()
    minutes = len(audio) / (rate * 2) / 60
    prompt = usage.get("promptTokenCount", 0)
    response = usage.get("responseTokenCount", 0)
    # Paid tier: $3.50/1M input audio tokens, $21.00/1M output text tokens.
    cost = prompt * 3.50e-6 + response * 21.00e-6
    return {
        "audio_minutes": round(minutes, 3),
        "usage_metadata": usage,
        "cost_usd": round(cost, 6),
        "usd_per_minute": round(cost / minutes, 5) if minutes else None,
        "catalog_usd_per_minute": 0.009,
    }


async def exp_smart(key: str, audio: bytes, rate: int, out: Path) -> dict:
    results = {}
    for mode in ("VERBATIM", "SMART"):
        rec = Recorder(out, f"smart-{mode.lower()}")
        ws = await dial(key, setup_message(mode=mode), rec)
        await stream(ws, rec, audio, rate)
        await ws.send(AUDIO_STREAM_END)
        await read_for(ws, rec, 10.0)
        await ws.close()
        rec.close()
        results[mode] = "".join(rec.texts("inputTranscription"))
    return results


async def exp_auth(key: str, audio: bytes, rate: int, out: Path) -> dict:
    """Where does a bad key surface — failed upgrade, or close after setup?"""
    rec = Recorder(out, "auth")
    # Shaped like nothing real: a plausible-looking key trips secret
    # scanners on the public repo even when it is obviously invalid.
    bad = "not-a-real-key-this-request-must-be-rejected"
    try:
        ws = await dial(bad, setup_message(), rec)
        await ws.close()
    except websockets.exceptions.InvalidStatus as exc:
        rec.note(event="upgrade_rejected", status=exc.response.status_code)
        rec.close()
        return {"surface": "handshake", "status": exc.response.status_code}
    except websockets.exceptions.ConnectionClosed as exc:
        rec.note(event="closed", code=exc.code, reason=exc.reason)
        rec.close()
        return {"surface": "close", "code": exc.code, "reason": exc.reason}
    except Exception as exc:  # noqa: BLE001
        rec.close()
        return {"surface": "other", "error": str(exc).replace(bad, "***")}
    rec.close()
    return {"surface": "accepted?!"}


RUNNERS = {
    "capture": exp_capture,
    "resume": exp_resume,
    "flush": exp_flush,
    "rate": exp_rate,
    "usage": exp_usage,
    "smart": exp_smart,
    "auth": exp_auth,
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav", help="16-bit mono PCM wav")
    parser.add_argument("--exp", default="capture",
                        help=f"comma-separated, or 'all': {', '.join(EXPERIMENTS)}")
    parser.add_argument("--out", default="gemini-probe", help="output directory")
    args = parser.parse_args()

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        print("set GEMINI_API_KEY", file=sys.stderr)
        return 2

    chosen = EXPERIMENTS if args.exp == "all" else tuple(e.strip() for e in args.exp.split(","))
    unknown = [e for e in chosen if e not in RUNNERS]
    if unknown:
        print(f"unknown experiment(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    audio, rate = load_wav(args.wav)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"model": MODEL, "wav": args.wav, "sample_rate": rate,
                     "audio_seconds": round(len(audio) / (rate * 2), 2)}
    for name in chosen:
        print(f"--- {name}")
        try:
            summary[name] = await RUNNERS[name](key, audio, rate, out)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the run
            summary[name] = {"error": str(exc).replace(key, "***")}
        print(json.dumps(summary[name], indent=2)[:2000])

    out.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

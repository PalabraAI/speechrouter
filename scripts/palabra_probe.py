#!/usr/bin/env python3
"""Capture Palabra's raw STT wire and answer the open questions in
docs/providers/palabra.md.

Talks DIRECTLY to Palabra, not through the gateway — the point is the
untouched vendor payloads that become test fixtures.

Usage:
    export PALABRA_API_KEY=...          # never pass the key as an argv
    uv run --project gateway python scripts/palabra_probe.py speech.wav --exp all

Each experiment writes <out>/<exp>.jsonl (one frame per line, with a
relative timestamp) plus <out>/summary.json with the verdicts.

Experiments, and the brief's open question each one closes:

    capture     baseline stream + clean close — frame shapes, is_eos cadence
    flush       stop sending, hold the socket open — does a final arrive?  (2)
    idle        connect, go silent — when does the server drop us?         (3)
    concurrent  two sockets on one key at the same time — 409?             (1)
    burst       push the file as fast as it goes — pacing enforced?        (8)
    translate   translate_languages=es — translated_transcription shape
    auth        deliberately bad key — 401 shape

`concurrent` is the one that matters most: if a key only allows one live
stream, failover re-dialing Palabra collides with its own closing socket.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

import websockets

WS_BASE = "wss://stream.palabra.ai/asr/v1/speech-to-text/stream"
CHUNK_MS = 320  # vendor-recommended chunk size (docs/providers/palabra.md)
EXPERIMENTS = ("capture", "flush", "idle", "concurrent", "burst", "translate", "auth")


def build_url(key: str, sample_rate: int, *, language: str | None, translate: str = "") -> str:
    query = [f"token={key}", "format=pcm_s16le", f"sample_rate={sample_rate}"]
    if language:
        query.append(f"language={language}")
    if translate:
        query.append(f"translate_languages={translate}")
    return f"{WS_BASE}?{'&'.join(query)}"


def load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2 or f.getnchannels() != 1:
            raise SystemExit("expected 16-bit mono PCM wav")
        return f.readframes(f.getnframes()), f.getframerate()


class Recorder:
    """Frames + wall-clock offsets for one socket."""

    def __init__(self, out: Path, name: str):
        self._fh = out.joinpath(f"{name}.jsonl").open("w")
        self._t0 = time.monotonic()
        self.frames: list[dict] = []
        self.records: list[dict] = []

    def at(self) -> float:
        return round(time.monotonic() - self._t0, 3)

    def note(self, kind: str, **fields) -> None:
        record = {"t": self.at(), "kind": kind, **fields}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.records.append(record)
        if kind == "recv":
            self.frames.append(record.get("frame", {}))

    def first(self, *kinds: str) -> dict:
        return next((r for r in self.records if r["kind"] in kinds), {})

    def recv(self, raw: str) -> dict:
        try:
            frame = json.loads(raw)
        except json.JSONDecodeError:
            self.note("recv_unparsed", raw=raw[:2000])
            return {}
        self.note("recv", frame=frame)
        return frame

    def close(self) -> None:
        self._fh.close()


async def pump(ws, audio: bytes, chunk_bytes: int, rec: Recorder, *, realtime: bool) -> None:
    for i in range(0, len(audio), chunk_bytes):
        await ws.send(audio[i : i + chunk_bytes])
        if realtime:
            await asyncio.sleep(CHUNK_MS / 1000)
    rec.note("audio_done", bytes_sent=len(audio))


async def drain(ws, rec: Recorder, *, seconds: float) -> None:
    """Read frames for a bounded window; return when it lapses or the socket dies."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except TimeoutError:
            return
        except websockets.exceptions.ConnectionClosed as exc:
            rec.note("closed", code=exc.code, reason=exc.reason)
            return
        if isinstance(raw, bytes):
            rec.note("recv_binary", size=len(raw))
            continue
        rec.recv(raw)


async def dial(url: str, rec: Recorder):
    """Connect, recording handshake failures (401/409) rather than raising.

    Ping settings match the official SDK (palabra-ai-python v2.1.0): liveness
    is standard WS ping/pong, NOT an app-level keepalive. Dialing with pings
    off would make the `idle` experiment measure our own silence rather than
    the server's audio-idle policy.
    """
    try:
        ws = await websockets.connect(
            url, open_timeout=15, ping_interval=10, ping_timeout=30, max_size=None
        )
    except websockets.exceptions.InvalidStatus as exc:
        rec.note("handshake_failed", status=exc.response.status_code)
        return None
    except Exception as exc:  # noqa: BLE001 — probe records everything
        rec.note("connect_error", error=f"{type(exc).__name__}: {exc}")
        return None
    rec.note("connected")
    return ws


async def exp_capture(args, key, audio, rate, out, name="capture", translate="") -> dict:
    """Baseline: stream at realtime, let the server finish, close cleanly."""
    rec = Recorder(out, name)
    ws = await dial(build_url(key, rate, language=args.language, translate=translate), rec)
    if ws is None:
        rec.close()
        return {"connected": False}
    chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
    async with ws:
        await asyncio.gather(
            pump(ws, audio, chunk_bytes, rec, realtime=True),
            drain(ws, rec, seconds=len(audio) / (rate * 2) + args.tail),
        )
    rec.note("client_closed", code=ws.close_code, reason=ws.close_reason)
    rec.close()
    types = [f.get("message_type") for f in rec.frames]
    finals = [f for f in rec.frames if f.get("is_eos")]
    return {
        "connected": True,
        "frames": len(rec.frames),
        "message_types": sorted(set(filter(None, types))),
        "finals": len(finals),
        "partials": len(types) - len(finals),
        "has_word_timestamps": any("words" in f.get("segment", {}) for f in rec.frames),
        "close_code": ws.close_code,
    }


async def exp_flush(args, key, audio, rate, out) -> dict:
    """Q2: stop sending audio but hold the socket open. Does a trailing final
    arrive on its own, or must finish() close the socket itself?"""
    rec = Recorder(out, "flush")
    ws = await dial(build_url(key, rate, language=args.language), rec)
    if ws is None:
        rec.close()
        return {"connected": False}
    chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
    async with ws:
        await pump(ws, audio, chunk_bytes, rec, realtime=True)
        before = len(rec.frames)
        await drain(ws, rec, seconds=args.hold)
        after = rec.frames[before:]
    rec.note("client_closed", code=ws.close_code, reason=ws.close_reason)
    rec.close()
    trailing_final = next((f for f in after if f.get("is_eos")), None)
    return {
        "frames_after_audio_stopped": len(after),
        "trailing_final_arrived": trailing_final is not None,
        "server_closed_on_its_own": ws.close_code is not None,
        "close_code": ws.close_code,
        "verdict": (
            "server flushes — finish() can just wait"
            if trailing_final
            else "no trailing final — finish() must grace-wait then close (Telnyx pattern)"
        ),
    }


async def exp_idle(args, key, audio, rate, out) -> dict:
    """Q3: send a little audio then go completely silent. How long until the
    server drops us, and is any keepalive expected?"""
    rec = Recorder(out, "idle")
    ws = await dial(build_url(key, rate, language=args.language), rec)
    if ws is None:
        rec.close()
        return {"connected": False}
    chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
    async with ws:
        await pump(ws, audio[: chunk_bytes * 3], chunk_bytes, rec, realtime=True)
        rec.note("going_silent", hold_seconds=args.idle)
        await drain(ws, rec, seconds=args.idle)
        dropped_at = rec.at() if ws.close_code is not None else None
    rec.close()
    return {
        "held_seconds": args.idle,
        "dropped_by_server": ws.close_code is not None,
        "dropped_at_seconds": dropped_at,
        "close_code": ws.close_code,
        "close_reason": ws.close_reason,
    }


async def exp_concurrent(args, key, audio, rate, out) -> dict:
    """Q1 — highest impact. Two sockets on one key at the same time. A 409 here
    means SPEECHROUTER_MAX_CONCURRENT_STREAMS oversubscribes Palabra and that
    failover will collide with its own closing socket."""
    rec_a, rec_b = Recorder(out, "concurrent_a"), Recorder(out, "concurrent_b")
    url = build_url(key, rate, language=args.language)
    ws_a = await dial(url, rec_a)
    if ws_a is None:
        rec_a.close()
        rec_b.close()
        return {"first_connected": False}
    chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
    # Keep A busy so it is unambiguously live when B dials.
    task_a = asyncio.create_task(pump(ws_a, audio, chunk_bytes, rec_a, realtime=True))
    await asyncio.sleep(1.0)
    ws_b = await dial(url, rec_b)
    second_ok = ws_b is not None
    if ws_b is not None:
        await drain(ws_b, rec_b, seconds=2.0)
        await ws_b.close()
    task_a.cancel()
    await ws_a.close()
    rejection = rec_b.first("handshake_failed", "connect_error")
    rec_a.close()
    rec_b.close()
    return {
        "first_connected": True,
        "second_connected": second_ok,
        "second_rejection": rejection,
        "verdict": (
            "concurrent streams OK on one key"
            if second_ok
            else f"ONE stream per key (second dial: {rejection}) — "
            "failover and MAX_CONCURRENT_STREAMS both need rework"
        ),
    }


async def exp_burst(args, key, audio, rate, out) -> dict:
    """Q8: push the whole file with no pacing. If the server objects, the
    adapter needs realtime_pacing_required and throttling inside send_audio."""
    rec = Recorder(out, "burst")
    ws = await dial(build_url(key, rate, language=args.language), rec)
    if ws is None:
        rec.close()
        return {"connected": False}
    chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
    async with ws:
        await pump(ws, audio, chunk_bytes, rec, realtime=False)
        await drain(ws, rec, seconds=args.tail)
    rec.note("client_closed", code=ws.close_code, reason=ws.close_reason)
    rec.close()
    # The SDK knows AUDIO_STREAM_TOO_FAST/TOO_SLOW/STALLED warning codes, so
    # the server may object in-band rather than by dropping the socket.
    warnings = [
        f for f in rec.frames if f.get("message_type") in ("warning", "error")
    ]
    return {
        "frames": len(rec.frames),
        "dropped": ws.close_code not in (None, 1000),
        "warnings": warnings,
        "close_code": ws.close_code,
        "close_reason": ws.close_reason,
        "verdict": (
            "pacing enforced — realtime_pacing_required=True"
            if warnings or ws.close_code not in (None, 1000)
            else "burst accepted silently; keep pacing anyway (SDK says the server requires it)"
        ),
    }


async def exp_auth(args, _key, _audio, rate, out) -> dict:
    """Confirm the documented 401 on a bad key, and what the client sees."""
    rec = Recorder(out, "auth")
    ws = await dial(build_url("definitely-not-a-real-key", rate, language=args.language), rec)
    if ws is not None:
        await ws.close()
    observed = rec.first("handshake_failed", "connect_error")
    rec.close()
    return {"connected": ws is not None, "observed": observed}


def prepare_out(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary(out: Path, summary: dict) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="16-bit mono PCM wav with real speech in it")
    parser.add_argument("--exp", default="capture", help=f"{'|'.join(EXPERIMENTS)}|all")
    parser.add_argument("--language", default="en", help="empty string = autodetect")
    parser.add_argument("--out", default="captures/palabra")
    parser.add_argument("--tail", type=float, default=5.0, help="drain window after audio")
    parser.add_argument("--hold", type=float, default=15.0, help="flush: silence hold")
    parser.add_argument("--idle", type=float, default=90.0, help="idle: silence hold")
    args = parser.parse_args()

    key = os.environ.get("PALABRA_API_KEY", "")
    if not key:
        print("set PALABRA_API_KEY in the environment", file=sys.stderr)
        return 1

    audio, rate = load_wav(args.wav)
    out = prepare_out(args.out)

    runners = {
        "capture": exp_capture,
        "flush": exp_flush,
        "idle": exp_idle,
        "concurrent": exp_concurrent,
        "burst": exp_burst,
        "translate": lambda a, k, au, r, o: exp_capture(a, k, au, r, o, "translate", "es"),
        "auth": exp_auth,
    }
    chosen = EXPERIMENTS if args.exp == "all" else tuple(args.exp.split(","))

    summary: dict[str, dict] = {}
    for name in chosen:
        if name not in runners:
            print(f"unknown experiment '{name}'", file=sys.stderr)
            return 1
        print(f"\n=== {name} ===")
        summary[name] = await runners[name](args, key, audio, rate, out)
        print(json.dumps(summary[name], indent=2, ensure_ascii=False))
        await asyncio.sleep(2)  # let the previous session settle before the next dial

    write_summary(out, summary)
    print(f"\nframes -> {out}/*.jsonl, verdicts -> {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

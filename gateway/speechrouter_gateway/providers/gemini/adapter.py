"""Gemini Live transcription adapter (raw WebSocket, BidiGenerateContent).

Protocol facts (docs/providers/gemini.md). Sourced from ai.google.dev/
gemini-api/docs/live-api/live-transcribe and /api/live, then **live-verified
against the real API on 2026-08-26** — the notes below marked (live) are
measured, and three of them contradict the documentation:

- This is the **Gemini API**, not Google Cloud Speech-to-Text. Different
  product, different credential (a plain `GEMINI_API_KEY`, no project or
  service account), different wire (WebSocket, not gRPC). It therefore gets
  its own provider id rather than another model under `google/`.
- Endpoint: wss://generativelanguage.googleapis.com/ws/
  google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent
  with the API key as the `?key=` query param.
- The first client frame is `setup`; the server answers `setupComplete`.
  **Audio sent before setupComplete is dropped**, so connect() waits for it
  (the STTStreamProvider contract: connect returns only when ready).
- Up: JSON text frames. Audio is base64 inside
  `realtimeInput.audio {data, mimeType: "audio/pcm;rate=16000"}` — raw
  16-bit little-endian mono PCM at 16 kHz.
- Down: `serverContent.interimInputTranscription.text` are speculative
  partials while the speaker is talking; `serverContent.inputTranscription
  .text` is the finalized text. (live) The final is the WHOLE utterance,
  not a fragment to concatenate. Neither carries any timestamp, word list,
  confidence or speaker label — see _NO_TIMES.
- (live) The end-of-turn signal is `serverContent.generationComplete`.
  `turnComplete` is documented but never sent by this model — see
  _TURN_END_KEYS.
- (live) **`realtimeInput.audioStreamEnd: true` is MANDATORY to get any
  final at all.** Stopping the audio and waiting yields nothing, ever: the
  endpointer does not fire on silence. That makes the flush in finish() and
  the pre-swap flush in _ROTATION load-bearing, not just tidy. The value
  must be the boolean `true` — the object form `{}` is rejected with a 1007
  close ("Invalid value at 'realtime_input' (audio_stream_end)").
- **A connection lives about 10 minutes.** The server sends `goAway
  {timeLeft}` shortly before it closes with ABORTED. See _ROTATION.
- (live) `sessionResumption` is accepted in setup but this model issues NO
  `sessionResumptionUpdate` handles, so rotation reconnects as a fresh
  session. Harmless for transcription — there is no conversational context
  worth carrying — and the request stays in place for when it starts
  working. Rotation was verified end to end with ROTATE_SECONDS shrunk to
  12s: three boundaries crossed, no audio lost across any swap.
- (live) `usageMetadata` is never sent on this model, so per-stream token
  spend cannot be read off the wire. See the Pricing note in the brief.

_NO_TIMES: the wire has no timestamps at all, so every Transcript here
carries start=None/end=None/words=None (same shape as the telnyx adapter).
That means the session layer's failover dedup gate — which keys on a final's
`end` — cannot suppress replayed finals for this provider. Synthesising a
timestamp from the ingest cursor would make the gate *look* like it works
while deduping on a number no provider ever reported, so it is not done.

_ROTATION: the ~10-minute connection cap is invisible to callers. This
adapter rotates its own socket at ROTATE_SECONDS (or immediately on
`goAway`), reconnecting fresh (no handles are issued — see above). Before the swap
it flushes the old socket with audioStreamEnd so the in-flight utterance
arrives as a final instead of being lost, and audio that arrives during the
swap is buffered rather than blocking the caller — a stall here would show
up as a latency spike in a live agent every nine minutes.
"""

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Iterator

import websockets
import websockets.exceptions  # lazy submodule: `import websockets` alone leaves it unbound

from ...audio import bytes_per_second
from ...config import Settings
from ...logging import logger
from ...protocol import Transcript
from ..base import (
    Capabilities,
    ProviderStreamError,
    STTConfig,
    STTEvent,
    STTStreamProvider,
)
from ..registry import ProviderNotConfigured, register_stt_stream
from ..wsconnect import ws_connect

WS_BASE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

# Rotate at 9:00, under the ~10-minute connection lifetime. See _ROTATION.
ROTATE_SECONDS = 540.0
SETUP_TIMEOUT = 15.0  # setupComplete after the upgrade
ROTATE_FLUSH_SECONDS = 1.5  # wait for the tail final before swapping sockets
FINISH_GRACE_SECONDS = 5.0  # wait for the last final after audioStreamEnd

# Docs recommend ~100ms chunks; this is the ceiling on one base64 frame
# (1s of 16kHz mono PCM), not a re-chunker. Client chunking passes through.
MAX_AUDIO_BYTES = 32 * 1024

# Vendor cap on customVocabulary; Google recommends staying near 100.
MAX_CUSTOM_VOCABULARY = 1000

# Ceiling on audio buffered across a socket swap, in seconds. A swap takes
# ~2s; anything past this means the reconnect is wedged and the oldest audio
# is dropped rather than growing the process's memory without bound.
MAX_PENDING_SECONDS = 30.0

PING_INTERVAL = 20
PING_TIMEOUT = 60

# setup keys the adapter owns; a provider_params copy would either be
# ignored or break rotation.
RESERVED_SETUP_KEYS = frozenset({"model", "sessionResumption"})
# provider_params routed into setup.inputAudioTranscription instead of the
# top level. These are the Live API's own camelCase field names.
TRANSCRIPTION_KEYS = frozenset({"mode", "customVocabulary", "languageCodes"})

# Handshake statuses where reconnecting or failing over to another Gemini
# session cannot help.
_UNRECOVERABLE_STATUS = {400, 401, 403, 404}

CAPABILITIES = Capabilities(
    streaming=True,
    interim_results=True,  # interimInputTranscription
    word_timestamps=False,  # see _NO_TIMES
    diarization=False,  # documented as unavailable in live mode
    endpointing=False,  # turnComplete is a VAD edge but carries no time
    keyterms=True,  # customVocabulary
    keyterms_max=MAX_CUSTOM_VOCABULARY,
    languages=frozenset({"auto"}),
    encodings=frozenset({"linear16"}),
    sample_rates=frozenset({16000}),
)


def language_codes(config: STTConfig) -> list[str]:
    """`[]` is the API's own spelling of automatic language detection."""
    if not config.language or config.language == "auto":
        return []
    return [config.language]


def build_setup_message(config: STTConfig, *, handle: str = "") -> dict:
    """The first client frame. Pure, so fixture tests need no socket."""
    transcription: dict = {"languageCodes": language_codes(config)}
    if config.keyterms:
        transcription["customVocabulary"] = list(config.keyterms)[:MAX_CUSTOM_VOCABULARY]

    setup: dict = {
        "model": f"models/{config.model}",
        "generationConfig": {"responseModalities": ["TEXT"]},
        # Asking for handles costs nothing when the model does not issue
        # them; when it does, rotation keeps the logical session.
        "sessionResumption": {"handle": handle} if handle else {},
    }

    for key, value in config.provider_params.items():
        if key in RESERVED_SETUP_KEYS:
            logger.warning(
                "gemini provider_params key is owned by the adapter; ignored",
                extra={"provider": "gemini", "param": key},
            )
            continue
        if key in TRANSCRIPTION_KEYS:
            transcription[key] = value
        else:
            setup[key] = value

    setup["inputAudioTranscription"] = transcription
    return {"setup": setup}


def audio_message(chunk: bytes, sample_rate: int) -> str:
    return json.dumps(
        {
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(chunk).decode("ascii"),
                    "mimeType": f"audio/pcm;rate={sample_rate}",
                }
            }
        }
    )


AUDIO_STREAM_END = json.dumps({"realtimeInput": {"audioStreamEnd": True}})

# End-of-turn signals. Live-verified 2026-08-26: this model emits
# `generationComplete`, NEVER `turnComplete` — watching only for the
# latter left finish() waiting out its full grace timeout on every
# stream. Both are accepted so a future turnComplete still ends cleanly.
_TURN_END_KEYS = ("generationComplete", "turnComplete")


def turn_ended(msg: dict) -> bool:
    content = msg.get("serverContent") or {}
    return any(content.get(key) for key in _TURN_END_KEYS)


def parse_server_message(msg: dict, include_raw: bool = False,
                         lang: str | None = None) -> list[STTEvent]:
    """Translate one decoded server frame into normalized events.

    Side-effect free and socket-free so fixture tests drive it directly.
    Control frames (setupComplete, goAway, sessionResumptionUpdate) carry no
    transcript and are the adapter's business, not this function's.
    """
    error = msg.get("error")
    if error:
        status = str(error.get("status") or error.get("code") or "")
        raise ProviderStreamError(
            f"gemini error: {error.get('message') or status or 'unknown'}",
            # INVALID_ARGUMENT/PERMISSION_DENIED reject the request itself;
            # retrying the same setup reproduces them exactly.
            recoverable=status not in {"INVALID_ARGUMENT", "PERMISSION_DENIED",
                                       "NOT_FOUND", "UNAUTHENTICATED", "400", "401",
                                       "403", "404"},
            provider="gemini",
            code=status,
        )

    content = msg.get("serverContent") or {}
    events: list[STTEvent] = []
    # Interim first: within one frame the finalized text supersedes the
    # speculative hypothesis, and the client applies events in order.
    for key, is_final in (("interimInputTranscription", False),
                          ("inputTranscription", True)):
        text = ((content.get(key) or {}).get("text") or "")
        if not text.strip():
            continue
        events.append(
            Transcript(
                type="transcript",
                is_final=is_final,
                text=text,
                words=None,  # see _NO_TIMES
                start=None,
                end=None,
                lang=lang,
                provider_raw=msg if include_raw else None,
            )
        )
    return events


def redact(text: str, api_key: str) -> str:
    """The key rides in the query string, and websockets' InvalidURI quotes
    the URL back — strip it before anything reaches a log or a client."""
    return text.replace(api_key, "***") if api_key else text


@register_stt_stream("gemini", capabilities=CAPABILITIES)
def build(settings: Settings) -> "GeminiSTTStream":
    if not settings.gemini_api_key:
        raise ProviderNotConfigured("gemini")
    return GeminiSTTStream(settings.gemini_api_key)


class GeminiSTTStream(STTStreamProvider):
    name = "gemini"
    capabilities = CAPABILITIES

    def __init__(self, api_key: str, ws_base: str = WS_BASE):
        self._api_key = api_key
        self._ws_base = ws_base
        self._ws: websockets.ClientConnection | None = None
        self._config: STTConfig | None = None
        self._include_raw = False
        self._lang: str | None = None
        self._sample_rate = 16000
        self._byte_rate = 0
        self._connected = False
        self._finished = False
        self._closed = False
        # Rotation state, see _ROTATION.
        self._send_lock = asyncio.Lock()
        self._rotating = False
        self._pending = bytearray()
        self._handle = ""

    # ------------------------------------------------------------- dialing

    def _url(self) -> str:
        return f"{self._ws_base}?key={self._api_key}"

    async def _dial(self) -> websockets.ClientConnection:
        """Open a socket, send setup, return once setupComplete arrives."""
        assert self._config is not None
        try:
            ws = await ws_connect(
                self._url(),
                ping_interval=PING_INTERVAL,  # WS pings are the only liveness
                ping_timeout=PING_TIMEOUT,
                max_size=None,
            )
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
            raise ProviderStreamError(
                f"gemini connect rejected ({status})",
                recoverable=status not in _UNRECOVERABLE_STATUS,
                provider=self.name,
                code=str(status),
            ) from exc
        except Exception as exc:
            raise ProviderStreamError(
                f"gemini connect failed: {redact(str(exc), self._api_key)}",
                recoverable=True,
                provider=self.name,
            ) from exc

        try:
            await ws.send(json.dumps(build_setup_message(self._config, handle=self._handle)))
            await self._await_setup_complete(ws)
        except ProviderStreamError:
            with contextlib.suppress(Exception):
                await ws.close()
            raise
        except Exception as exc:
            with contextlib.suppress(Exception):
                await ws.close()
            raise ProviderStreamError(
                f"gemini setup failed: {redact(str(exc), self._api_key)}",
                recoverable=True,
                provider=self.name,
            ) from exc
        return ws

    async def _await_setup_complete(self, ws: websockets.ClientConnection) -> None:
        """Audio sent before setupComplete is dropped, so block on it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SETUP_TIMEOUT
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ProviderStreamError(
                    "gemini setup timed out", recoverable=True, provider=self.name
                )
            try:
                msg = _decode(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except TimeoutError as exc:
                raise ProviderStreamError(
                    "gemini setup timed out", recoverable=True, provider=self.name
                ) from exc
            except websockets.exceptions.ConnectionClosed as exc:
                # A rejected key can also arrive as a close during setup
                # rather than as a failed upgrade.
                raise ProviderStreamError(
                    f"gemini closed during setup {exc.code}: {exc.reason}",
                    recoverable=exc.code != 1008,  # 1008 = policy violation
                    provider=self.name,
                    code=str(exc.code),
                ) from exc
            if msg is None:
                continue
            parse_server_message(msg)  # raises on an error frame
            self._absorb_handle(msg)
            if "setupComplete" in msg:
                return

    def _absorb_handle(self, msg: dict) -> None:
        update = msg.get("sessionResumptionUpdate") or {}
        if update.get("resumable") and update.get("newHandle"):
            self._handle = update["newHandle"]

    # ------------------------------------------------------------- lifecycle

    async def connect(self, config: STTConfig) -> None:
        # The wire carries no channel count and the API documents mono
        # input, so interleaved stereo comes back as garbage at double
        # speed. Refuse rather than transcribe nonsense.
        if config.channels != 1:
            raise ProviderStreamError(
                f"gemini accepts mono audio only (got {config.channels} channels)",
                recoverable=False,
                provider=self.name,
            )
        self._config = config
        self._include_raw = config.include_raw
        # The wire reports no detected language; echoing the one we asked
        # for is the only honest tag, and auto-detect stays untagged.
        self._lang = config.language if config.language and config.language != "auto" else None
        self._sample_rate = config.sample_rate
        self._byte_rate = (
            bytes_per_second(config.encoding, config.sample_rate, config.channels) or 0
        )
        self._ws = await self._dial()
        self._connected = True
        logger.info("gemini connected", extra={"provider": self.name, "model": config.model})

    async def send_audio(self, chunk: bytes) -> None:
        async with self._send_lock:
            ws = self._ws
            if ws is None:
                if not self._connected:
                    raise ProviderStreamError(
                        "send before connect", recoverable=False, provider=self.name
                    )
                # Mid-rotation, or a rotation that failed and is about to
                # surface out of events() as a failover trigger. Buffering
                # here keeps send_audio from racing that verdict.
                self._buffer(chunk)
                return
            if self._rotating:
                self._buffer(chunk)
                return
            await self._send_now(ws, chunk)

    def _buffer(self, chunk: bytes) -> None:
        """Hold audio across a socket swap. See _ROTATION."""
        self._pending.extend(chunk)
        cap = int(MAX_PENDING_SECONDS * self._byte_rate) if self._byte_rate else 0
        if cap and len(self._pending) > cap:
            dropped = len(self._pending) - cap
            del self._pending[:dropped]
            logger.warning(
                "gemini dropped buffered audio while rotating",
                extra={"provider": self.name, "bytes": dropped},
            )

    async def _send_now(self, ws: websockets.ClientConnection, chunk: bytes) -> None:
        for frame in _split(chunk, MAX_AUDIO_BYTES):
            await ws.send(audio_message(frame, self._sample_rate))

    async def events(self) -> AsyncIterator[STTEvent]:
        if self._ws is None:
            raise ProviderStreamError(
                "events before connect", recoverable=False, provider=self.name
            )
        while True:
            ws = self._ws
            if ws is None:
                return
            rotate = False
            try:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + ROTATE_SECONDS
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        rotate = True
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except TimeoutError:
                        rotate = True
                        break
                    msg = _decode(raw)
                    if msg is None:
                        continue
                    self._absorb_handle(msg)
                    if "goAway" in msg:
                        logger.info(
                            "gemini goAway; rotating socket",
                            extra={"provider": self.name,
                                   "time_left": (msg["goAway"] or {}).get("timeLeft")},
                        )
                        rotate = True
                        break
                    for event in parse_server_message(msg, self._include_raw, self._lang):
                        yield event
                    if self._finished and turn_ended(msg):
                        return  # flush after audioStreamEnd is done
            except websockets.exceptions.ConnectionClosedOK:
                if self._finished:
                    return
                rotate = True  # the 10-minute cap can also arrive as a clean close
            except websockets.exceptions.ConnectionClosed as exc:
                if self._finished:
                    return
                raise ProviderStreamError(
                    f"gemini closed {exc.code}: {exc.reason}",
                    recoverable=exc.code != 1008,  # 1008 = policy violation
                    provider=self.name,
                    code=str(exc.code),
                ) from exc
            if self._finished or not rotate:
                return
            async for event in self._rotate():
                yield event

    # ------------------------------------------------------------- rotation

    async def _rotate(self) -> AsyncIterator[STTEvent]:
        """Swap in a fresh socket without the caller noticing. See _ROTATION."""
        old = self._ws
        async with self._send_lock:
            self._rotating = True  # send_audio buffers from here on
        try:
            if old is not None:
                async for event in self._flush(old):
                    yield event
                with contextlib.suppress(Exception):
                    await old.close()
            new = await self._dial()
        except ProviderStreamError:
            async with self._send_lock:
                self._rotating = False
                self._ws = None
            raise
        async with self._send_lock:
            self._ws = new
            if self._pending:
                buffered = bytes(self._pending)
                self._pending.clear()
                await self._send_now(new, buffered)
            self._rotating = False
        logger.info(
            "gemini socket rotated",
            extra={"provider": self.name, "resumed": bool(self._handle)},
        )
        if self._finished:
            # finish() landed mid-rotation, so its audioStreamEnd went to the
            # socket being replaced. Repeat it here or the client's last
            # final never arrives.
            async for event in self._flush(new):
                yield event
            with contextlib.suppress(Exception):
                await new.close()

    async def _flush(self, ws: websockets.ClientConnection) -> AsyncIterator[STTEvent]:
        """audioStreamEnd on the outgoing socket, then collect the tail final."""
        try:
            await ws.send(AUDIO_STREAM_END)
        except websockets.exceptions.ConnectionClosed:
            return  # already gone: nothing to flush into
        loop = asyncio.get_running_loop()
        deadline = loop.time() + ROTATE_FLUSH_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                msg = _decode(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except (TimeoutError, websockets.exceptions.ConnectionClosed):
                return
            if msg is None:
                continue
            self._absorb_handle(msg)
            try:
                for event in parse_server_message(msg, self._include_raw, self._lang):
                    yield event
            except ProviderStreamError:
                # The socket being replaced is not worth failing the session
                # over; the fresh dial is the real verdict.
                logger.warning("gemini error frame while rotating",
                               extra={"provider": self.name})
                return
            if turn_ended(msg):
                return

    # ------------------------------------------------------------- shutdown

    async def finish(self) -> None:
        """audioStreamEnd makes the endpointer fire and emit the last final.

        events() returns as soon as the matching turnComplete arrives; the
        grace close below is the backstop for a server that stays quiet, so
        a client waiting on `done` never hangs to the session hard cap.
        """
        if self._finished:
            return
        self._finished = True
        async with self._send_lock:
            ws = self._ws
            if ws is not None and self._pending:
                buffered = bytes(self._pending)
                self._pending.clear()
                with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                    await self._send_now(ws, buffered)
            if ws is not None:
                with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                    await ws.send(AUDIO_STREAM_END)
        if ws is None:
            return
        await asyncio.sleep(FINISH_GRACE_SECONDS)
        # events() treats a close at this point (finished=True) as a clean
        # end of stream, whatever close code arrives.
        with contextlib.suppress(Exception):
            await ws.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._finished = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass


def _decode(raw) -> dict | None:
    """Server frames are JSON; the API may deliver them as binary."""
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", "replace")
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return msg if isinstance(msg, dict) else None


def _split(chunk: bytes, size: int) -> Iterator[bytes]:
    for i in range(0, len(chunk), size):
        yield chunk[i : i + size]

"""Meta Muse Voice Transcribe streaming adapter (raw WebSocket, /v1/asr/realtime).

Protocol notes and the open questions behind the choices here:
docs/providers/meta.md (written from dev.meta.ai/docs/speech-to-text and the
Voice API reference, 2026-09-01).

- The credential travels in the FIRST JSON text frame (the handshake), not in
  an HTTP header. connect() returns after the `{"sessionId": ...}` ack, which
  is the only server frame without a `type`.
- Audio is raw signed 16-bit LE mono PCM at 16 kHz or 24 kHz. Nothing else is
  accepted, so connect() refuses other rates and channel counts up front.
- Timestamps come only as `audioProcessedMs`, a processing-progress counter
  in ms from the start of the stream. It is monotonic, which is all the
  session layer's dedup gate needs, but it is not an acoustic boundary.
  No word timestamps, no confidence.
- Three modes. `ENDPOINTING` (our default) emits one turn per utterance:
  speechStart -> partial transcripts -> speechEnd -> speechComplete. The
  final text is in speechComplete, which may differ from the last partial
  (punctuation, casing), so `transcript` frames are always interim in the
  turn modes even when they say `final: true`. `PUSH_TO_TALK` (opt in via
  provider_params.mode) has no turns: the one `final: true` transcript
  arrives after endStream. `DIARIZATION` is selected by diarization=true and
  adds one `speaker` frame per turn; the label rides out as a segment-level
  Word carrying the speaker index, the same convention openai_compat uses
  for diarized segments.
- End of input is `{"type": "endStream"}`; the server flushes and closes 1000.
- An `error` frame always precedes a fatal close, so events() records the
  message and classifies on the close code (1008 = fix the request, never
  retry; 1011/1013 = retry with backoff).
- The server enforces pacing: no more than ~5 s of audio ahead of processing,
  and not slower than realtime either. send_audio meters through a token
  bucket so a failover replay of the ring buffer cannot trip the backlog cap.
  See _PACING.
- There is no application keepalive; the server closes a stream that stops
  sending audio without endStream. A live source must keep sending PCM
  (silence included). The adapter does not synthesize silence because
  injected audio would shift every later timestamp against the client's own
  audio clock.

_PACING: _MAX_BURST_SECONDS of audio may go out immediately (below the 5 s
backlog cap, with headroom for the server's own processing lag); after that
frames are metered at _MAX_REALTIME_FACTOR x realtime so a replayed backlog
still drains. Both are unverified against the live service; see the doc.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Mapping
from urllib.parse import urlencode

import websockets

from ...audio import bytes_per_second
from ...config import Settings
from ...logging import logger
from ...protocol import SpeechStarted, Transcript, UtteranceEnd, Word
from ..base import (
    Capabilities,
    ProviderStreamError,
    STTConfig,
    STTEvent,
    STTStreamProvider,
)
from ..registry import ProviderNotConfigured, register_stt_stream
from ..wsconnect import ws_connect

WS_URL = "wss://api.meta.ai/v1/asr/realtime"

# Server-side deadline for the handshake is 10 s after connect; this bounds
# how long we wait for the ack once it has been sent.
HANDSHAKE_TIMEOUT = 15.0

# Only linear16 mono at these rates exists on the wire.
_AUDIO_ENCODINGS = {16000: "PCM_16KHZ", 24000: "PCM_24KHZ"}
SAMPLE_RATES = frozenset(_AUDIO_ENCODINGS)

MODE_PUSH_TO_TALK = "PUSH_TO_TALK"
MODE_ENDPOINTING = "ENDPOINTING"
MODE_DIARIZATION = "DIARIZATION"
MODES = frozenset({MODE_PUSH_TO_TALK, MODE_ENDPOINTING, MODE_DIARIZATION})
# Meta's own default is PUSH_TO_TALK, which never emits a final until
# endStream. A streaming gateway wants one final per utterance.
DEFAULT_MODE = MODE_ENDPOINTING

# Handshake keys the adapter owns. partialMode stays CUMULATIVE because the
# normalized interim contract is "replace the previous interim"; DELTA frames
# would be forwarded as if they were whole hypotheses. audioProgress frames
# carry nothing we surface.
RESERVED_HANDSHAKE_KEYS = frozenset(
    {"authorization", "audioEncoding", "model", "partialMode", "emitAudioProgress"}
)

# `languageBias` wants language names, not tags. Primary subtags of the 25
# documented languages; anything else passes through verbatim so a caller
# who already sends a name is not second-guessed.
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "nl": "Dutch",
    "en": "English",
    "fr": "French",
    "de": "German",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ms": "Malay",
    "zh": "Mandarin Chinese",
    "mr": "Marathi",
    "pl": "Polish",
    "pt": "Portuguese",
    "es": "Spanish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}

# Ingest pacing, see _PACING.
_MAX_REALTIME_FACTOR = 1.5
_MAX_BURST_SECONDS = 3.0
_MAX_FRAME_MS = 320  # a replayed 10 s ring chunk is split so the bucket can meter it

# Close codes (docs/providers/meta.md, "Close codes").
_CLOSE_INVALID_REQUEST = 1008  # bad config, bad key, or a pacing violation
_CLOSE_RATE_LIMITED = 1013

CAPABILITIES = Capabilities(
    streaming=True,
    interim_results=True,
    word_timestamps=False,  # turn-level audioProcessedMs only
    diarization=True,  # DIARIZATION mode; label per turn, not per word
    endpointing=True,  # ENDPOINTING mode: speechStart / speechEnd
    keyterms=True,  # `keywords`
    languages=frozenset({"auto", *LANGUAGE_NAMES}),
    encodings=frozenset({"linear16"}),
    sample_rates=SAMPLE_RATES,
    realtime_pacing_required=True,
    chunk_ms_max=_MAX_FRAME_MS,
)


def language_bias(language: str | None, provider_params: Mapping) -> list[str]:
    """Merge the unified `language` hint with any explicit `languageBias`."""
    names: list[str] = []
    raw = provider_params.get("languageBias") or []
    for item in raw if isinstance(raw, (list, tuple)) else str(raw).split(","):
        name = str(item).strip()
        if name and name not in names:
            names.append(name)
    if language and language != "auto":
        primary = language.split("-", 1)[0].lower()
        name = LANGUAGE_NAMES.get(primary)
        if name is None:
            logger.warning(
                "meta language is not one of the 25 documented languages; forwarded verbatim",
                extra={"provider": "meta", "language": language},
            )
            name = language
        if name not in names:
            names.append(name)
    return names


def select_mode(config: STTConfig) -> str:
    """diarization=true wins; otherwise provider_params.mode, else ENDPOINTING."""
    if config.diarization:
        return MODE_DIARIZATION
    requested = str(config.provider_params.get("mode") or DEFAULT_MODE).upper()
    if requested not in MODES:
        raise ProviderStreamError(
            f"meta mode must be one of {', '.join(sorted(MODES))} (got '{requested}')",
            recoverable=False,
            provider="meta",
        )
    return requested


def build_handshake(config: STTConfig, api_key: str) -> dict:
    """The first JSON text frame. Raises on audio the wire cannot carry."""
    if config.encoding != "linear16" or config.channels != 1:
        raise ProviderStreamError(
            "meta accepts linear16 mono only "
            f"(got {config.encoding}, {config.channels} channel(s))",
            recoverable=False,
            provider="meta",
        )
    audio_encoding = _AUDIO_ENCODINGS.get(config.sample_rate)
    if audio_encoding is None:
        raise ProviderStreamError(
            f"meta accepts sample_rate 16000 or 24000 only (got {config.sample_rate})",
            recoverable=False,
            provider="meta",
        )
    handshake: dict = {
        "authorization": {"accessToken": f"Bearer {api_key}"},
        "audioEncoding": audio_encoding,
        "model": config.model,
        "mode": select_mode(config),
        "partialMode": "CUMULATIVE",
        "emitAudioProgress": False,
    }
    keywords = list(config.keyterms)
    extra_keywords = config.provider_params.get("keywords") or []
    for term in extra_keywords if isinstance(extra_keywords, (list, tuple)) else [extra_keywords]:
        term = str(term).strip()
        if term and term not in keywords:
            keywords.append(term)
    if keywords:
        handshake["keywords"] = keywords
    bias = language_bias(config.language, config.provider_params)
    if bias:
        handshake["languageBias"] = bias
    for key, value in config.provider_params.items():
        if key in RESERVED_HANDSHAKE_KEYS:
            logger.warning(
                "meta provider_params key is reserved by the adapter; ignored",
                extra={"provider": "meta", "param": key},
            )
            continue
        if key in {"mode", "keywords", "languageBias", "sessionId"}:
            continue  # already folded in above, or a query param
        handshake[key] = value
    return handshake


def build_url(config: STTConfig, base: str = WS_URL) -> str:
    """sessionId is the one query parameter; it is a log-correlation id."""
    session_id = config.provider_params.get("sessionId")
    if session_id:
        return f"{base}?{urlencode({'sessionId': str(session_id)})}"
    return base


def redact(text: str, api_key: str) -> str:
    """The key rides in the handshake frame; keep it out of anything logged."""
    return text.replace(api_key, "***") if api_key else text


def speaker_index(label: str, seen: dict[str, int]) -> int:
    """Session-scoped label -> 0-based index, in order of first appearance."""
    if label not in seen:
        seen[label] = len(seen)
    return seen[label]


class TurnState:
    """Turn bookkeeping for one upstream session. Needs no socket, so fixture
    tests drive process() directly with documented wire frames."""

    def __init__(self, mode: str = DEFAULT_MODE, include_raw: bool = False):
        self.mode = mode
        self.include_raw = include_raw
        self.current_turn: int | None = None  # partials carry no turnId
        self.turn_start: dict[int, float] = {}
        self.turn_end: dict[int, float] = {}
        self.turn_speaker: dict[int, str] = {}
        self.speakers: dict[str, int] = {}
        self.error_message: str | None = None
        self._ptt_final_seen = False

    def process(self, msg: dict) -> list[STTEvent]:
        kind = msg.get("type")
        at = float(msg.get("audioProcessedMs") or 0) / 1000.0
        raw = msg if self.include_raw else None

        if kind == "speechStart":
            turn_id = int(msg.get("turnId", 0))
            self.current_turn = turn_id
            self.turn_start[turn_id] = at
            return [SpeechStarted(type="speech_started", at=at)]

        if kind == "transcript":
            text = msg.get("transcript", "")
            if not text.strip():
                return []
            # In the turn modes the post-processed text arrives in
            # speechComplete; a `final: true` here would be deduplicated
            # against it by the session layer and the clean text lost.
            is_final = bool(msg.get("final")) and self.mode == MODE_PUSH_TO_TALK
            if is_final:
                self._ptt_final_seen = True
            start = None if self.current_turn is None else self.turn_start.get(self.current_turn)
            return [
                Transcript(
                    type="transcript",
                    is_final=is_final,
                    text=text,
                    words=None,
                    start=start,
                    end=at,
                    provider_raw=raw,
                )
            ]

        if kind == "speaker":
            # Labels the span behind it; exactly one per turn in DIARIZATION.
            if self.current_turn is not None:
                self.turn_speaker[self.current_turn] = str(msg.get("label", ""))
            return []

        if kind == "speechEnd":
            turn_id = int(msg.get("turnId", 0))
            self.turn_end[turn_id] = at
            return [UtteranceEnd(type="utterance_end", at=at)]

        if kind == "speechComplete":
            turn_id = int(msg.get("turnId", 0))
            start = self.turn_start.pop(turn_id, None)
            end = self.turn_end.pop(turn_id, at)
            label = self.turn_speaker.pop(turn_id, None)
            text = msg.get("transcript", "")
            if self.mode == MODE_PUSH_TO_TALK and self._ptt_final_seen:
                return []  # the `final: true` transcript already carried it
            if not text.strip():
                return []  # a turn the stream ended part-way through
            words = None
            if label:
                words = [
                    Word(
                        w=text,
                        start=end if start is None else start,
                        end=end,
                        speaker=speaker_index(label, self.speakers),
                    )
                ]
            return [
                Transcript(
                    type="transcript",
                    is_final=True,
                    text=text,
                    words=words,
                    start=start,
                    end=end,
                    provider_raw=raw,
                )
            ]

        if kind == "error":
            # Always followed by a close frame; the close code decides
            # recoverability, the message goes into the error text.
            self.error_message = str(msg.get("message") or "unknown error")
            return []

        # audioProgress, the typeless handshake ack, and anything additive.
        return []


def parse_message(raw: str, state: TurnState) -> list[STTEvent]:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(msg, dict):
        return []
    return state.process(msg)


@register_stt_stream("meta", capabilities=CAPABILITIES)
def build(settings: Settings) -> "MetaSTTStream":
    if not settings.meta_api_key:
        raise ProviderNotConfigured("meta")
    return MetaSTTStream(settings.meta_api_key)


class MetaSTTStream(STTStreamProvider):
    name = "meta"
    capabilities = CAPABILITIES

    def __init__(self, api_key: str, *, ws_url: str = WS_URL):
        self._api_key = api_key
        self._ws_url = ws_url
        self._ws: websockets.ClientConnection | None = None
        self._state = TurnState()
        self._finished = False
        self._closed = False
        self._session_id = ""
        # Ingest pacing state, see _PACING.
        self._frame_bytes = 0
        self._byte_rate = 0
        self._next_send = 0.0

    async def connect(self, config: STTConfig) -> None:
        handshake = build_handshake(config, self._api_key)  # validates audio params
        self._state = TurnState(mode=handshake["mode"], include_raw=config.include_raw)
        self._byte_rate = (
            bytes_per_second(config.encoding, config.sample_rate, config.channels) or 0
        )
        self._frame_bytes = int(self._byte_rate * _MAX_FRAME_MS / 1000)
        self._next_send = 0.0
        url = build_url(config, self._ws_url)
        try:
            self._ws = await ws_connect(url)
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
            raise ProviderStreamError(
                f"meta connect rejected ({status})",
                recoverable=status not in (401, 403),
                provider=self.name,
                code=str(status),
            ) from exc
        except Exception as exc:
            raise ProviderStreamError(
                f"meta connect failed: {redact(str(exc), self._api_key)}",
                recoverable=True,
                provider=self.name,
            ) from exc
        await self._handshake(handshake)
        logger.info(
            "meta connected",
            extra={"provider": self.name, "model": config.model, "mode": handshake["mode"],
                   "upstream_session": self._session_id},
        )

    async def _handshake(self, handshake: dict) -> None:
        assert self._ws is not None
        try:
            await self._ws.send(json.dumps(handshake))
            raw = await asyncio.wait_for(self._ws.recv(), HANDSHAKE_TIMEOUT)
        except websockets.exceptions.ConnectionClosed as exc:
            await self.close()
            raise self._closed_error(exc.code, exc.reason, "handshake") from exc
        except TimeoutError as exc:
            await self.close()
            raise ProviderStreamError(
                "meta handshake ack timed out", recoverable=True, provider=self.name,
                code="timeout",
            ) from exc
        except Exception as exc:
            await self.close()
            raise ProviderStreamError(
                f"meta handshake failed: {redact(str(exc), self._api_key)}",
                recoverable=True,
                provider=self.name,
            ) from exc
        ack = json.loads(raw) if isinstance(raw, str) else {}
        if isinstance(ack, dict) and "sessionId" in ack and "type" not in ack:
            self._session_id = str(ack["sessionId"])
            return
        message = ack.get("message") if isinstance(ack, dict) else None
        # A rejected handshake is followed by a close; wait briefly for its
        # code so a 1013 (rate limited) can still be retried.
        code = await self._await_close_code()
        await self.close()
        raise ProviderStreamError(
            f"meta handshake rejected: {message or 'no session id in ack'}",
            recoverable=code in (1011, _CLOSE_RATE_LIMITED),
            provider=self.name,
            code=str(code or ""),
        )

    async def _await_close_code(self) -> int | None:
        assert self._ws is not None
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._ws.wait_closed(), 2.0)
        return getattr(self._ws, "close_code", None)

    def _closed_error(self, code: int, reason: str, phase: str) -> ProviderStreamError:
        detail = self._state.error_message or reason or ""
        return ProviderStreamError(
            f"meta {phase} closed {code}: {detail}".rstrip(": "),
            # 1008 = the request itself is wrong (config, key, pacing);
            # 1011 / 1013 / anything else = retry with backoff.
            recoverable=code != _CLOSE_INVALID_REQUEST,
            provider=self.name,
            code=str(code),
        )

    async def send_audio(self, chunk: bytes) -> None:
        """Split oversized chunks and forward under the pacing bucket (_PACING).

        Frame boundaries carry no meaning upstream, so nothing is buffered;
        blocking here is how backpressure reaches the client socket.
        """
        if self._ws is None:
            raise ProviderStreamError("send before connect", recoverable=False, provider=self.name)
        if self._frame_bytes <= 0:
            await self._ws.send(chunk)
            return
        for offset in range(0, len(chunk), self._frame_bytes):
            await self._send_paced(chunk[offset : offset + self._frame_bytes])

    async def _send_paced(self, frame: bytes) -> None:
        assert self._ws is not None
        if self._byte_rate > 0:
            loop = asyncio.get_running_loop()
            now = loop.time()
            floor = now - _MAX_BURST_SECONDS
            if self._next_send < floor:  # cap the credit banked while idle
                self._next_send = floor
            if self._next_send > now:
                await asyncio.sleep(self._next_send - now)
            self._next_send += len(frame) / self._byte_rate / _MAX_REALTIME_FACTOR
        await self._ws.send(frame)

    async def events(self) -> AsyncIterator[STTEvent]:
        if self._ws is None:
            raise ProviderStreamError(
                "events before connect", recoverable=False, provider=self.name
            )
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                for event in parse_message(raw, self._state):
                    yield event
        except websockets.exceptions.ConnectionClosedOK as exc:
            if self._state.error_message:
                raise self._closed_error(exc.code, exc.reason, "session") from exc
            return
        except websockets.exceptions.ConnectionClosed as exc:
            if self._finished and not self._state.error_message:
                return
            raise self._closed_error(exc.code, exc.reason, "session") from exc
        if self._state.error_message:
            # Error frame with no close frame behind it: still a failure.
            raise ProviderStreamError(
                f"meta session error: {self._state.error_message}",
                recoverable=True,
                provider=self.name,
            )

    async def finish(self) -> None:
        """Half-close input; the server flushes pending results and closes 1000."""
        if self._finished or self._ws is None:
            return
        self._finished = True
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await self._ws.send(json.dumps({"type": "endStream"}))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

"""Palabra streaming STT adapter (raw WebSocket, /asr/v1/speech-to-text/stream).

Protocol facts (docs/providers/palabra.md, verified 2026-08-13 against the
docs and the official SDK palabra-ai-python v2.1.0):
- Auth: API key as the `token` query param (an `Authorization` header also
  works). NO session brokering: the server creates the streaming session for
  the lifetime of the connection and reaps it when the socket ends, so
  close() has nothing to delete upstream.
- All config is URL query params. There is NO config frame after connect.
- Endpoints are per-region; STT exists only in `eu` today. Settings.palabra_ws_base
  (SPEECHROUTER_PALABRA_WS_BASE) overrides the region table with a full stream
  URL, e.g. to point a gateway at a non-production environment.
- Up: raw PCM as binary frames, 320ms chunks, paced to realtime (the wire
  carries AUDIO_STREAM_TOO_FAST/TOO_SLOW/STALLED warnings). Both are enforced
  HERE — client framing is not forwarded verbatim. See _PACING.
- Down: JSON text frames keyed by `message_type`:
    {"message_type":"transcription","transcription_id":"..","language":"en",
     "is_eos":false,"segment":{"text":"..","start_time":0.32,"end_time":1.84},
     "delta":{...}}
  `is_eos` is the finality flag. Times are SECONDS already.
- **segment.text is authoritative; `delta` is ignored.** Delta semantics flip
  with enable_filler_filter (off: append-only; on: the tail may be rewritten
  mid-segment). The filter defaults ON, so reading the whole segment every
  time is the only reading correct under both.
- No word timestamps, no confidence, no speaker labels -> words=None and
  Capabilities.diarization False so the resolver rejects diarization before
  a socket is opened.
- `translated_transcription` frames appear only when the caller opts in via
  provider_params.translate_languages. They are delivered as ordinary
  Transcripts tagged with the TARGET language in `lang`. See _TRANSLATION.
- Errors: 401/409 land on the HTTP upgrade, not as frames. After a successful
  upgrade the docs promise no application-level error frames — the server just
  closes. The SDK's parser nonetheless knows `error`/`warning` messages, so we
  handle them defensively and still classify on the close code.
- Liveness is standard WS ping/pong (the SDK dials ping_interval=10), NOT an
  app-level keepalive frame. Do not copy Telnyx's ping_interval=None here.
- Billing: $0.002 per minute of audio processed (AUDIO_TIME).

_TRANSLATION: a translated frame is emitted as an ordinary Transcript tagged
with the TARGET language in `lang`. These only exist when the caller passed
provider_params.translate_languages, so nobody gets them unasked, and `lang`
keeps them separable from the source stream — the Soniox model, where one
stream carries both and the client picks.

Live-verified 2026-08-13: a translated final carries the SAME `end` timestamp
as the source final it belongs to, and it arrives AFTER that source final.
router/session.py enforces the spec's failover-dedup guarantee by dropping any
final whose end does not advance past the previous one, so a verbatim
translated final is swallowed one layer above this adapter and the client sees
nothing. Rather than weaken that invariant for one provider, this adapter
advances the translated final's `end` by _TRANSLATION_END_NUDGE per target
language (1ms, 2ms, ... in the order the caller listed them in
translate_languages). Consequences, all deliberate:
- the offset is deterministic — same target list, same slot — so a replay
  after failover produces the SAME nudged end and is still deduped;
- it is applied to FINALS only, since partials never reach the dedup gate;
- it is a fabrication of at most a few ms on a segment-level timestamp that
  is, by construction, a copy of the source segment's.
The honest fix is a dedicated translation event in packages/spec; this keeps
translation deliverable until that lands, without touching the session engine.

_PACING: the wire wants 320ms frames paced to realtime, and nothing above the
adapter re-chunks or paces (Capabilities.chunk_ms_*/realtime_pacing_required
are declarative — the session engine never reads them). send_audio therefore
buffers into exact CHUNK_MS frames and runs them through a token bucket. The
bucket is not pinned to 1x realtime: after a failover the session replays up
to ring_buffer_seconds of audio into a fresh adapter, and a strictly-realtime
adapter could never work that backlog off. _MAX_REALTIME_FACTOR caps how fast
the backlog drains, _MAX_BURST_SECONDS caps the credit a quiet stretch banks.
Whether Palabra drops or merely warns above realtime is still open
(docs/providers/palabra.md #8), so the ceiling stays low and
AUDIO_STREAM_TOO_FAST is surfaced in the logs.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Mapping
from urllib.parse import urlencode, urlsplit

import websockets

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

# Region -> ASR endpoint. Only `eu` serves STT today (SDK REGIONS table);
# kept as a map so adding `us` later is a one-line change.
ASR_ENDPOINTS = {"eu": "wss://stream.palabra.ai/asr/v1/speech-to-text/stream"}
DEFAULT_REGION = "eu"
WS_BASE = ASR_ENDPOINTS[DEFAULT_REGION]

# WS-level liveness, matching the official SDK. Palabra has no keepalive frame.
PING_INTERVAL = 10
PING_TIMEOUT = 30

_ENCODING_MAP = {
    "linear16": "pcm_s16le",
    "linear32": "pcm_s32le",
    "mulaw": "mulaw",
    "alaw": "alaw",
}

_TRANSCRIPT_TYPES = frozenset({"transcription", "translated_transcription"})
_TRANSLATED_TYPE = "translated_transcription"

# Digital silence differs per codec: zero bytes are silence for linear PCM but
# a loud buzz in mulaw/alaw, where the silent code is 0xFF / 0xD5.
_SILENCE_BYTE = {"linear16": b"\x00", "linear32": b"\x00", "mulaw": b"\xff", "alaw": b"\xd5"}
_BYTES_PER_SAMPLE = {"linear16": 2, "linear32": 4, "mulaw": 1, "alaw": 1}
CHUNK_MS = 320  # vendor-recommended chunk length

# Query keys this adapter owns. urlencode would happily emit a SECOND copy of
# any of them from provider_params, leaving the server to choose — and a
# provider_params `sample_rate` disagreeing with the audio actually sent is
# silent corruption, not a passthrough feature.
RESERVED_QUERY_PARAMS = frozenset({"token", "format", "sample_rate", "language"})

# See _TRANSLATION. Must survive the session layer's round(_, 3) on timestamps,
# so 1ms is the smallest usable step.
_TRANSLATION_END_NUDGE = 0.001

# Ingest pacing, see _PACING.
_MAX_REALTIME_FACTOR = 4.0  # ingest ceiling, in multiples of realtime
_MAX_BURST_SECONDS = 2.0  # how much unspent pacing credit may bank

CAPABILITIES = Capabilities(
    streaming=True,
    interim_results=True,  # is_eos=false partials
    word_timestamps=False,  # segment-level times only
    diarization=False,
    endpointing=False,  # is_eos is a segment edge; unverified as a VAD edge
    keyterms=False,  # hotword glossaries exist, but only on the S2S pipeline
    # Deliberately no language list: the server owns it and refuses unknown
    # codes on the upgrade with HTTP 400, which connect() forwards verbatim.
    languages=frozenset({"auto"}),
    encodings=frozenset(_ENCODING_MAP),
    sample_rates=frozenset(),  # flexible; server assumes 16000 when omitted
    realtime_pacing_required=True,
    chunk_ms_min=320,
    chunk_ms_max=320,
)


def build_url(config: STTConfig, api_key: str, base: str = WS_BASE) -> str:
    """Build the WebSocket URL. All session config travels as query params."""
    params: list[tuple[str, str]] = [
        ("token", api_key),
        ("format", _ENCODING_MAP[config.encoding]),
        ("sample_rate", str(config.sample_rate)),
    ]
    if config.language and config.language != "auto":
        params.append(("language", config.language))
    for key, value in config.provider_params.items():
        if key in RESERVED_QUERY_PARAMS:
            logger.warning(
                "palabra provider_params key is reserved by the adapter; ignored",
                extra={"provider": "palabra", "param": key},
            )
            continue
        params.append((key, _query_value(value)))
    return f"{base}?{urlencode(params)}"


def _query_value(value) -> str:
    """provider_params arrive as JSON, query params are strings."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _response_reason(exc: "websockets.exceptions.InvalidStatus", api_key: str) -> str:
    """The upgrade response body as one redacted line, or the status alone."""
    body = getattr(exc.response, "body", None) or b""  # bytes or bytearray in practice
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", "replace")
    else:
        text = str(body)
    text = " ".join(text.split())[:300]
    return redact(text, api_key) or f"status {exc.response.status_code}"


def redact(text: str, api_key: str) -> str:
    """Strip the API key out of anything headed for a log or a client.

    The key rides in the query string, so an exception that echoes the URL
    back (websockets' InvalidURI does) would otherwise leak it.
    """
    return text.replace(api_key, "***") if api_key else text


def translation_nudges(provider_params: Mapping) -> dict[str, float]:
    """Per-target `end` offsets keyed by target language. See _TRANSLATION.

    Slot order follows the caller's own translate_languages list, so the map
    is identical in every adapter instance of the session — which is what
    keeps failover replays deduplicated.
    """
    raw = provider_params.get("translate_languages") or ""
    targets = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
    nudges: dict[str, float] = {}
    for lang in (str(t).strip() for t in targets):
        if lang and lang.lower() not in nudges:
            nudges[lang.lower()] = _TRANSLATION_END_NUDGE * (len(nudges) + 1)
    return nudges


def _silence_chunk(config: STTConfig) -> bytes:
    """One CHUNK_MS chunk of digital silence in the session's own encoding."""
    width = _BYTES_PER_SAMPLE[config.encoding]
    samples = int(config.sample_rate * CHUNK_MS / 1000) * config.channels
    return _SILENCE_BYTE[config.encoding] * (samples * width)


def parse_message(
    raw: str,
    include_raw: bool = False,
    translation_nudge: Mapping[str, float] | None = None,
) -> list[STTEvent]:
    """Pure translation of one Palabra JSON frame into normalized events.

    Side-effect free so fixture tests can drive it without a socket.
    """
    msg = json.loads(raw)
    kind = msg.get("message_type", "")

    # Documented as absent after a successful upgrade, but the official SDK
    # parses them, so a swallowed error would mean a silent session.
    if kind == "error":
        data = msg.get("data") or msg
        raise ProviderStreamError(
            f"palabra error: {data.get('desc') or data.get('message') or 'unknown'}",
            recoverable=False,
            provider="palabra",
            code=str(data.get("code", "")),
        )
    if kind == "warning":
        data = msg.get("data") or msg
        # AUDIO_STREAM_TOO_FAST / TOO_SLOW / STALLED — pacing complaints, not
        # stream-fatal. Surface in logs; the stream keeps running.
        logger.warning(
            "palabra warning",
            extra={"provider": "palabra", "code": data.get("code", "")},
        )
        return []

    if kind not in _TRANSCRIPT_TYPES:
        return []

    segment = msg.get("segment") or {}
    text = segment.get("text", "")
    if not text.strip():
        return []

    # For translated frames this is the TARGET language — the tag that keeps
    # them separable from the source stream.
    lang = msg.get("language") or None
    is_final = bool(msg.get("is_eos", False))
    end = segment.get("end_time")
    if kind == _TRANSLATED_TYPE and is_final and end is not None:
        # See _TRANSLATION: a verbatim copy of the source final's end would be
        # dropped by the session layer's dedup gate.
        nudges = translation_nudge or {}
        end = round(end + nudges.get((lang or "").lower(), _TRANSLATION_END_NUDGE), 4)

    return [
        Transcript(
            type="transcript",
            is_final=is_final,
            text=text,
            words=None,  # segment-level times only; no word timestamps
            start=segment.get("start_time"),
            end=end,
            lang=lang,
            provider_raw=msg if include_raw else None,
        )
    ]


@register_stt_stream("palabra", capabilities=CAPABILITIES)
def build(settings: Settings) -> "PalabraSTTStream":
    if not settings.palabra_api_key:
        raise ProviderNotConfigured("palabra")
    return PalabraSTTStream(
        settings.palabra_api_key,
        region=settings.palabra_region,
        ws_base=settings.palabra_ws_base,
    )


class PalabraSTTStream(STTStreamProvider):
    name = "palabra"
    capabilities = CAPABILITIES

    # The STT lane has no EOS message (the S2S lane's end_task/eos_timeout
    # does not exist here), so the only way to make the recognizer emit the
    # last utterance is to feed it silence until its endpointer fires.
    #
    # Live-verified 2026-08-13: streaming a file and simply closing LOSES the
    # tail — the last utterance stayed a partial forever ("...that
    # transcripts", dropping "arrive correctly") and the server sat idle for
    # 10s before closing with 1001. Padding with silence produced the
    # complete final 1.03s later. These two constants are that measurement
    # plus headroom.
    _SILENCE_FLUSH_SECONDS = 1.5
    _FINISH_GRACE_SECONDS = 1.0

    def __init__(self, api_key: str, *, region: str = DEFAULT_REGION, ws_base: str = ""):
        self._api_key = api_key
        if ws_base:
            self._ws_base = ws_base
        elif region in ASR_ENDPOINTS:
            self._ws_base = ASR_ENDPOINTS[region]
        else:
            # ProviderNotConfigured is what the resolver turns into a clean
            # invalid_request instead of a mid-connect failure; its message is
            # generic, so the operator-facing detail goes to the log.
            logger.error(
                "palabra serves no STT endpoint in the configured region",
                extra={
                    "provider": "palabra",
                    "region": region,
                    "available": ",".join(sorted(ASR_ENDPOINTS)),
                },
            )
            raise ProviderNotConfigured("palabra")
        self._ws: websockets.ClientConnection | None = None
        self._finished = False
        self._closed = False
        self._include_raw = False
        self._silence_chunk = b""
        self._translation_nudge: dict[str, float] = {}
        # Ingest re-chunking / pacing state, see _PACING.
        self._chunk_bytes = 0
        self._byte_rate = 0
        self._pending = bytearray()
        self._next_send = 0.0  # event-loop clock deadline for the next frame

    async def connect(self, config: STTConfig) -> None:
        # The wire carries no channel count and the server assumes mono, so
        # interleaved stereo would come back as garbage at double speed.
        # Refuse rather than transcribe nonsense.
        if config.channels != 1:
            raise ProviderStreamError(
                f"palabra accepts mono audio only (got {config.channels} channels)",
                recoverable=False,
                provider=self.name,
            )
        self._include_raw = config.include_raw
        self._translation_nudge = translation_nudges(config.provider_params)
        self._silence_chunk = _silence_chunk(config)
        self._chunk_bytes = len(self._silence_chunk)
        self._byte_rate = (
            bytes_per_second(config.encoding, config.sample_rate, config.channels) or 0
        )
        self._pending = bytearray()
        self._next_send = 0.0
        url = build_url(config, self._api_key, self._ws_base)
        endpoint = urlsplit(self._ws_base).netloc  # host only, never the token
        try:
            self._ws = await ws_connect(
                url,
                ping_interval=PING_INTERVAL,  # WS pings are Palabra's liveness
                ping_timeout=PING_TIMEOUT,
            )
        except websockets.exceptions.InvalidStatus as exc:
            status = exc.response.status_code
            if status == 400:
                # The server validated OUR query and refused it — today that is
                # an unsupported `language` code, and the body says so
                # ("unsupported language code \"xx\"; one of: ar, de, ..."). It is
                # the caller's request that is wrong, so the reason travels
                # under invalid_request, which router/session.py forwards to
                # the client verbatim instead of the masked outage text.
                # Retrying Palabra cannot help; fallbacks, if any, still run.
                raise ProviderStreamError(
                    f"palabra rejected the request ({status}): "
                    f"{_response_reason(exc, self._api_key)}",
                    recoverable=False,
                    provider=self.name,
                    code="invalid_request",
                ) from exc
            # 401 = bad key: retrying or failing over to another Palabra
            # session cannot help. 409 = a session is already live for this
            # identity, which a retry after backoff may clear.
            raise ProviderStreamError(
                f"palabra connect rejected ({status}) by {endpoint}",
                recoverable=status != 401,
                provider=self.name,
                code=str(status),
            ) from exc
        except Exception as exc:
            # redact: the key is a query param and some websockets exceptions
            # (InvalidURI) quote the URL back at us.
            raise ProviderStreamError(
                f"palabra connect failed at {endpoint}: {redact(str(exc), self._api_key)}",
                recoverable=True,
                provider=self.name,
            ) from exc
        logger.info(
            "palabra connected",
            # host only: the token rides in the query string and must not be logged
            extra={
                "provider": self.name,
                "model": config.model,
                "endpoint": endpoint,
            },
        )

    async def send_audio(self, chunk: bytes) -> None:
        """Re-chunk to CHUNK_MS and forward under the pacing bucket (_PACING).

        Blocking here is the contract's way of pushing backpressure back at
        the client socket.
        """
        if self._ws is None:
            raise ProviderStreamError("send before connect", recoverable=False, provider=self.name)
        if self._chunk_bytes <= 0:  # no encoding known (never connected): pass through
            await self._ws.send(chunk)
            return
        self._pending.extend(chunk)
        while len(self._pending) >= self._chunk_bytes:
            frame = bytes(self._pending[: self._chunk_bytes])
            del self._pending[: self._chunk_bytes]
            await self._send_paced(frame)

    async def _send_paced(self, frame: bytes) -> None:
        """Token bucket: never faster than _MAX_REALTIME_FACTOR x realtime,
        with at most _MAX_BURST_SECONDS of banked credit."""
        assert self._ws is not None
        if self._byte_rate > 0:
            loop = asyncio.get_running_loop()
            now = loop.time()
            floor = now - _MAX_BURST_SECONDS
            if self._next_send < floor:  # cap credit banked while idle or live
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
                for event in parse_message(raw, self._include_raw, self._translation_nudge):
                    yield event
        except websockets.exceptions.ConnectionClosedOK:
            pass
        except websockets.exceptions.ConnectionClosed as exc:
            if self._finished:
                return
            raise ProviderStreamError(
                f"palabra closed {exc.code}: {exc.reason}",
                recoverable=exc.code != 1008,  # 1008 = policy violation
                provider=self.name,
                code=str(exc.code),
            ) from exc

    async def finish(self) -> None:
        """Flush by feeding silence, then close.

        Closing the socket outright drops whatever utterance was still in
        flight, so pad with digital silence until Palabra's own endpointer
        fires and emits the final, then give it a moment to arrive.
        """
        if self._finished or self._ws is None:
            return
        self._finished = True
        try:
            if self._pending:  # sub-chunk tail left over by the re-chunker
                tail = bytes(self._pending)
                self._pending.clear()
                await self._send_paced(tail)
            if self._silence_chunk:
                chunks = max(1, int(self._SILENCE_FLUSH_SECONDS * 1000 / CHUNK_MS))
                for _ in range(chunks):
                    await self._send_paced(self._silence_chunk)
        except websockets.exceptions.ConnectionClosed:
            pass  # server already hung up: nothing left to flush into
        await asyncio.sleep(self._FINISH_GRACE_SECONDS)
        # events() treats a close at this point (finished=True) as a clean end
        # of stream, whether it arrives as ConnectionClosedOK or not.
        with contextlib.suppress(Exception):
            await self._ws.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - teardown must never raise
                pass

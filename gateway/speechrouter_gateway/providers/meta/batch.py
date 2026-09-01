"""Meta Muse Voice Transcribe batch adapter (POST /v1/asr/transcribe).

Facts (docs/providers/meta.md):
- multipart with two parts: `request` (JSON settings) and `audio` (the clip).
- The clip must be a RIFF/WAVE container of mono 16-bit PCM at 16 or 24 kHz,
  at most 32 MB and 10 minutes. Nothing is transcoded here: a header that
  clearly violates that is refused before the upload with a message that
  says what to convert to; anything the stdlib cannot parse is left for the
  server to judge.
- Turn-level timestamps only (`turns[].startMs/endMs`), populated in the
  ENDPOINTING and DIARIZATION modes; PUSH_TO_TALK returns no turns. Turns
  surface as segment-level Words (one per turn, speaker index attached in
  DIARIZATION) so srt/vtt output and diarization work through the existing
  formatters — the same convention openai_compat uses for diarized segments.
- Errors are an HTTP status plus one message; no error code field.
"""

import io
import json
import wave

import httpx

from ...config import Settings
from ...protocol import Transcript, Word
from ..base import Capabilities, ProviderStreamError, STTBatchProvider, STTConfig
from ..registry import ProviderNotConfigured, register_stt_batch
from .adapter import LANGUAGE_NAMES, SAMPLE_RATES, language_bias, select_mode, speaker_index

BASE_URL = "https://api.meta.ai/v1/asr/transcribe"

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_AUDIO_SECONDS = 600

# Request keys the adapter owns. partialMode / emitAudioProgress only affect
# the SSE response shape, which this adapter never asks for.
RESERVED_REQUEST_KEYS = frozenset({"model", "audioEncoding", "partialMode", "emitAudioProgress"})

CAPABILITIES = Capabilities(
    batch=True,
    word_timestamps=False,  # turn-level only
    diarization=True,
    keyterms=True,
    languages=frozenset({"auto", *LANGUAGE_NAMES}),
)


def check_wav(audio: bytes) -> None:
    """Refuse audio the endpoint documents as unsupported, before uploading.

    Only a header the stdlib can read is judged; a container it cannot parse
    goes upstream, where the server's own 400 is the authority.
    """
    if len(audio) > MAX_UPLOAD_BYTES:
        raise ProviderStreamError(
            f"meta batch accepts at most {MAX_UPLOAD_BYTES // (1024 * 1024)} MB of audio",
            recoverable=False,
            provider="meta",
            code="413",
        )
    try:
        with wave.open(io.BytesIO(audio), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.getnframes()
    except (wave.Error, EOFError):
        return  # not a WAV the stdlib understands; let the server decide
    problems: list[str] = []
    if channels != 1:
        problems.append(f"{channels} channels")
    if width != 2:
        problems.append(f"{width * 8}-bit samples")
    if rate not in SAMPLE_RATES:
        problems.append(f"{rate} Hz")
    if problems:
        raise ProviderStreamError(
            "meta batch accepts mono 16-bit PCM WAV at 16 kHz or 24 kHz only "
            f"(got {', '.join(problems)}); convert with "
            "`ffmpeg -i in -ac 1 -ar 24000 -c:a pcm_s16le out.wav`",
            recoverable=False,
            provider="meta",
            code="400",
        )
    if rate and frames / rate > MAX_AUDIO_SECONDS:
        raise ProviderStreamError(
            f"meta batch accepts at most {MAX_AUDIO_SECONDS // 60} minutes of audio per request",
            recoverable=False,
            provider="meta",
            code="400",
        )


def build_request(config: STTConfig) -> dict:
    """The JSON `request` part. Mirrors the realtime handshake minus auth."""
    request: dict = {
        "model": config.model,
        "audioEncoding": "WAV",
        "mode": select_mode(config),
    }
    keywords = list(config.keyterms)
    extra = config.provider_params.get("keywords") or []
    for term in extra if isinstance(extra, (list, tuple)) else [extra]:
        term = str(term).strip()
        if term and term not in keywords:
            keywords.append(term)
    if keywords:
        request["keywords"] = keywords
    bias = language_bias(config.language, config.provider_params)
    if bias:
        request["languageBias"] = bias
    for key, value in config.provider_params.items():
        if key in RESERVED_REQUEST_KEYS or key in {"mode", "keywords", "languageBias", "sessionId"}:
            continue
        request[key] = value
    return request


def parse_response(payload: dict, include_raw: bool = False) -> Transcript:
    speakers: dict[str, int] = {}
    words: list[Word] = []
    for turn in payload.get("turns") or []:
        text = str(turn.get("transcript") or "").strip()
        if not text or turn.get("startMs") is None or turn.get("endMs") is None:
            continue  # a turn the clip ended part-way through
        label = turn.get("speaker")
        words.append(
            Word(
                w=text,
                start=float(turn["startMs"]) / 1000.0,
                end=float(turn["endMs"]) / 1000.0,
                speaker=speaker_index(str(label), speakers) if label else None,
            )
        )
    duration = payload.get("audioDurationMs")
    return Transcript(
        type="transcript",
        is_final=True,
        text=payload.get("transcript", ""),
        words=words or None,
        start=0.0,
        end=float(duration) / 1000.0 if duration is not None else (
            words[-1].end if words else None
        ),
        provider_raw=payload if include_raw else None,
    )


@register_stt_batch("meta", capabilities=CAPABILITIES)
def build(settings: Settings) -> "MetaSTTBatch":
    if not settings.meta_api_key:
        raise ProviderNotConfigured("meta")
    return MetaSTTBatch(settings.meta_api_key)


class MetaSTTBatch(STTBatchProvider):
    name = "meta"
    capabilities = CAPABILITIES

    def __init__(self, api_key: str, base_url: str = BASE_URL):
        self._api_key = api_key
        self._base_url = base_url

    async def transcribe(self, audio: bytes, content_type: str, config: STTConfig) -> Transcript:
        check_wav(audio)
        request = build_request(config)
        params = {}
        if config.provider_params.get("sessionId"):
            params["sessionId"] = str(config.provider_params["sessionId"])
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    self._base_url,
                    params=params,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    files={
                        "request": (None, json.dumps(request), "application/json"),
                        "audio": ("audio.wav", audio, "audio/wav"),
                    },
                )
        except httpx.TimeoutException as exc:
            raise ProviderStreamError(
                "meta batch timed out", recoverable=True, provider=self.name, code="timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderStreamError(
                f"meta batch request failed: {exc}", recoverable=True, provider=self.name
            ) from exc
        if response.status_code != 200:
            raise ProviderStreamError(
                f"meta batch {response.status_code}: {response.text[:300]}",
                # 429 = tenant concurrency / hourly budget, 5xx = server budget:
                # both documented as retry-with-backoff. 4xx otherwise means
                # the request itself (key, WAV, length) is wrong.
                recoverable=response.status_code >= 500 or response.status_code == 429,
                provider=self.name,
                code=str(response.status_code),
            )
        return parse_response(response.json(), config.include_raw)

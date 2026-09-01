"""Gemini Live transcription fixtures — BidiGenerateContent frame shapes
captured from the real API on 2026-08-26 via scripts/gemini_probe.py and
written up in docs/providers/gemini.md. No socket required."""

import asyncio
import base64
import json

import pytest

from speechrouter_gateway.protocol import Transcript
from speechrouter_gateway.providers.base import ProviderStreamError, STTConfig
from speechrouter_gateway.providers.gemini.adapter import (
    MAX_AUDIO_BYTES,
    MAX_CUSTOM_VOCABULARY,
    GeminiSTTStream,
    audio_message,
    build_setup_message,
    language_codes,
    parse_server_message,
    redact,
    turn_ended,
)


def _config(**kw) -> STTConfig:
    base = {"model": "gemini-3.5-transcribe-live", "encoding": "linear16", "sample_rate": 16000}
    return STTConfig(**{**base, **kw})


def _server_content(**fields) -> dict:
    return {"serverContent": fields}


# --------------------------------------------------------------------- setup


def test_setup_names_the_model_and_asks_for_text():
    setup = build_setup_message(_config())["setup"]
    assert setup["model"] == "models/gemini-3.5-transcribe-live"
    assert setup["generationConfig"]["responseModalities"] == ["TEXT"]


def test_auto_language_is_an_empty_code_list():
    """`[]` is the API's own spelling of automatic detection — not a missing key."""
    assert language_codes(_config()) == []
    assert language_codes(_config(language="auto")) == []
    assert language_codes(_config(language="en-US")) == ["en-US"]
    setup = build_setup_message(_config())["setup"]
    assert setup["inputAudioTranscription"]["languageCodes"] == []


def test_keyterms_become_custom_vocabulary():
    setup = build_setup_message(_config(keyterms=("Kubernetes", "BigQuery")))["setup"]
    assert setup["inputAudioTranscription"]["customVocabulary"] == ["Kubernetes", "BigQuery"]


def test_custom_vocabulary_is_capped_at_the_vendor_limit():
    terms = tuple(f"term{i}" for i in range(MAX_CUSTOM_VOCABULARY + 50))
    setup = build_setup_message(_config(keyterms=terms))["setup"]
    assert len(setup["inputAudioTranscription"]["customVocabulary"]) == MAX_CUSTOM_VOCABULARY


def test_transcription_params_land_under_input_audio_transcription():
    setup = build_setup_message(
        _config(provider_params={"mode": "SMART", "languageCodes": ["en-US", "es-ES"]})
    )["setup"]
    transcription = setup["inputAudioTranscription"]
    assert transcription["mode"] == "SMART"
    # An explicit languageCodes overrides what `language` would have set.
    assert transcription["languageCodes"] == ["en-US", "es-ES"]
    assert "mode" not in setup


def test_other_params_land_at_the_top_level_of_setup():
    vad = {"automaticActivityDetection": {"silenceDurationMs": 250}}
    setup = build_setup_message(_config(provider_params={"realtimeInputConfig": vad}))["setup"]
    assert setup["realtimeInputConfig"] == vad


def test_adapter_owned_setup_keys_are_dropped():
    """model and sessionResumption drive rotation; a caller copy would break it."""
    setup = build_setup_message(
        _config(provider_params={"model": "models/other", "sessionResumption": {"handle": "x"}})
    )["setup"]
    assert setup["model"] == "models/gemini-3.5-transcribe-live"
    assert setup["sessionResumption"] == {}


def test_resumption_handle_rides_the_setup_on_reconnect():
    setup = build_setup_message(_config(), handle="h4nd13")["setup"]
    assert setup["sessionResumption"] == {"handle": "h4nd13"}


# --------------------------------------------------------------------- audio


def test_audio_is_base64_with_the_session_sample_rate_in_the_mime_type():
    msg = json.loads(audio_message(b"\x01\x02\x03\x04", 16000))
    blob = msg["realtimeInput"]["audio"]
    assert base64.b64decode(blob["data"]) == b"\x01\x02\x03\x04"
    assert blob["mimeType"] == "audio/pcm;rate=16000"
    assert json.loads(audio_message(b"\x00", 8000))["realtimeInput"]["audio"]["mimeType"] == (
        "audio/pcm;rate=8000"
    )


# -------------------------------------------------------------------- events


def test_interim_transcription_is_not_final():
    (t,) = parse_server_message(
        _server_content(interimInputTranscription={"text": "hello wor"})
    )
    assert isinstance(t, Transcript)
    assert t.is_final is False
    assert t.text == "hello wor"


def test_input_transcription_is_final():
    (t,) = parse_server_message(
        _server_content(inputTranscription={"text": "hello world."}, turnComplete=True)
    )
    assert t.is_final is True
    assert t.text == "hello world."


def test_no_timestamps_anywhere_on_the_wire():
    """The Live API reports no word, start or end times — see _NO_TIMES."""
    (t,) = parse_server_message(_server_content(inputTranscription={"text": "hi"}))
    assert t.words is None and t.start is None and t.end is None


def test_interim_precedes_final_when_one_frame_carries_both():
    events = parse_server_message(
        _server_content(
            interimInputTranscription={"text": "hello worl"},
            inputTranscription={"text": "Hello world."},
        )
    )
    assert [e.is_final for e in events] == [False, True]


def test_generation_complete_is_the_end_of_turn_signal():
    """Live-verified 2026-08-26: this model emits generationComplete and never
    the documented turnComplete. Watching only for turnComplete left finish()
    waiting out its full grace timeout on every stream."""
    assert turn_ended(_server_content(generationComplete=True)) is True
    assert turn_ended(_server_content(turnComplete=True)) is True
    assert turn_ended(_server_content(interimInputTranscription={"text": "hi"})) is False
    assert turn_ended(_server_content()) is False
    assert turn_ended({"setupComplete": {}}) is False


def test_blank_and_control_frames_yield_nothing():
    assert parse_server_message({"setupComplete": {}}) == []
    assert parse_server_message({"goAway": {"timeLeft": "5s"}}) == []
    assert parse_server_message({"sessionResumptionUpdate": {"newHandle": "h"}}) == []
    assert parse_server_message(_server_content(generationComplete=True)) == []
    # bare {"serverContent": {}} frames really do arrive on this wire
    assert parse_server_message(_server_content()) == []
    assert parse_server_message(_server_content(inputTranscription={"text": "   "})) == []


def test_language_tag_echoes_the_requested_language_only():
    """The wire reports no detected language; auto-detect stays untagged."""
    (t,) = parse_server_message(
        _server_content(inputTranscription={"text": "bonjour"}), False, "fr-FR"
    )
    assert t.lang == "fr-FR"
    (t,) = parse_server_message(_server_content(inputTranscription={"text": "bonjour"}))
    assert t.lang is None


def test_include_raw_attaches_the_whole_frame():
    frame = _server_content(inputTranscription={"text": "hi"})
    (t,) = parse_server_message(frame, True)
    assert t.provider_raw == frame


# -------------------------------------------------------------------- errors


@pytest.mark.parametrize("status", ["INVALID_ARGUMENT", "PERMISSION_DENIED", "NOT_FOUND"])
def test_request_rejecting_errors_are_unrecoverable(status):
    with pytest.raises(ProviderStreamError) as exc:
        parse_server_message({"error": {"code": 400, "message": "nope", "status": status}})
    assert exc.value.recoverable is False
    assert exc.value.code == status


def test_transient_errors_are_recoverable():
    with pytest.raises(ProviderStreamError) as exc:
        parse_server_message({"error": {"code": 503, "message": "busy",
                                        "status": "UNAVAILABLE"}})
    assert exc.value.recoverable is True


def test_api_key_is_stripped_from_error_text():
    assert redact("connect failed for key=sk_secret", "sk_secret") == "connect failed for key=***"
    assert redact("no key here", "") == "no key here"


# ------------------------------------------------------------------- ingest


class _FakeWS:
    def __init__(self):
        self.sent: list = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        pass


async def test_send_audio_splits_oversized_chunks():
    stream = GeminiSTTStream("k")
    ws = _FakeWS()
    stream._ws = ws
    stream._connected = True
    await stream.send_audio(b"\x00" * (MAX_AUDIO_BYTES * 2 + 10))
    assert len(ws.sent) == 3
    total = sum(
        len(base64.b64decode(json.loads(m)["realtimeInput"]["audio"]["data"])) for m in ws.sent
    )
    assert total == MAX_AUDIO_BYTES * 2 + 10


async def test_audio_is_buffered_not_dropped_while_the_socket_rotates():
    """A stall in send_audio would show up as a latency spike every nine
    minutes, so a rotating adapter buffers instead of blocking. See _ROTATION."""
    stream = GeminiSTTStream("k")
    ws = _FakeWS()
    stream._ws = ws
    stream._connected = True
    stream._rotating = True
    await stream.send_audio(b"\x11" * 100)
    assert ws.sent == []
    assert bytes(stream._pending) == b"\x11" * 100

    # ...and the buffer goes out on the new socket once the swap completes.
    new = _FakeWS()
    stream._ws = new
    stream._rotating = False
    buffered = bytes(stream._pending)
    stream._pending.clear()
    await stream._send_now(new, buffered)
    assert base64.b64decode(json.loads(new.sent[0])["realtimeInput"]["audio"]["data"]) == (
        b"\x11" * 100
    )


async def test_pending_buffer_is_capped_and_drops_the_oldest_audio():
    stream = GeminiSTTStream("k")
    stream._byte_rate = 32000  # 16kHz mono linear16
    stream._connected = True
    stream._rotating = True
    for _ in range(40):
        await stream.send_audio(b"\x00" * 32000)  # 40 seconds, cap is 30
    assert len(stream._pending) == int(30.0 * 32000)


async def test_send_before_connect_is_a_hard_error():
    stream = GeminiSTTStream("k")
    with pytest.raises(ProviderStreamError) as exc:
        await stream.send_audio(b"\x00")
    assert exc.value.recoverable is False


async def test_stereo_is_refused_rather_than_transcribed_as_garbage():
    stream = GeminiSTTStream("k")
    with pytest.raises(ProviderStreamError) as exc:
        await stream.connect(_config(channels=2))
    assert exc.value.recoverable is False
    assert "mono" in str(exc.value)


async def test_close_is_safe_from_any_state_and_idempotent():
    stream = GeminiSTTStream("k")
    await stream.close()
    await stream.close()  # idempotent, and never dialed


async def test_finish_sends_audio_stream_end():
    stream = GeminiSTTStream("k")
    ws = _FakeWS()
    stream._ws = ws
    stream._connected = True
    stream._config = _config()
    task = asyncio.create_task(stream.finish())
    await asyncio.sleep(0.05)  # let finish() reach its grace sleep
    assert json.loads(ws.sent[-1]) == {"realtimeInput": {"audioStreamEnd": True}}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ------------------------------------------------------------------ catalog


def test_registered_capabilities_match_the_documented_wire():
    from speechrouter_gateway.providers.registry import stt_stream_provider

    registered = stt_stream_provider("gemini")
    assert registered is not None
    caps = registered.capabilities
    assert caps.streaming and caps.interim_results
    # Documented as unavailable in live mode.
    assert not caps.word_timestamps and not caps.diarization
    assert caps.keyterms and caps.keyterms_max == MAX_CUSTOM_VOCABULARY
    assert caps.encodings == frozenset({"linear16"})


def test_model_resolves_and_prices_from_the_catalog():
    from speechrouter_gateway.config import Settings
    from speechrouter_gateway.router.catalog import Catalog
    from speechrouter_gateway.router.resolver import StreamRequest, resolve_stream

    resolved = resolve_stream(
        "gemini/gemini-3.5-transcribe-live",
        StreamRequest(),
        Settings(gemini_api_key="x", _env_file=None),
        Catalog.load(),
    )
    assert resolved.config.model == "gemini-3.5-transcribe-live"
    assert abs(resolved.price_per_second_usd - 0.30 / 3600) < 1e-12


def test_missing_key_makes_the_model_unavailable_instead_of_failing_mid_connect():
    from speechrouter_gateway.config import Settings
    from speechrouter_gateway.router.catalog import Catalog
    from speechrouter_gateway.router.resolver import ResolveError, StreamRequest, resolve_stream

    with pytest.raises(ResolveError):
        resolve_stream(
            "gemini/gemini-3.5-transcribe-live",
            StreamRequest(),
            Settings(gemini_api_key="", _env_file=None),
            Catalog.load(),
        )

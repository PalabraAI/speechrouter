"""Meta Muse Voice Transcribe fixtures — wire shapes taken from the documented
turn examples (docs/providers/meta.md, written 2026-09-01 from
dev.meta.ai/docs/speech-to-text and the Voice API reference). No socket
required."""

import asyncio
import io
import json
import wave
from urllib.parse import parse_qs

import pytest

from speechrouter_gateway.protocol import SpeechStarted, Transcript, UtteranceEnd
from speechrouter_gateway.providers.base import ProviderStreamError, STTConfig
from speechrouter_gateway.providers.meta.adapter import (
    MODE_DIARIZATION,
    MODE_ENDPOINTING,
    MODE_PUSH_TO_TALK,
    MetaSTTStream,
    TurnState,
    build_handshake,
    build_url,
    language_bias,
    parse_message,
    redact,
)
from speechrouter_gateway.providers.meta.batch import (
    MetaSTTBatch,
    build_request,
    check_wav,
    parse_response,
)


def _config(**kw):
    defaults = dict(model="muse-voice-transcribe-1.0", encoding="linear16", sample_rate=16000)
    defaults.update(kw)
    return STTConfig(**defaults)


def _frames(state, *frames):
    events = []
    for frame in frames:
        events.extend(parse_message(json.dumps(frame), state))
    return events


# The documented ENDPOINTING turn, verbatim.
_TURN = (
    {"type": "speechStart", "turnId": 1, "audioProcessedMs": 1200},
    {"type": "transcript", "transcript": "how is the", "final": False, "audioProcessedMs": 2400},
    {"type": "transcript", "transcript": "how is the weather", "final": False,
     "audioProcessedMs": 3200},
    {"type": "speechEnd", "turnId": 1, "audioProcessedMs": 3600},
    {"type": "speechComplete", "turnId": 1, "transcript": "How is the weather?",
     "audioProcessedMs": 3600},
)


# ------------------------------------------------------------- turn parsing


def test_endpointing_turn_maps_to_speech_edges_and_one_final():
    events = _frames(TurnState(MODE_ENDPOINTING), *_TURN)
    assert [type(e) for e in events] == [
        SpeechStarted, Transcript, Transcript, UtteranceEnd, Transcript,
    ]
    start, p1, p2, end, final = events
    assert start.at == 1.2
    assert p1.is_final is False and p1.text == "how is the"
    assert p1.start == 1.2 and p1.end == 2.4  # partials inherit the turn's start
    assert p2.is_final is False and p2.end == 3.2
    assert end.at == 3.6
    assert final.is_final is True
    assert final.text == "How is the weather?"  # post-processed, not the last partial
    assert final.start == 1.2 and final.end == 3.6
    assert final.words is None  # turn-level times only


def test_final_flag_on_transcript_is_interim_in_turn_modes():
    """The post-processed text arrives in speechComplete. A final here would
    reach the session layer first and its dedup gate would then drop the
    clean speechComplete text as a replay of the same audio range."""
    state = TurnState(MODE_ENDPOINTING)
    (t,) = _frames(
        state,
        {"type": "transcript", "transcript": "hello", "final": True, "audioProcessedMs": 900},
    )
    assert t.is_final is False


def test_push_to_talk_final_transcript_is_final():
    state = TurnState(MODE_PUSH_TO_TALK)
    events = _frames(
        state,
        {"type": "transcript", "transcript": "hello", "final": False, "audioProcessedMs": 900},
        {"type": "transcript", "transcript": "Hello there.", "final": True,
         "audioProcessedMs": 1800},
    )
    assert [e.is_final for e in events] == [False, True]
    assert events[1].end == 1.8 and events[1].start is None  # no turn in PTT
    # Should the server also send a speechComplete, it must not double up.
    assert _frames(state, {"type": "speechComplete", "turnId": 1, "transcript": "Hello there.",
                           "audioProcessedMs": 1800}) == []


def test_diarization_speaker_label_rides_as_a_segment_word():
    """One `speaker` frame per turn labels the audio behind it. It surfaces
    as a single segment-level Word carrying the speaker index, so
    words[].speaker works for this provider too (openai_compat convention)."""
    state = TurnState(MODE_DIARIZATION)
    events = _frames(
        state,
        {"type": "speechStart", "turnId": 1, "audioProcessedMs": 1200},
        {"type": "transcript", "transcript": "thanks for calling", "final": False,
         "audioProcessedMs": 2400},
        {"type": "speaker", "label": "A", "audioProcessedMs": 2480},
        {"type": "speechEnd", "turnId": 1, "audioProcessedMs": 3600},
        {"type": "speechComplete", "turnId": 1, "transcript": "Thanks for calling.",
         "audioProcessedMs": 3600},
        {"type": "speechStart", "turnId": 2, "audioProcessedMs": 4000},
        {"type": "speaker", "label": "B", "audioProcessedMs": 4500},
        {"type": "speechEnd", "turnId": 2, "audioProcessedMs": 5000},
        {"type": "speechComplete", "turnId": 2, "transcript": "Hi.", "audioProcessedMs": 5100},
        {"type": "speechStart", "turnId": 3, "audioProcessedMs": 5500},
        {"type": "speaker", "label": "A", "audioProcessedMs": 5900},
        {"type": "speechComplete", "turnId": 3, "transcript": "Yes.", "audioProcessedMs": 6200},
    )
    finals = [e for e in events if isinstance(e, Transcript) and e.is_final]
    assert [f.text for f in finals] == ["Thanks for calling.", "Hi.", "Yes."]
    assert [f.words[0].speaker for f in finals] == [0, 1, 0]  # first-seen order
    assert finals[0].words[0].w == "Thanks for calling."
    assert finals[0].words[0].start == 1.2 and finals[0].words[0].end == 3.6
    # speechEnd's clock wins over speechComplete's later one for `end`
    assert finals[1].end == 5.0
    # no speechEnd for turn 3: fall back to speechComplete's own clock
    assert finals[2].end == 6.2


def test_overlapping_turns_are_keyed_by_turn_id():
    """A later turn can open before the earlier turn's speechComplete."""
    state = TurnState(MODE_ENDPOINTING)
    events = _frames(
        state,
        {"type": "speechStart", "turnId": 1, "audioProcessedMs": 1000},
        {"type": "speechEnd", "turnId": 1, "audioProcessedMs": 2000},
        {"type": "speechStart", "turnId": 2, "audioProcessedMs": 2100},
        {"type": "speechComplete", "turnId": 1, "transcript": "One.", "audioProcessedMs": 2300},
        {"type": "speechEnd", "turnId": 2, "audioProcessedMs": 3000},
        {"type": "speechComplete", "turnId": 2, "transcript": "Two.", "audioProcessedMs": 3100},
    )
    finals = [e for e in events if isinstance(e, Transcript)]
    assert [(f.text, f.start, f.end) for f in finals] == [("One.", 1.0, 2.0), ("Two.", 2.1, 3.0)]


def test_finals_are_monotonic_for_the_dedup_gate():
    """audioProcessedMs only ever grows, so consecutive finals clear the
    session layer's `end > last_final_end` check."""
    state = TurnState(MODE_ENDPOINTING)
    events = _frames(
        state,
        {"type": "speechComplete", "turnId": 1, "transcript": "A.", "audioProcessedMs": 1000},
        {"type": "speechComplete", "turnId": 2, "transcript": "B.", "audioProcessedMs": 2000},
        {"type": "speechComplete", "turnId": 3, "transcript": "C.", "audioProcessedMs": 3000},
    )
    ends = [e.end for e in events]
    assert ends == sorted(ends) and len(set(ends)) == 3


def test_empty_transcripts_and_partial_turns_are_skipped():
    state = TurnState(MODE_ENDPOINTING)
    assert _frames(state, {"type": "transcript", "transcript": "", "final": False,
                           "audioProcessedMs": 10}) == []
    assert _frames(state, {"type": "transcript", "transcript": "   ", "final": False,
                           "audioProcessedMs": 10}) == []
    # "A turn the clip ended part-way through carries ... an empty transcript."
    assert _frames(state, {"type": "speechComplete", "turnId": 9, "transcript": "",
                           "audioProcessedMs": 10}) == []


def test_progress_ack_unknown_and_garbage_frames_are_ignored():
    state = TurnState()
    assert _frames(state, {"type": "audioProgress", "audioProcessedMs": 500}) == []
    assert _frames(state, {"sessionId": "abc"}) == []  # the typeless handshake ack
    assert _frames(state, {"type": "somethingNew", "x": 1}) == []
    assert parse_message("not json", state) == []
    assert parse_message("[1,2]", state) == []


def test_error_frame_is_recorded_not_raised():
    """The error frame always precedes a close frame, whose code decides
    whether the failover engine may retry. Raising here would lose it."""
    state = TurnState()
    assert _frames(state, {"type": "error", "message": "budget exceeded",
                           "sessionId": "s"}) == []
    assert state.error_message == "budget exceeded"


def test_include_raw_attaches_provider_payload():
    state = TurnState(MODE_ENDPOINTING, include_raw=True)
    (t,) = _frames(state, {"type": "speechComplete", "turnId": 1, "transcript": "Hi.",
                           "audioProcessedMs": 100})
    assert t.provider_raw == {"type": "speechComplete", "turnId": 1, "transcript": "Hi.",
                              "audioProcessedMs": 100}
    (t,) = _frames(TurnState(MODE_ENDPOINTING), {"type": "speechComplete", "turnId": 1,
                                                  "transcript": "Hi.", "audioProcessedMs": 100})
    assert t.provider_raw is None


# -------------------------------------------------------------- handshake


def test_handshake_minimal():
    hs = build_handshake(_config(), "LLM|123|abc")
    assert hs["authorization"] == {"accessToken": "Bearer LLM|123|abc"}
    assert hs["audioEncoding"] == "PCM_16KHZ"
    assert hs["model"] == "muse-voice-transcribe-1.0"
    assert hs["mode"] == "ENDPOINTING"  # our default, not Meta's PUSH_TO_TALK
    assert hs["partialMode"] == "CUMULATIVE"
    assert hs["emitAudioProgress"] is False
    assert "keywords" not in hs and "languageBias" not in hs


def test_handshake_24khz_is_the_native_rate():
    assert build_handshake(_config(sample_rate=24000), "k")["audioEncoding"] == "PCM_24KHZ"


def test_handshake_rejects_audio_the_wire_cannot_carry():
    for bad in (dict(sample_rate=8000), dict(sample_rate=44100), dict(channels=2),
                dict(encoding="mulaw", sample_rate=16000)):
        with pytest.raises(ProviderStreamError) as exc:
            build_handshake(_config(**bad), "k")
        assert exc.value.recoverable is False


def test_handshake_diarization_selects_diarization_mode():
    hs = build_handshake(_config(diarization=True), "k")
    assert hs["mode"] == "DIARIZATION"
    # and it wins over an explicit provider_params.mode
    hs = build_handshake(_config(diarization=True, provider_params={"mode": "PUSH_TO_TALK"}), "k")
    assert hs["mode"] == "DIARIZATION"


def test_handshake_mode_from_provider_params():
    hs = build_handshake(_config(provider_params={"mode": "push_to_talk"}), "k")
    assert hs["mode"] == "PUSH_TO_TALK"
    with pytest.raises(ProviderStreamError):
        build_handshake(_config(provider_params={"mode": "TURBO"}), "k")


def test_handshake_language_and_keyterms_become_names_and_keywords():
    hs = build_handshake(
        _config(language="fr", keyterms=("Acme Mobile", "eSIM"),
                provider_params={"keywords": ["5G", "eSIM"], "languageBias": ["English"]}),
        "k",
    )
    assert hs["languageBias"] == ["English", "French"]
    assert hs["keywords"] == ["Acme Mobile", "eSIM", "5G"]


def test_language_bias_maps_tags_to_names_and_passes_unknowns_through():
    assert language_bias("en-US", {}) == ["English"]
    assert language_bias("zh", {}) == ["Mandarin Chinese"]
    assert language_bias("auto", {}) == []
    assert language_bias(None, {}) == []
    assert language_bias("sv", {}) == ["sv"]  # not documented; the server decides
    assert language_bias("en", {"languageBias": "English,French"}) == ["English", "French"]


def test_handshake_reserved_keys_are_dropped_and_others_forwarded():
    hs = build_handshake(
        _config(provider_params={
            "authorization": {"accessToken": "stolen"},
            "audioEncoding": "PCM_24KHZ",
            "model": "other",
            "partialMode": "DELTA",
            "emitAudioProgress": True,
            "zdrOverride": True,
            "sessionId": "abc",
        }),
        "key-1",
    )
    assert hs["authorization"] == {"accessToken": "Bearer key-1"}
    assert hs["audioEncoding"] == "PCM_16KHZ"
    assert hs["model"] == "muse-voice-transcribe-1.0"
    assert hs["partialMode"] == "CUMULATIVE"
    assert hs["emitAudioProgress"] is False
    assert hs["zdrOverride"] is True
    assert "sessionId" not in hs  # travels as a query param instead


def test_build_url_session_id_is_the_only_query_param():
    assert build_url(_config()) == "wss://api.meta.ai/v1/asr/realtime"
    url = build_url(_config(provider_params={"sessionId": "mic 1"}))
    assert parse_qs(url.split("?", 1)[1]) == {"sessionId": ["mic 1"]}


def test_redact_hides_the_key():
    assert "secret" not in redact("sent Bearer secret", "secret")


# -------------------------------------------------------------- lifecycle


class _FakeWS:
    def __init__(self, incoming=()):
        self.sent: list = []
        self.closed = False
        self._incoming = list(incoming)
        self.close_code = None

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        item = self._incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True

    async def wait_closed(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._incoming:
            raise StopAsyncIteration
        item = self._incoming.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patch_dial(monkeypatch, ws):
    from speechrouter_gateway.providers.meta import adapter as mod

    async def dial(*_args, **_kwargs):
        return ws

    monkeypatch.setattr(mod, "ws_connect", dial)


def test_connect_sends_handshake_first_and_waits_for_the_ack(monkeypatch):
    ws = _FakeWS([json.dumps({"sessionId": "550e8400"})])
    _patch_dial(monkeypatch, ws)

    async def run():
        adapter = MetaSTTStream("k")
        await adapter.connect(_config(keyterms=("Vercel",)))
        assert adapter._session_id == "550e8400"
        first = json.loads(ws.sent[0])
        assert first["authorization"]["accessToken"] == "Bearer k"
        assert first["keywords"] == ["Vercel"]
        assert len(ws.sent) == 1  # no audio before the ack

    asyncio.run(run())


def test_connect_classifies_a_rejected_handshake(monkeypatch):
    import websockets
    from websockets.frames import Close

    async def run():
        # An error frame then a 1008 close: the request is wrong, never retry.
        ws = _FakeWS([json.dumps({"type": "error", "message": "invalid api key",
                                  "sessionId": ""})])
        ws.close_code = 1008
        _patch_dial(monkeypatch, ws)
        adapter = MetaSTTStream("k")
        with pytest.raises(ProviderStreamError) as exc:
            await adapter.connect(_config())
        assert exc.value.recoverable is False
        assert "invalid api key" in str(exc.value)
        assert ws.closed is True

        # An error frame then a 1013 close: rate limited, back off and retry.
        ws = _FakeWS([json.dumps({"type": "error", "message": "rate limited", "sessionId": ""})])
        ws.close_code = 1013
        _patch_dial(monkeypatch, ws)
        with pytest.raises(ProviderStreamError) as exc:
            await MetaSTTStream("k").connect(_config())
        assert exc.value.recoverable is True

        # The socket closing before any ack: classify on the close code.
        ws = _FakeWS([websockets.exceptions.ConnectionClosed(Close(1011, "backend"), None, None)])
        _patch_dial(monkeypatch, ws)
        with pytest.raises(ProviderStreamError) as exc:
            await MetaSTTStream("k").connect(_config())
        assert exc.value.recoverable is True and exc.value.code == "1011"

    asyncio.run(run())


def test_connect_rejects_bad_audio_before_dialing(monkeypatch):
    dialed = False

    async def dial(*_a, **_k):
        nonlocal dialed
        dialed = True

    from speechrouter_gateway.providers.meta import adapter as mod

    monkeypatch.setattr(mod, "ws_connect", dial)

    async def run():
        with pytest.raises(ProviderStreamError):
            await MetaSTTStream("k").connect(_config(sample_rate=8000))
        assert dialed is False

    asyncio.run(run())


def test_send_audio_splits_large_chunks_but_does_not_buffer():
    """Frame boundaries carry no meaning upstream, so a client frame is never
    held back — but a replayed 10 s ring chunk is split so it can be paced."""

    async def run():
        adapter = MetaSTTStream("k")
        adapter._ws = _FakeWS()
        adapter._frame_bytes = 10240  # 320 ms of 16 kHz linear16
        adapter._byte_rate = 0  # pacing off; this test is about framing
        await adapter.send_audio(b"\x01" * 4096)
        await adapter.send_audio(b"\x02" * 25000)
        assert [len(f) for f in adapter._ws.sent] == [4096, 10240, 10240, 4520]

    asyncio.run(run())


def test_send_audio_paces_a_failover_replay():
    """The ring replay after a failover arrives as fast as the feeder runs;
    the server drops streams more than ~5 s ahead of processing."""
    from speechrouter_gateway.providers.meta import adapter as mod

    async def run():
        adapter = MetaSTTStream("k")
        adapter._ws = _FakeWS()
        adapter._frame_bytes = 10240
        adapter._byte_rate = 32000
        loop = asyncio.get_running_loop()
        started = loop.time()
        await adapter.send_audio(b"\x02" * (32000 * 10))  # 10 s, the default ring
        elapsed = loop.time() - started
        expected = 10 / mod._MAX_REALTIME_FACTOR - mod._MAX_BURST_SECONDS
        assert expected * 0.5 < elapsed < expected + 1.0
        assert elapsed < 10  # never throttled all the way down to realtime

    asyncio.run(run())


def test_finish_sends_end_stream_and_leaves_the_socket_open():
    """endStream half-closes input; the server flushes and closes 1000.
    Closing the socket ourselves could discard pending events."""

    async def run():
        adapter = MetaSTTStream("k")
        adapter._ws = _FakeWS()
        await adapter.finish()
        await adapter.finish()
        assert adapter._ws.sent == [json.dumps({"type": "endStream"})]
        assert adapter._ws.closed is False
        await adapter.close()
        await adapter.close()
        assert adapter._ws.closed is True

    asyncio.run(run())


def test_events_classify_the_close_code():
    import websockets
    from websockets.frames import Close

    def closing(code, reason="bye"):
        return _FakeWS([websockets.exceptions.ConnectionClosed(Close(code, reason), None, None)])

    async def run():
        for code, recoverable in ((1008, False), (1011, True), (1013, True)):
            adapter = MetaSTTStream("k")
            adapter._ws = closing(code)
            with pytest.raises(ProviderStreamError) as exc:
                async for _ in adapter.events():
                    pass
            assert exc.value.code == str(code)
            assert exc.value.recoverable is recoverable

        # After finish() a close is the expected end of stream.
        adapter = MetaSTTStream("k")
        adapter._ws = closing(1000)
        adapter._finished = True
        assert [e async for e in adapter.events()] == []

        # ...unless an error frame came first: then even a 1000 is a failure
        # and the message is what the client sees.
        adapter = MetaSTTStream("k")
        adapter._ws = _FakeWS([
            json.dumps({"type": "error", "message": "processing budget exceeded",
                        "sessionId": "s"}),
            websockets.exceptions.ConnectionClosed(Close(1000, ""), None, None),
        ])
        adapter._finished = True
        with pytest.raises(ProviderStreamError) as exc:
            async for _ in adapter.events():
                pass
        assert "processing budget exceeded" in str(exc.value)

    asyncio.run(run())


def test_events_yield_the_documented_turn_end_to_end():
    async def run():
        adapter = MetaSTTStream("k")
        adapter._state = TurnState(MODE_ENDPOINTING)
        adapter._ws = _FakeWS([json.dumps(f) for f in _TURN])
        events = [e async for e in adapter.events()]
        assert [type(e) for e in events] == [
            SpeechStarted, Transcript, Transcript, UtteranceEnd, Transcript,
        ]
        assert events[-1].text == "How is the weather?"

    asyncio.run(run())


# ------------------------------------------------------------------ batch


def _wav(rate=16000, channels=1, width=2, seconds=1.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b"\x00" * int(rate * seconds) * channels * width)
    return buf.getvalue()


def test_check_wav_accepts_the_documented_container():
    check_wav(_wav(16000))
    check_wav(_wav(24000))
    check_wav(b"\xff\xfb\x90\x00 not a wav")  # undecidable here: the server judges


def test_check_wav_refuses_what_the_endpoint_documents_as_unsupported():
    for bad in (_wav(channels=2), _wav(width=1), _wav(rate=44100), _wav(rate=8000)):
        with pytest.raises(ProviderStreamError) as exc:
            check_wav(bad)
        assert exc.value.recoverable is False
        assert "ffmpeg" in str(exc.value)
    with pytest.raises(ProviderStreamError) as exc:
        check_wav(_wav(rate=16000, seconds=601))
    assert "10 minutes" in str(exc.value)


def test_build_request_mirrors_the_handshake_without_auth():
    req = build_request(_config(language="es", keyterms=("Celebrex",),
                                provider_params={"sessionId": "x", "partialMode": "DELTA"}))
    assert req == {
        "model": "muse-voice-transcribe-1.0",
        "audioEncoding": "WAV",
        "mode": "ENDPOINTING",
        "keywords": ["Celebrex"],
        "languageBias": ["Spanish"],
    }
    assert build_request(_config(diarization=True))["mode"] == "DIARIZATION"
    assert build_request(_config(provider_params={"mode": "PUSH_TO_TALK"}))["mode"] == (
        "PUSH_TO_TALK"
    )


def test_parse_response_turns_become_speaker_segments():
    payload = {
        "sessionId": "9f1c",
        "transcript": "How is the weather? It is raining.",
        "audioDurationMs": 8240,
        "turns": [
            {"turnId": 1, "startMs": 1520, "endMs": 4640, "transcript": "How is the weather?",
             "speaker": "A"},
            {"turnId": 2, "startMs": 5900, "endMs": 8240, "transcript": "It is raining.",
             "speaker": "B"},
            {"turnId": 3, "startMs": 8240, "endMs": 8240, "transcript": ""},  # cut off
        ],
    }
    t = parse_response(payload)
    assert t.is_final is True
    assert t.text == "How is the weather? It is raining."
    assert t.start == 0.0 and t.end == 8.24
    assert [(w.w, w.start, w.end, w.speaker) for w in t.words] == [
        ("How is the weather?", 1.52, 4.64, 0),
        ("It is raining.", 5.9, 8.24, 1),
    ]
    assert t.provider_raw is None
    assert parse_response(payload, include_raw=True).provider_raw == payload


def test_parse_response_push_to_talk_has_no_turns():
    t = parse_response({"sessionId": "s", "transcript": "Hello.", "audioDurationMs": 1500,
                        "turns": []})
    assert t.text == "Hello." and t.words is None and t.end == 1.5


def test_batch_posts_two_part_multipart(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"sessionId": "s", "transcript": "Hi.", "audioDurationMs": 1000, "turns": []}

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def post(self, url, **kw):
            captured.update(url=url, **kw)
            return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    async def run():
        t = await MetaSTTBatch("LLM|1|k").transcribe(
            _wav(), "audio/wav", _config(provider_params={"sessionId": "corr-1"})
        )
        assert t.text == "Hi."
        assert captured["url"] == "https://api.meta.ai/v1/asr/transcribe"
        assert captured["params"] == {"sessionId": "corr-1"}
        assert captured["headers"]["Authorization"] == "Bearer LLM|1|k"
        assert captured["headers"]["Accept"] == "application/json"
        request_part = captured["files"]["request"]
        assert request_part[0] is None and request_part[2] == "application/json"
        assert json.loads(request_part[1])["audioEncoding"] == "WAV"
        assert captured["files"]["audio"][2] == "audio/wav"

    asyncio.run(run())


def test_batch_status_codes_are_classified(monkeypatch):
    import httpx

    def client_for(status):
        class FakeResponse:
            status_code = status
            text = "nope"

        class FakeClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def post(self, url, **kw):
                return FakeResponse()

        return FakeClient

    async def run():
        for status, recoverable in ((400, False), (401, False), (413, False),
                                    (429, True), (500, True)):
            monkeypatch.setattr(httpx, "AsyncClient", client_for(status))
            with pytest.raises(ProviderStreamError) as exc:
                await MetaSTTBatch("k").transcribe(_wav(), "audio/wav", _config())
            assert exc.value.code == str(status)
            assert exc.value.recoverable is recoverable

    asyncio.run(run())


# ----------------------------------------------------------------- wiring


def test_provider_is_registered_and_resolvable():
    from speechrouter_gateway.config import KeyStoreKind, Settings
    from speechrouter_gateway.router.catalog import Catalog
    from speechrouter_gateway.router.resolver import (
        StreamRequest,
        resolve_batch,
        resolve_stream,
    )

    settings = Settings(keystore=KeyStoreKind.local, keys="k", _env_file=None, meta_api_key="x")
    attempt = resolve_stream(
        "meta/muse-voice-transcribe-1.0",
        StreamRequest(encoding="linear16", sample_rate=16000, diarization=True,
                      keyterms=("Acme",)),
        settings, Catalog.load(),
    )
    assert attempt.config.model == "muse-voice-transcribe-1.0"
    # $0.18/hr of audio -> per-second derived by the catalog
    assert attempt.price_per_second_usd == pytest.approx(0.18 / 3600)

    batch = resolve_batch(
        "meta/muse-voice-transcribe-1.0", StreamRequest(diarization=True), settings,
        Catalog.load(),
    )
    assert batch.config.diarization is True


def test_missing_credentials_and_bad_encoding_are_rejected_before_dialing():
    from speechrouter_gateway.config import KeyStoreKind, Settings
    from speechrouter_gateway.router.catalog import Catalog
    from speechrouter_gateway.router.resolver import ResolveError, StreamRequest, resolve_stream

    settings = Settings(keystore=KeyStoreKind.local, keys="k", _env_file=None)
    with pytest.raises(ResolveError):
        resolve_stream("meta/muse-voice-transcribe-1.0",
                       StreamRequest(encoding="linear16", sample_rate=16000),
                       settings, Catalog.load())

    settings = Settings(keystore=KeyStoreKind.local, keys="k", _env_file=None, meta_api_key="x")
    with pytest.raises(ResolveError):
        resolve_stream("meta/muse-voice-transcribe-1.0",
                       StreamRequest(encoding="mulaw", sample_rate=8000),
                       settings, Catalog.load())

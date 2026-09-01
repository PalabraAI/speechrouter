# Meta (Muse Voice Transcribe) — STT protocol brief (docs read 2026-09-01)

Docs: dev.meta.ai (`/docs/speech-to-text`, `/docs/api-reference/voice/{realtime,transcribe,schemas}`, `/docs/authentication`). The reference pages are hand-written from Meta's internal thrift (`duplex.thrift`, `transcribe.thrift`) — there is no public OpenAPI spec and no official SDK for the voice endpoints, so nothing below is SDK-cross-checked. **Not yet live-verified**: see Open.

One model, `muse-voice-transcribe-1.0`, served two ways from `api.meta.ai` (the same host as the Model API chat endpoints):

| Endpoint | Auth |
|---|---|
| `wss://api.meta.ai/v1/asr/realtime` | in the handshake frame; the HTTP `Authorization` header is **ignored** |
| `POST https://api.meta.ai/v1/asr/transcribe` | `Authorization: Bearer <key>` |

Keys look like `LLM|607358788850350|nx9.....LJY` and come from the Model API dashboard.

## Realtime WS
- Optional query `sessionId` (log-correlation id; generated and returned when absent). Nothing else in the URL.
- **First JSON text frame is the handshake, within 10 s of connect.** Fields: `authorization.accessToken` (`"Bearer <key>"`, prefix included — req), `audioEncoding` (`PCM_24KHZ` | `PCM_16KHZ` — req, no default), `model` (req), `mode` (`PUSH_TO_TALK` def | `ENDPOINTING` | `DIARIZATION`), `partialMode` (`CUMULATIVE` def | `DELTA`), `emitAudioProgress` (def true), `keywords[]`, `languageBias[]` (language **names**, e.g. `"English"`, not tags), `zdrOverride`. Config is fixed for the session.
- Ack: `{"sessionId": "..."}` — the only server frame with no `type`. Send no audio before it.
- Audio: raw **signed 16-bit LE mono PCM at 24 kHz (native) or 16 kHz (server resamples)**, binary frames. Frame boundaries carry no meaning. Nothing else — no mulaw/alaw, no 8 kHz, no stereo. Ours→theirs: linear16@16000 → `PCM_16KHZ`, linear16@24000 → `PCM_24KHZ`; everything else is refused in `connect()`.
- **Pacing is enforced**, checked once a second: not more than **5 s of audio ahead of processing**, and not slower than realtime either; a live source must keep sending PCM silence during pauses. Violations close with 1008.
- End of input: text frame `{"type":"endStream"}`. Half-closes input; the server flushes pending results and closes **1000**. Closing the socket outright can discard pending events.
- No application keepalive frame. "Idle input: the server closes a stream that stops sending audio without sending endStream" — threshold undocumented.
- Session cap 60 min (`1011` with reason `Max session duration reached`), no resume token.

```
wss://api.meta.ai/v1/asr/realtime?sessionId=<id>
→ {"authorization":{"accessToken":"Bearer LLM|..."},"audioEncoding":"PCM_16KHZ","model":"muse-voice-transcribe-1.0","mode":"ENDPOINTING","partialMode":"CUMULATIVE","emitAudioProgress":false,"keywords":[...],"languageBias":["English"]}
← {"sessionId":"550e8400-..."}
```

## Modes
| Mode | Completion signal | Turn timestamps |
|---|---|---|
| `PUSH_TO_TALK` (Meta's default) | one `transcript` with `final: true`, after `endStream` | none |
| `ENDPOINTING` (**our default**) | `speechComplete` per `turnId` | yes |
| `DIARIZATION` (selected by `diarization=true`) | `speechComplete` per `turnId`, plus `speaker` | yes |

A gateway that wants one final per utterance cannot use PUSH_TO_TALK, so the adapter defaults to ENDPOINTING; PUSH_TO_TALK is reachable via `provider_params.mode`. Meta says DIARIZATION "marks a possible new speaker rather than a clean speech endpoint" and is not tuned for low-latency use.

## Response
JSON text frames dispatched on `type`; unknown types must be ignored (additive). Every event carries `audioProcessedMs` — **total audio processed so far, ms from stream start**. It is progress, "not a precise acoustic boundary", and the only timestamp there is. No word timestamps, no confidence.

Documented ENDPOINTING turn:
```
{"type":"speechStart","turnId":1,"audioProcessedMs":1200}
{"type":"transcript","transcript":"how is the","final":false,"audioProcessedMs":2400}
{"type":"transcript","transcript":"how is the weather","final":false,"audioProcessedMs":3200}
{"type":"speechEnd","turnId":1,"audioProcessedMs":3600}
{"type":"speechComplete","turnId":1,"transcript":"How is the weather?","audioProcessedMs":3600}
```
- `transcript` frames carry **no `turnId`**; they belong to the most recent `speechStart`. Under `CUMULATIVE` each one replaces the previous partial (may revise earlier text).
- `speechEnd` is the boundary, not the text. The text is in `speechComplete` and "may differ from the last partial" (punctuation, casing).
- **Turns can overlap**: a later `speechStart` may arrive before the earlier turn's `speechComplete`. Key state on `turnId`.
- `speaker` (DIARIZATION): `{"type":"speaker","label":"A","audioProcessedMs":...}` labels the span *behind* it (from the previous `speechStart` or `speaker`, whichever is later). Exactly one per turn. Labels are session-scoped, the same label can repeat, and it does not mark a boundary.
- `audioProgress`: `{"type":"audioProgress","audioProcessedMs":...}`, on by default; we turn it off.
- `error`: `{"type":"error","message":"...","sessionId":"..."}` — "sent just before it closes the session". No code field; the **close code** is the classification.

### Mapping
- `speechStart` → `speech_started(at)`; `speechEnd` → `utterance_end(at)`; `at = audioProcessedMs/1000`.
- `transcript` → interim `Transcript(start=turn start, end=at)`. In the turn modes a `final: true` transcript is **still emitted as interim**: the post-processed `speechComplete` follows with the same audio range, and the session layer's dedup gate (`end > last_final_end`) would otherwise drop the clean text. In PUSH_TO_TALK `final: true` is the final.
- `speechComplete` → final `Transcript(text, start=turn's speechStart, end=turn's speechEnd, else own clock)`, `words=None`. In DIARIZATION the turn's label rides as one segment-level `Word{w=text,start,end,speaker}` (first-seen index; the openai_compat convention for diarized segments) so `words[].speaker` works for this provider too. Empty transcript ("a turn the clip ended part-way through") → skipped.
- `audioProcessedMs` is monotonic, so finals emitted in turn order clear the dedup gate. Turns completing out of order (possible per the overlap note) would make the earlier one look like a replay; not seen in the docs' examples.

## Batch
- `POST /v1/asr/transcribe`, multipart: `request` part (JSON, same fields as the handshake minus `authorization`/`zdrOverride`; `audioEncoding` must be `"WAV"`) + `audio` part.
- Audio: **RIFF/WAVE, mono, 16-bit integer PCM, 16 kHz or 24 kHz. Max 32 MB, max 10 minutes.** Meta's own docs say to convert everything else with ffmpeg first. The adapter refuses a WAV header that clearly violates this (clear message, no wasted upload) and passes anything the stdlib cannot parse through to the server's own 400.
- `Accept: application/json` (default) → `{sessionId, transcript, audioDurationMs, turns[{turnId,startMs,endMs,transcript,speaker?}]}`. `turns` is empty in PUSH_TO_TALK; `speaker` present in DIARIZATION only. `text/event-stream` replays the realtime events; `text/plain` gives one turn per line. We use JSON.
- Status: 400 unsupported WAV / >10 min / bad multipart; 406 bad Accept; 413 >32 MB; 429 tenant concurrency or hourly limit (retry with backoff); 500 processing budget (retry once). Errors are status + one message, no code field.
- Mapping: turns → segment-level `Word`s (speaker index in DIARIZATION), `end = audioDurationMs/1000`. Default mode ENDPOINTING so turn times exist.

## Errors / close codes
| Code | Meaning | Adapter |
|---|---|---|
| 1000 | normal completion after `endStream` | clean end of `events()` (unless an `error` frame preceded it) |
| 1008 | invalid request, or a streaming-policy failure (backlog, below-realtime ingress) | `recoverable=False` |
| 1011 | internal/backend failure; `Max session duration reached` | `recoverable=True` |
| 1013 | rate limited (8 concurrent streams / 1,000 per hour per tenant, shared with batch) | `recoverable=True` |

A rejected handshake (bad key, bad config) arrives as an `error` frame followed by a close; `connect()` reads the ack, and on an error frame waits ≤2 s for the close code so a 1013 stays retryable.

## Pricing (dev.meta.ai/docs/speech-to-text#availability-pricing, 2026-09-01)
**$0.18 per hour of audio processed**, streaming and batch alike; ZDR at parity → `per_audio_hour_usd: 0.18`, `BillingBasis.AUDIO_TIME`. Billed on audio actually processed, rounded down to whole seconds; failed and 429'd requests are not billed. Free-tier credits apply.

## Languages
25, with code-switching: Arabic, Bengali, Dutch, English, French, German, Hebrew, Hindi, Indonesian, Italian, Japanese, Kannada, Korean, Malay, Mandarin Chinese, Marathi, Polish, Portuguese, Spanish, Tagalog, Tamil, Telugu, Thai, Turkish, Vietnamese. `languageBias` is a hint, not a constraint; omit it for autodetect. The adapter maps the unified `language` tag's primary subtag to the name (`en-US` → `English`, `zh` → `Mandarin Chinese`); anything undocumented is forwarded verbatim with a warning.

## Open — verify against the live service
1. **Pacing constants.** The adapter bursts up to 3 s then meters at 1.5× realtime (`_MAX_BURST_SECONDS`, `_MAX_REALTIME_FACTOR`). "5 s ahead of processing" depends on the server's processing speed, which is not documented; a failover replay of the 10 s ring is the case to test. If the server processes well above realtime, the factor can go up.
2. **Idle-input threshold.** How long a connected stream may go without audio before the close. Clients that pause their mic will hit it; the fix on our side would be silence injection, which shifts every later timestamp against the client's audio clock — deliberately not done.
3. Do turn modes also send `transcript{final:true}` before `speechComplete`? The adapter treats it as interim either way.
4. Does PUSH_TO_TALK also emit `speechComplete`? Guarded (dropped once a final transcript was seen).
5. WS ping/pong: not mentioned; the adapter keeps websockets' default 20 s pings. Telnyx needed them off; Meta may too.
6. Whether turns can *complete* out of order (docs only say they can *open* out of order). If so, the earlier turn's final is lost to the dedup gate.
7. Handshake rejection shape for a bad key: assumed `error` frame + 1008. If it is an HTTP 401 on the upgrade instead, `connect()` already maps that to unrecoverable.
8. Concurrency: 8 concurrent streams per tenant is far below `SPEECHROUTER_MAX_CONCURRENT_STREAMS` (20). A house key will hit 1013 under modest load; BYOK is the realistic path for anyone above that.

## Adapter notes
- Config-in-first-frame + close-code classification → closest existing shape is **soniox** (config frame) crossed with **palabra** (pacing bucket, reserved-key handling). `partialMode` and `emitAudioProgress` are reserved: DELTA partials would be forwarded as if they were whole hypotheses, and progress frames carry nothing the wire schema surfaces.
- `keyterms` → `keywords`, merged with `provider_params.keywords`. `language` → `languageBias`, merged likewise.
- `Capabilities`: streaming+batch, `interim_results`, `diarization`, `endpointing`, `keyterms` True; `word_timestamps` False; encodings `{linear16}`; sample rates `{16000, 24000}`; `realtime_pacing_required`.

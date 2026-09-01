---
title: Gemini
description: Gemini 3.5 Transcribe Live at $0.30/hr — 85+ languages, custom vocabulary, and a SMART mode that cleans up disfluencies as it goes.
sidebar: { order: 8 }
---

Gemini 3.5 Transcribe Live at $0.30/hr — 85+ languages, custom vocabulary, and a SMART mode that cleans up disfluencies as it goes.

All prices are the vendor's public list price — 0% markup. Extra provider
knobs pass through untouched via `provider_params`.

| Model | Modes | List price | Diarization | Word timings |
| --- | --- | --- | --- | --- |
| `gemini/gemini-3.5-transcribe-live` | streaming | $0.3/hr | <span class="sr-no">—</span> | <span class="sr-no">—</span> |

## Provider options

Reach past the unified surface with [`provider_params`](/guides/streaming/#query-parameters)
— forwarded streaming → the Live API `setup` message. Typed in the SDKs as
`{provider}Params` interfaces (`providerParams` option / `provider_params=` kwarg).

:::note
Keys are the Live API's own camelCase field names. `mode`, `languageCodes` and `customVocabulary` are merged into setup.inputAudioTranscription; anything else lands at the top level of `setup` (realtimeInputConfig, contextWindowCompression, ...). `model` and `sessionResumption` are owned by the adapter — it reconnects on its own around the ~10-minute connection cap — and are dropped with a warning. The unified `language` sets languageCodes and `keyterms` sets customVocabulary; restating them here overrides that.
:::

| Param | Type | Default | Applies to | What it does |
| --- | --- | --- | --- | --- |
| `mode` | `VERBATIM` · `SMART` | `"VERBATIM"` | streaming | VERBATIM keeps fillers, repetitions and false starts; SMART removes disfluencies and applies punctuation, grammar and casing |
| `languageCodes` | array | — | streaming | BCP-47 codes to constrain recognition; empty (the default) means automatic detection. Use this instead of `language` to allow several languages |
| `customVocabulary` | array | — | streaming | Up to 1000 phrases biasing recognition toward domain terms; Google recommends staying near 100. Same field the unified `keyterms` sets |
| `realtimeInputConfig` | object | — | streaming | Voice-activity tuning, e.g. {"automaticActivityDetection": {"silenceDurationMs": 500, "prefixPaddingMs": 100}}; set automaticActivityDetection.disabled to drive turns yourself |

## Try it

```text
wss://api.speechrouter.ai/v1/listen?model=gemini/gemini-3.5-transcribe-live
```
Streaming-only — connect with the [SDKs](/sdks/javascript/) or see [Streaming](/guides/streaming/).

<sub>Generated from the gateway catalog — the billing engine's own source of truth.</sub>

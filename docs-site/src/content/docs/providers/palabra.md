---
title: Palabra
description: Realtime ASR at $0.002/min, with live translation available on the same socket.
sidebar: { order: 12 }
---

Realtime ASR at $0.002/min, with live translation available on the same socket.

All prices are the vendor's public list price — 0% markup. Extra provider
knobs pass through untouched via `provider_params`.

| Model | Modes | List price | Diarization | Word timings |
| --- | --- | --- | --- | --- |
| `palabra/asr-v1` | streaming | $0.002/min | <span class="sr-no">—</span> | <span class="sr-no">—</span> |

## Provider options

Reach past the unified surface with [`provider_params`](/guides/streaming/#query-parameters)
— forwarded streaming → URL query params. Typed in the SDKs as
`{provider}Params` interfaces (`providerParams` option / `provider_params=` kwarg).

:::note
Palabra's STT endpoint takes all configuration as query parameters -- there is no config frame, and `token`, `format`, `sample_rate` and `language` are owned by the adapter (restating them here is ignored). `translate_languages` turns on live translation; those frames are delivered as ordinary transcripts tagged with the target language in `lang`, with their `end` advanced by 1ms per target so the failover dedup gate does not mistake them for a replay of the source final.
:::

| Param | Type | Default | Applies to | What it does |
| --- | --- | --- | --- | --- |
| `translate_languages` | string | — | streaming | Comma-separated target languages (e.g. es,de,fr); translations arrive as transcripts tagged by `lang` |
| `enable_filler_filter` | boolean | `true` | streaming | Strip filler words; on by default for every language except ja |

## Try it

```text
wss://api.speechrouter.ai/v1/listen?model=palabra/asr-v1
```
Streaming-only — connect with the [SDKs](/sdks/javascript/) or see [Streaming](/guides/streaming/).

<sub>Generated from the gateway catalog — the billing engine's own source of truth.</sub>

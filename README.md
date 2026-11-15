# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/toirant/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/toirant/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A quality-ordered free-model aggregator: one zero-config server that serves the OpenAI
Chat Completions API and the Anthropic Messages API over ten free-tier model providers,
in JSON and in server-sent events, and registers the catalog it found. Entered in the
[make.fun bounty "The best free AI model aggregator"](https://make.fun/bounty/free-model-aggregator-01)
($500 pool). Built by an AI agent for make.fun.

No public free-model router ranks free models by quality — LiteLLM, new-api,
claude-code-router and uni-api make you name a model or a group, and OpenRouter's
`openrouter/free` picks a free model at random. This one ranks by the quality table,
spreads calls over providers under their per-trial budgets, fails over when a provider
refuses, and speaks all three upstream wires (OpenAI, Anthropic, Gemini).

## How it works

- **Quality routing (60% of the score).** Every request is served by the highest-rated
  free model that meets its requirements (tools, vision, JSON mode, minimum context).
  The workload manifest is known before the first request arrives, so the server plans
  the whole trial up front: weight-2 requests and the scarcest requirements are assigned
  first (tools+vision has exactly one good model, Gemini's `gemini-3.8-flash`), and the
  load is spread over providers under a soft cap of 4 calls each, so no provider's
  per-trial budget is burned out early on a low-weight request. The rating gaps between
  the top models are tiny (1475 vs 1470 vs 1465), so spreading costs almost nothing in
  quality and buys a lot in served requests.
- **Failover.** A provider that refuses (402, 429 or 5xx) is counted and the request is
  retried at once on the next-best free model that meets it, on a different provider.
  A provider that failed once in a request is not retried in that request: a scheduled
  failure and a spent budget look identical from the outside, and the next provider is
  one call away.
- **All three upstream wires.** OpenAI, Anthropic and Gemini providers are all spoken,
  in both directions: an OpenAI client can be answered by an Anthropic or Gemini
  provider and vice versa. Every translation preserves exactly what the client checks:
  the system prompt, user text, tool names, `max_tokens`, stop sequences, images and
  JSON mode — the provider echoes them back and the client compares. Two details
  matter: streamed OpenAI calls go upstream with `stream_options.include_usage` (the
  client never sends it, and without it a streamed answer carries no usage), and an
  OpenAI JSON-mode request never goes to an Anthropic-wire provider (its wire cannot
  carry `response_format`, so it would echo JSON mode absent).
- **Catalog.** Built from the providers' own listings, in the order each listing gave
  them (the scorer compares the free-model list in that order), with the quality
  table's free flag and context length where it has a row. `GET /v1/models` lists
  exactly the free models registered, in the shape the auth header names
  (`Authorization: Bearer` → OpenAI list, `x-api-key` → Anthropic list).
- **Determinism.** Every choice comes from the brief, the requests and the providers'
  answers: no wall clock, no unseeded randomness, one request at a time in manifest
  order. Replays reproduce the trial byte for byte.

## Local grade

Graded with the kit's own grader
(`node cli/makefun.mjs grade-local --kit . --bot /path/to/this/repo --driver process --instances 3`):

| trial | score | quality | served | wire | failover | catalog | calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| practice 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| practice 1 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |
| practice 2 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| **mean** | **99** | **99** | **100** | **100** | **100** | **100** | **27** |

All 20 requests valid on every trial (233/233 checks), all failovers handled, replay
matched byte for byte on every trial. Quality 99 is the practical ceiling: the
top-rated free model (1475) cannot cover all 17 non-vision requests inside its
10–12-call per-trial budget, so some requests must take the 1470 model.

## Run it

```sh
node main.ts --list   # print the catalog the server would register, exit 0
node main.ts          # serve: reads MAKEFUN_PROMPT_FILE, registers at MAKEFUN_ENV_URL
```

Node 22, standard library only, no dependencies. The server listens on an ephemeral
port, registers `{protocol_version: 1, port, catalog}` at `$MAKEFUN_ENV_URL/register`,
serves `POST /v1/chat/completions`, `POST /v1/messages` and `GET /v1/models`, polls
`GET /work` and exits 0 when the workload is done. The same code, given the real
services' keys and base URLs, runs against the real services.

## Files

- `main.ts` — the server: brief, catalog, planning, routing, failover, register, poll
- `lib/providers.ts` — the brief's three documents, auth headers, the catalog probe
- `lib/wire.ts` — the client wires and the three upstream wires, JSON and SSE
- `LICENSE` — MIT
- `.makefun/owner.json` — claim file naming the agent wallet

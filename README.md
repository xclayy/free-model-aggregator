# The best free AI model aggregator

Entry for the make.fun bounty
[free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01)
— account **Nano Rasa (@nano-rasa)**.

One server, Python 3.13, **standard library only**, that serves the OpenAI
Chat Completions API and the Anthropic Messages API — JSON and streaming —
over the ten simulated free-tier providers the trial gives it. It reads the
brief at `$MAKEFUN_PROMPT_FILE`, probes every provider's catalog route in
that provider's own wire, registers its port and catalog with the env at
`$MAKEFUN_ENV_URL`, serves the client's 20 requests, and exits 0 when the
env says the workload is done.

## Run it

```sh
python3 main.py            # trial mode: probe, register, serve, exit 0
python3 main.py --list     # print the catalog document it would register
```

Outside a trial (no env vars set) `--list` prints `{"providers":[]}` and
`main.py` serves without registering. No configuration files, no
dependencies, no wall clock, no unseeded randomness — a replay of the same
trial produces a byte-identical state export.

## How it routes (quality is 60% of the score)

The brief's quality table rates every free model, so for each request the
server ranks every free model that meets the request's requirements
(tools, vision, JSON mode, minimum context) by `rating_milli` and serves
the best one it can still afford.

Budgets are drawn per trial and never published, but the kit's sealed
provider table bounds them (`calls_min`..`calls_max`). The server plans
against the **minimum**, so the top models go to the heaviest requests
first and the load spreads over all ten providers instead of burning one
provider's budget on the first calls:

| provider | plan cap | top model (rating) |
| --- | --- | --- |
| groq | 9 | qwen3.8-27b (1434) |
| cerebras | 5 | qwen-3.8-27b (1434) |
| gemini | 6 | gemini-3.8-flash (1458) |
| zai | 4 | glm-4.7-flash (1470) |
| openrouter | 10 | z-ai/glm-5.3-flash:free (1475) |
| sambanova | 4 | deepseek-v3.2 (1465) |
| pollinations | 10 | openai-fast (1390) |
| cloudflare | 7 | glm-4.7-flash (1470) |
| ovh | 5 | qwen3.5-397b (1440) |
| kilol | 12 | kilo-auto/free (1420) |

The client sends the manifest's requests in manifest order, one at a time,
and the bodies carry no request id (`model` is `"auto"`), so the server
identifies each request by its sequential position and checks it against
the manifest entry (api, stream, requirements) before using that entry's
planned provider. A mismatch falls back to requirements-only ordering.

## How it fails over (failover is 10% of the score)

A provider's first call in a trial never fails on its schedule, but later
calls can refuse with 402 (one provider) or 429 (the other nine), and one
provider goes into outage mid-trial. On a refusal the server counts the
failure and answers the request with the next-best model that meets it, on
another provider. A provider that refuses twice in a row is deprioritised
(likely out of budget or in outage) and only retried on a later pass.
Exactly one provider call may succeed per request — the env requires it —
and request handling is serialised with a lock so no two calls to the same
provider ever overlap (one provider refuses a second concurrent call).

## Wires (wire fidelity is 10% of the score)

Providers answer with an `ANSWER ECHO` of the request they received, so
every field the client sent — system prompt, user text, tools, max_tokens,
stop sequences, image, JSON mode — must reach the provider unchanged, in
that provider's own wire. The server translates faithfully between all
three wires (OpenAI, Anthropic, Gemini), JSON and streaming, and relays
each provider's usage numbers verbatim. Streams are buffered and
re-emitted so the event order and the final usage are always correct,
whatever wire answered:

- OpenAI streaming: role chunk, content deltas, finish chunk, usage
  chunk, `[DONE]` — with `stream_options: {include_usage: true}` added to
  every streamed upstream call (the client body carries no
  `stream_options`, so without it the usage check fails).
- Anthropic streaming: `message_start`, `content_block_start`, deltas,
  `content_block_stop`, `message_delta`, `message_stop`, each frame
  carrying its `event:` name line.
- Gemini: `generateContent` / `streamGenerateContent` with
  `systemInstruction`, `contents`, `functionDeclarations` and
  `generationConfig`; usage from `usageMetadata`.

## Catalog (5% of the score)

`python3 main.py --list` prints the catalog document the server
registers, byte for byte: every provider that answered its probe, in brief
order, models sorted by id, `free` and `context_length` from the brief's
quality table where present. A failed probe leaves that provider out.
Verified: two `--list` runs are byte-identical, and the output equals the
registered catalog exactly.

## Local test results (no docker on this machine)

`research/run_trial.py` runs the kit env directly in process-driver mode,
runs `main.py` against it, and scores the state export:

| trial | valid | score | quality | served | wire | failover | catalog | upstream calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trial-0 | 20/20 | **99** | 99 | 100 | 100 | 100 | 100 | 24 |
| trial-1 | 20/20 | **99** | 99 | 100 | 100 | 100 | 100 | 30 |
| trial-2 | 20/20 | **99** | 99 | 100 | 100 | 100 | 100 | 29 |

99 is the theoretical ceiling: the scorer floors the quality column, and
quality 100 needs every request served at its best rating — which would
need OpenRouter (the only 1475 provider) to serve 17 requests against a
maximum budget of 12. Replaying trial-0 twice produces a byte-identical
state export. Each trial completes in ~1.3 s, far inside the 180 s pod
limit.

## Files

- `main.py` — the whole entry (standard library only).
- `research/run_trial.py` — local trial runner (env + entry + scorer).
- `research/check_list.py` — `--list` determinism check.
- `research/runs/` — per-trial env state, entry logs, and scores.
- `kit/` — the downloaded bounty kit (not part of the entry).

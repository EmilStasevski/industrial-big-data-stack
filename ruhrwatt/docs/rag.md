# RAG assistant: local LLM on the gold data

`ruhrwatt rag ask "..."` / `ruhrwatt rag chat` answer questions about the NRW energy data in
English or German, fully local: the LLM and the embedding model run in Ollama on the GPU,
the data never leaves the machine.

## Design: numbers from tools, words from the model

A language model that "knows" prices will invent them. So the assistant splits the work:

| Question type | Answered from | How |
|---|---|---|
| Figures ("average price in March", "savings for a 40 MW plant", "cheapest hours tomorrow") | gold/silver tables | the model calls a tested query function (tool) and only phrases the result |
| Methods and caveats ("why Amprion", "how is the range calibrated") | project docs | vector search over `README.md` and `docs/*.md` (LanceDB) |

The system prompt forbids any number that did not come from a tool result in the
conversation, and every answer lists the sources of the tool results it used.

**Tools** (`rag/tools.py`): `dataset_info`, `price_stats(start, end, flat_mw)`,
`period_breakdown(start, end, by=week|day, flat_mw)`, `monthly_market(months)`,
`cheapest_hours(day, hours, contiguous, most_expensive)`,
`load_shift_savings(avg_mw, min_frac, max_frac, preset, start, end)`, `forecast_quality()`,
`search_docs(query)`. Costs of a constant load and week/day rankings are computed by the
tools (`flat_mw`, `period_breakdown`), so the model never does arithmetic. A preset gives
starting values that explicit `avg_mw`/`min_frac`/`max_frac` override, and savings from
fewer than 28 days carry a warning instead of a silent yearly extrapolation.

**Push-back:** if the model answers without any tool result, or announces a tool call
instead of making it, the agent asks once more for the tool call. A real refusal ("gas
prices are not in the data") is simply repeated and kept.

**Grounding check:** after a tool-backed answer, every number in it must appear in a tool
result or the question, or be a difference or percentage of two of them (rounding
tolerated; dates, times, years and small counts ignored). Otherwise the agent names the
unsupported numbers and asks once to fetch or remove them. Numbers that survive are listed
in `Answer.ungrounded` and in the batch report. It is a heuristic: it reliably catches
invented figures (e.g. a second half of a comparison that was never fetched), but a made-up
number that happens to equal a difference or ratio of tool numbers can pass.

**Strict arguments:** a tool call with an argument the tool does not accept returns an error
naming the accepted arguments, so the model retries. Silently dropping it was worse: in a
live run, `start`/`end` sent to `monthly_market` were ignored and a ranking over all months
answered a question about one month.

**Also in the tools:** `monthly_market(sort_by=..., order=highest|lowest, year=..., top=...)` ranks complete
months itself (an 8B model misreads 12-row tables), every period tool warns when a requested
range reaches beyond the data instead of silently clipping it, and `forecast_quality`
returns the real top features with a note that there are no weather or fuel inputs. The
system prompt carries a short block of verified data facts (one DE-LU price for all of
Germany, 23/25-hour clock-change days, Amprion as the NRW proxy).

## Lazy loading

* **Data:** tools read the Parquet tables with Polars lazy scans, so only the needed columns
  and rows are loaded (predicate and projection pushdown).
* **Docs index:** built once (`make rag-index`); on refresh only new or changed chunks are
  embedded again (chunk id = hash of its text). The LanceDB table opens on the first search.
* **Models:** Ollama loads a model into VRAM on the first request and keeps it for
  `RUHRWATT_LLM_KEEP_ALIVE` (default 10 min), then frees the memory.

## Live results (qwen3, 28 German questions)

| Run | Fully right | Typical errors |
|---|---|---|
| qwen3:8b, first tools | 17 | presets overrode user numbers, invented periods, no tool calls |
| qwen3:8b, fixed tools + push-back | 19 | misread 12-row tables, invented a second half, clipped periods |
| qwen3:14b, + ranking, grounding, data facts | 22 | dropped `start`/`end`, wrong sort column / direction |
| qwen3:14b, + strict arguments, `order`, TSO fact | 23 | relative period dropped, two-part ranking, an invented date |

Each round's failures became tool or prompt fixes and regression cases in the eval set
(rules for relative periods and two-part rankings came from the fourth run).

### Known limitations

- **Dates in free text are not grounded.** The grounding check ignores dates and times, so a
  wrong date in an explanation ("the clocks changed on 30 October") is not caught.
- **Reasoning on top of correct numbers** can still be weak or one-sided (e.g. comparing each
  region's best month instead of a typical one). The numbers are right; the interpretation
  needs a human eye. Larger models help: qwen3:14b beat qwen3:8b by 3-4 questions.
- **The grounding check is heuristic** (see above): a made-up number that equals a
  difference or ratio of two tool numbers passes.

## Fitting into 16 GB VRAM

Approximate sizes at Ollama's default 4-bit quantization (check `ollama list` and
`ruhrwatt rag doctor`, which shows the VRAM actually used):

| Model | Role | VRAM (approx.) |
|---|---|---|
| `qwen3:8b` (default) | chat + tool calling | ~5-6 GB + context |
| `qwen3:14b` | stronger chat, still fits | ~9-10 GB + context |
| `llama3.1:8b` | alternative with tool calling | ~5 GB + context |
| `bge-m3` (default) | multilingual embeddings (German and English) | ~1.2 GB |

Chat model + embedding model + an 8k context fit comfortably on a 16 GB card; the 14B model
fits too. For qwen3 the reasoning trace is switched off by default (`RUHRWATT_LLM_THINK`),
which makes tool calls much faster; set it to `true` to compare.

## Commands

```bash
make rag-setup                      # ollama pull qwen3:8b + bge-m3, then `rag doctor`
make rag-index                      # embed the docs
make ask Q="Wie hoch war der durchschnittliche Strompreis im September 2026?"
make chat                           # interactive, keeps context; /reset clears it
make rag-eval                       # evaluation set
make rag-eval MODELS="qwen3:8b qwen3:14b llama3.1:8b"   # compare models
```

Settings (environment): `RUHRWATT_LLM_MODEL`, `RUHRWATT_EMBED_MODEL`, `RUHRWATT_LLM_CTX`
(8192), `RUHRWATT_LLM_KEEP_ALIVE`, `RUHRWATT_LLM_THINK` (auto/true/false), `OLLAMA_HOST`.

## Evaluation

`rag/evaluate.py` holds about a dozen questions whose correct answers are **recomputed from
the current data on every run** (no hard-coded figures): average and peak price of the latest
complete month, negative-price hours, Germany vs Amprion renewable share, the highest price
and its date in the last two weeks, the cheapest 3-hour block on a recent day, load-shifting
savings (custom and preset), the best forecasting model, two methodology questions, one in
German, and a question about 2019 (outside the data), which must be declined.

A case passes when the answer contains the right number (German or English number format,
"3.8 million" and "4 Mio." included) or the right facts, **and** the expected tool was called:
the right number without a tool call counts as a guess and fails. Results (answers, checks,
latency) go to `data/rag/eval_results.parquet`.

The offline tests run the whole harness with a rule-based stand-in for the LLM, so the
questions, tools and checks are known to be answerable; the live run measures the model.

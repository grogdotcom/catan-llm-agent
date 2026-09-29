# `catan_llm.llm.openai_batch` — CLI Reference

> **One tool for two workflows:** build a 24 h **Batch API** JSONL from the Catan corpora, *or* fire a **single immediate** `chat.completions` request and score it on the spot. Works identically for either corpus — `data/initial_placements/raw/initial_placements.jsonl` (all players, 8 placements / game) and `data/sft/high_decision_moves.jsonl` (winner high decisions) share the same `{prompt, completion, chosen_index, phase, color, game_id, decision_id}` schema, so you can pass one file or merge both.

```
python -m catan_llm.llm.openai_batch -i <corpus.jsonl> [-o batch.jsonl] [options]
python -m catan_llm.llm.openai_batch --single -i <corpus.jsonl> [selector] [options]
```

---

## Table of Contents

1. [Synopsis & Modes](#synopsis--modes)
2. [Quick Start](#quick-start)
3. [Input / Output & Corpus Selection](#input--output--corpus-selection)
4. [All Options — Common](#all-options--common)
5. [Batch Mode Options](#batch-mode-options)
6. [Batch Upload Mode (`--create-batch`)](#batch-upload-mode---create-batch)
7. [Single-Request Mode (`--single`)](#single-request-mode---single)
8. [System Prompt Handling](#system-prompt-handling)
9. [Request & Evaluation Format](#request--evaluation-format)
10. [Examples](#examples)
11. [Python API Equivalence](#python-api-equivalence)
12. [Environment Variables & Authentication](#environment-variables--authentication)
13. [Exit Codes & Errors](#exit-codes--errors)
14. [Tips & Troubleshooting](#tips--troubleshooting)

---

## Synopsis & Modes

```
# Batch: corpus → Batch JSONL (offline, no API key) → optionally upload → create batch
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o batch.jsonl
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o batch.jsonl --create-batch

# Single: pick ONE decision → chat.completions → immediate scoring
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --game-id 0 --decision-id 2 --dry-run
```

| Mode | Flag | Needs API key? | What it does |
|------|------|----------------|--------------|
| **Batch (default)** | *(no `--single`)* | No — except with `--create-batch` | Writes one JSONL line per record ready for `files.create` + `batches.create`. Ground truth preserved in `metadata._expected_completion` for offline scoring. |
| **Single** | `--single` | No for `--dry-run`, yes otherwise | Loads one record (by index / game+decision / phase / random), calls `chat.completions.create` immediately, parses `predicted_index`, returns `{correct, valid, expected vs predicted}`. |

Both modes accept **one or many** inputs: `-i a.jsonl` or `-i a.jsonl b.jsonl` (merged in order, `custom_id`s deduped).

---

## Quick Start

```bash
# 1) Batch — 400 placements → JSONL (no key)
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o placements.batch.jsonl
# 2) Batch — 292 high decisions → JSONL
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o high.batch.jsonl
# 3) Batch — merged
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl data/sft/high_decision_moves.jsonl -o merged.batch.jsonl

# 4) Single — dry-run: show what would be sent for one placement
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0 --dry-run

# 5) Single — live: score one decision immediately
export OPENAI_API_KEY=sk-...
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --phase MOVE_ROBBER --model gpt-4o-mini

# 6) Single — random decision with full prompt printed
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --random --seed 42 --dry-run --show-prompt-full
```

---

## Input / Output & Corpus Selection

- `-i, --input INPUT [INPUT ...]` **(required, common)** — One or more JSONL paths. Either corpus works; mixing them merges sequentially. Each line must be `{"prompt": str, "completion": str|int, ...}`. Blank lines skipped. Invalid JSON → `ValueError` with `file:line`.
- `-o, --output OUTPUT` **(required in batch mode, ignored in single)** — Destination for Batch JSONL. Parent dirs created. Existing file overwritten.

**Corpus contents:**
- `data/initial_placements/raw/initial_placements.jsonl` — 400 records (50 games × 8 placements), `phase=BUILD_INITIAL_SETTLEMENT`, `grouped=true`, `next_road` bundled.
- `data/sft/high_decision_moves.jsonl` — 292 records (winner only), phases: `PLAY_TURN` (179), `DISCARD` (59), `MOVE_ROBBER` (34), `BUILD_INITIAL_SETTLEMENT` (20). Includes `top_action_values` when available.

Both share keys: `prompt`, `completion` (=`chosen_index` as string), `chosen_label`, `phase`, `color`, `turn`, `num_moves`, `move_labels`, `selected_action`, `winner`, `game_id`, `decision_id`.

---

## All Options — Common

These apply to **both** batch and single (where noted, single ignores batch-only flags).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-h, --help` | — | — | Show help and exit. |
| `-i, --input` | `str+` | *required* | Input JSONL path(s). |
| `--model` | `str` | `gpt-4o-mini` | `body.model` for batch lines / `model` for single. Any chat model (e.g. `gpt-4o`, `gpt-4o-mini`, `o1`). |
| `--system-prompt` | `str` | *expert Catan prompt* | Custom system message. See [System Prompt Handling](#system-prompt-handling). |
| `--no-system-prompt` | flag | `false` | Omit system message entirely — only `user` prompt is sent. Overrides `--system-prompt`. |
| `--temperature` | `float` | `0.0` | Sampling temperature. `0` → deterministic (recommended for scoring). |
| `--max-tokens` | `int` | `16` | Max completion tokens. Corpus completions are a single integer, so `16` suffices; raise if you want reasoning. |
| `--endpoint` | `str` | `/v1/chat/completions` | Batch endpoint URL. Also used as reference in single dry-run preview. |
| `--extra-body-json` | JSON str | `null` | JSON merged into each `body` (e.g. `'{"response_format":{"type":"json_object"}}'`, `'{"reasoning_effort":"low"}'`). |
| `--api-key` | `str` | `null` (env) | OpenAI API key. If omitted, `OPENAI_API_KEY` env var is used when the SDK needs it. |

**Default system prompt:**
```
You are an expert Catan player. Select the best move from the available options.
Respond with only the move ID number (e.g. "42").
```

---

## Batch Mode Options

Only when **not** passing `--single`.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-o, --output` | `path` | *required* | Where to write the Batch JSONL. |
| `--limit` | `int` | `null` (=all) | Only emit first N requests (after `--offset`). Useful for pilot batches. |
| `--offset` | `int` | `0` | Skip N records before building. Combine with `--limit` for slicing. |
| `--custom-id-prefix` | `str` | `null` | Prefix prepended to every `custom_id` (e.g. `placements-`). Helps when merging files that reuse game IDs. Dedup suffix `--1`, `--2` auto-added if collisions remain. |
| `--strict` | flag | `false` | **Strict Batch spec:** omit the non-spec `metadata` key so each line is exactly `custom_id/method/url/body`. Use when a validator rejects extra keys. Without it, `metadata._expected_completion` etc. are included for offline scoring (OpenAI ignores unknown top-level keys). |
| `--completion-window` | `str` | `24h` | Only with `--create-batch` — passed to `batches.create(completion_window=…)`. |
| `--metadata-json` | JSON str | `null` | Only with `--create-batch` — string→string metadata for the Batch object (e.g. `'{"project":"catan","corpus":"placements"}'`). |
| `--create-batch` | flag | `false` | After writing the file, immediately `files.create` + `batches.create`. Requires API key. Prints `{prepare_info, file_id, batch_id}`. |

**Batch limits (warnings, not hard failures):** `>50,000` requests or `>200 MB` → `warnings` array in the printed JSON. Split with `--limit`/`--offset` or multiple files.

**Batch output — each line:**
```json
{
  "custom_id": "game-0-decision-0--WHITE-BUILD_INITIAL_SETTLEMENT",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are an expert..."},
      {"role": "user", "content": "[FULL BOARD MAP]..."}
    ],
    "temperature": 0,
    "max_tokens": 16
  },
  "metadata": {
    "_expected_completion": "56",
    "_expected_index": 56,
    "_phase": "BUILD_INITIAL_SETTLEMENT",
    "_color": "WHITE",
    "_game_id": 0,
    "_decision_id": 0,
    "_num_moves": 144
  }
}
```
`custom_id` = `game-{id}-decision-{id}--{COLOR}-{PHASE}` (or `request-{idx}` fallback). Stable and unique.

**What the CLI prints (batch mode):**
```json
{
  "output_path": "placements.batch.jsonl",
  "count": 400,
  "bytes": 4421325,
  "model": "gpt-4o-mini",
  "endpoint": "/v1/chat/completions",
  "warnings": []
}
```

---

## Batch Upload Mode (`--create-batch`)

Requires `pip install openai` and an API key. Equivalent Python:

```python
from catan_llm.llm.openai_batch import OpenAIBatchClient
client = OpenAIBatchClient(api_key="sk-...", model="gpt-4o-mini")
result = client.create_batch_from_corpus("data/initial_placements/raw/initial_placements.jsonl", output_path="batch.jsonl")
# result = {"prepare_info": {...}, "file_id": "file-abc", "batch_id": "batch-xyz", "batch": <obj>}
```

CLI:

```bash
export OPENAI_API_KEY=sk-...
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o batch.jsonl --create-batch
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o batch.jsonl --create-batch --api-key sk-... --metadata-json '{"project":"catan"}'
# check status later
# python -c "from catan_llm.llm.openai_batch import OpenAIBatchClient; print(OpenAIBatchClient().retrieve_batch('batch_...'))"
```

---

## Single-Request Mode (`--single`)

`--single` switches from bulk Batch to **one immediate** decision. You pick the record with **exactly one selector** (or none → `index=0`). Needs API key **unless** `--dry-run`.

### Selectors (mutually exclusive; first match wins)

| Flag | Type | Description | Example |
|------|------|-------------|---------|
| `--index` | `int` | 0-based positional index in iteration order. Supports `-1` for last. | `--index 0`, `--index -1` |
| `--game-id` (+ `--decision-id`) | `int` | Filter by `game_id` / `decision_id`. Use both for exact match, one for first match of that game/decision. | `--game-id 0 --decision-id 1` |
| `--phase` | `str` | First record with this phase. Common phases: `BUILD_INITIAL_SETTLEMENT`, `MOVE_ROBBER`, `PLAY_TURN`, `DISCARD`. For `initial_placements` only settlement exists. | `--phase MOVE_ROBBER` |
| `--random` | flag | Uniform random pick. Combine with `--seed` for determinism. | `--random --seed 42` |
| `--seed` | `int` | Seed for `--random`. | `--seed 123` |

Mixing `--input` files: selectors scan in the order given (first match across the concatenated corpora).

### Single-specific flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--single` | flag | `false` | Enable single mode. |
| `--dry-run` | flag | `false` | **No API call:** print what *would* be sent + expected answer. Ideal for preview / CI. Shows `messages_preview` (first 500 chars per message) and expects. |
| `--show-prompt` | flag | `false` | Also print the full prompt (truncated to 2000 chars) to stdout. In dry-run/live, also adds `prompt` to JSON. |
| `--show-prompt-full` | flag | `false` | Print **entire** prompt without truncation (can be 8–12 k chars). |
| `--api-key` | — | — | Same as batch — key for live call. Ignored if `--dry-run`. |

**Single dry-run output (`--dry-run`):**
```json
{
  "custom_id": "game-0-decision-0--WHITE-BUILD_INITIAL_SETTLEMENT",
  "positional_index": 0,
  "model": "gpt-4o-mini",
  "messages_preview": [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "[FULL BOARD MAP - 19 HEXES]..."}
  ],
  "expected_index": 56,
  "expected_completion": "56",
  "expected_label": "Settlement Node 18 [6-Wd, 9-Br, 10-Sh | 12p] | Road (17, 18) -> Target N15...",
  "phase": "BUILD_INITIAL_SETTLEMENT",
  "color": "WHITE",
  "game_id": 0,
  "decision_id": 0,
  "num_moves": 144
}
```

**Single live output (`--single` without `--dry-run`):**
```json
{
  "custom_id": "game-0-decision-2--ORANGE-MOVE_ROBBER",
  "positional_index": 2,
  "expected_index": 8,
  "expected_completion": "8",
  "expected_label": "Move robber to Tile 4: 5 SHEEP ... and steal from RED",
  "predicted_text": "8",
  "predicted_index": 8,
  "predicted_label": "Move robber to Tile 4: 5 SHEEP ... and steal from RED",
  "correct": true,
  "valid": true,
  "num_moves": 21,
  "phase": "MOVE_ROBBER",
  "color": "ORANGE",
  "game_id": 0,
  "decision_id": 2,
  "model": "gpt-4o-mini",
  "usage": {"prompt_tokens": 2400, "completion_tokens": 2, "total_tokens": 2402}
}
```
Plus human summary to stdout:
```
✓ CORRECT — predicted 8 vs expected 8 (phase=MOVE_ROBBER color=ORANGE valid=True)
  expected: Move robber to Tile 4: 5 SHEEP (4 pips) ... and steal from RED
  predicted: Move robber to Tile 4: 5 SHEEP (4 pips) ... and steal from RED
```
`✗ WRONG` when mismatch, `valid=false` when predicted `null` or out of `1..num_moves`.

---

## System Prompt Handling

| Invocation | System message? | Content |
|------------|-----------------|---------|
| *(default)* | Yes | Expert Catan prompt above. |
| `--system-prompt "Custom..."` | Yes | Your string. |
| `--no-system-prompt` | **No** | Only `user` prompt (corpus prompt already ends with `[DECISION REQUIRED]`). |
| `--system-prompt ""` | No | Treated as empty → omitted (same as `--no-system-prompt`). |

The corpus prompt itself always contains the full six-section prompt (`[FULL BOARD MAP]` → `[PLAYABLE MOVES]`) plus footer (`Select the best action…` or `The Grandmaster engine has selected Move ID 56…` for placements). System prompt is additive.

---

## Request & Evaluation Format

**Parsing:** `parse_predicted_index` extracts the first integer via regex `\d+`. If `num_moves` known, it prefers the first integer in `1..num_moves` (scanning left→right), so reasoning like `"I considered 100 but pick 3"` yields `3` if `3` is valid.

**Evaluation (`evaluate_response` / single mode):**
```python
{
  "expected_index": int(completion),   # 1-indexed
  "predicted_index": int|None,         # parsed or None
  "correct": predicted == expected,
  "valid": predicted is not None and 1 <= predicted <= num_moves,
  "expected_label": move_labels[expected-1],
  "predicted_label": move_labels[predicted-1]  # if valid
}
```

---

## Examples

All examples assume repo root with the two default corpora.

### Batch — offline (no key)

```bash
# Basic: placements → batch
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/placements.batch.jsonl

# High decisions → batch, custom model
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o /tmp/high.batch.jsonl --model gpt-4o

# Merged corpora
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl data/sft/high_decision_moves.jsonl -o /tmp/merged.batch.jsonl

# Slice: first 10 of high decisions
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o /tmp/pilot.batch.jsonl --limit 10

# Paginate: next 10
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o /tmp/pilot2.batch.jsonl --offset 10 --limit 10

# Strict spec (no metadata) + prefix for dedup
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl data/sft/high_decision_moves.jsonl -o /tmp/strict.batch.jsonl --strict --custom-id-prefix m1-

# Custom body (e.g. JSON mode)
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/json.batch.jsonl --extra-body-json '{"response_format":{"type":"json_object"}}'

# No system prompt (leaner context)
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/nosys.batch.jsonl --no-system-prompt

# Different endpoint / window (edge)
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/alt.batch.jsonl --endpoint /v1/chat/completions --completion-window 24h
```

### Batch — upload & create

```bash
export OPENAI_API_KEY=sk-...
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/batch.jsonl --create-batch
python -m catan_llm.llm.openai_batch -i data/sft/high_decision_moves.jsonl -o /tmp/batch.jsonl --create-batch --metadata-json '{"corpus":"high","version":"1"}'
python -m catan_llm.llm.openai_batch -i data/initial_placements/raw/initial_placements.jsonl -o /tmp/batch.jsonl --create-batch --api-key sk-... --model gpt-4o-mini --temperature 0 --max-tokens 32
```

### Single — dry-run (no key, works offline)

```bash
# First placement
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0 --dry-run

# By game/decision composite key
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --game-id 0 --decision-id 1 --dry-run

# First MOVE_ROBBER from high decisions
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --phase MOVE_ROBBER --dry-run

# First DISCARD
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --phase DISCARD --dry-run

# Random with seed (reproducible)
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --random --seed 42 --dry-run

# Random from merged corpora
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl data/sft/high_decision_moves.jsonl --random --dry-run

# With prompt preview
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0 --dry-run --show-prompt
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0 --dry-run --show-prompt-full | less

# Different model in preview
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --index 5 --dry-run --model gpt-4o --no-system-prompt
```

### Single — live (scored)

```bash
export OPENAI_API_KEY=sk-...

python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --index 0 --model gpt-4o-mini
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --phase MOVE_ROBBER --model gpt-4o-mini
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl --game-id 2 --decision-id 0 --model gpt-4o --temperature 0 --max-tokens 16 --show-prompt
python -m catan_llm.llm.openai_batch --single -i data/sft/high_decision_moves.jsonl --random --seed 123 --model gpt-4o-mini
# Low temp deterministic scoring
python -m catan_llm.llm.openai_batch --single -i data/initial_placements/raw/initial_placements.jsonl data/sft/high_decision_moves.jsonl --phase PLAY_TURN --temperature 0 --max-tokens 8 --model gpt-4o-mini
```

---

## Python API Equivalence

Every CLI flag has a direct Python call. Useful for notebooks / tests with mocked clients.

```python
from catan_llm.llm.openai_batch import (
    build_batch_jsonl, build_batch_requests, build_chat_messages,
    parse_predicted_index, evaluate_response, load_single_record, OpenAIBatchClient
)

# Batch equivalents
build_batch_jsonl("data/initial_placements/raw/initial_placements.jsonl", "out.jsonl", model="gpt-4o-mini", limit=10, strict=True)
build_batch_requests("data/sft/high_decision_moves.jsonl", offset=5, limit=5, custom_id_prefix="pilot-")
client = OpenAIBatchClient(model="gpt-4o-mini")
client.prepare_batch_file("data/sft/high_decision_moves.jsonl", "out.jsonl")
client.create_batch_from_corpus("data/initial_placements/raw/initial_placements.jsonl")  # needs key

# Single equivalents
rec, pos = load_single_record("data/initial_placements/raw/initial_placements.jsonl", index=0)
rec, pos = load_single_record("data/sft/high_decision_moves.jsonl", game_id=0, decision_id=2)
rec, pos = load_single_record("data/sft/high_decision_moves.jsonl", phase="DISCARD")
rec, pos = load_single_record(["data/initial_placements/raw/initial_placements.jsonl","data/sft/high_decision_moves.jsonl"], random_pick=True, seed=42)
build_chat_messages(rec, system_prompt=None)
parse_predicted_index("I think 56 is best", num_moves=144)  # -> 56
evaluate_response("56", rec)  # -> {correct, valid, ...}
client = OpenAIBatchClient(api_key="sk-...")
client.evaluate_single(rec)
client.evaluate_single_from_corpus("data/sft/high_decision_moves.jsonl", phase="MOVE_ROBBER")
```

Mocking (tests/offline):

```python
from unittest.mock import MagicMock
mock = MagicMock()
mock.chat.completions.create.return_value.choices[0].message.content = "56"
client = OpenAIBatchClient(client=mock)
client.evaluate_single(rec)  # no network
```

---

## Environment Variables & Authentication

| Variable | Used when | Priority |
|----------|-----------|----------|
| `OPENAI_API_KEY` | `client` not passed + any live call (`--create-batch` or `--single` without `--dry-run`) | `--api-key` flag overrides env |

If neither is set and a live call is attempted, `openai.OpenAI` will raise an authentication error. Offline modes (`prepare_batch_file`, `build_batch_jsonl`, `dry-run`) need **no** key or `openai` install.

Install SDK only for live paths: `pip install openai`

---

## Exit Codes & Errors

| Situation | Exit | Message |
|-----------|------|---------|
| `--output` missing in batch mode | `2` (argparse) | `--output is required for batch mode (or use --single)` |
| No records match selector (`--index` out of range, no `game_id`, unknown `phase`) | `2` | `no record found for phase='...'` / `corpus index 99 out of range (size 50)` |
| Input file not found | `1` (exception) | `corpus file not found: ...` |
| Invalid JSONL line | `1` | `invalid JSON at file:line: ...` |
| Record missing `prompt`/`completion` | `1` | `record missing non-empty string field 'prompt'` |
| Empty corpus after `--offset`/`--limit` | `1` | `no batch requests built — corpus is empty ...` |
| `openai` not installed on live path | `1` | `the 'openai' package is required ... pip install openai` |
| Batch limits exceeded | `0` + `warnings` array | Does **not** fail — prints warning to split. |

---

## Tips & Troubleshooting

- **Which corpus for what?** Placements → test spatial planning; high decisions → full-game strategy (robber, discard, trades, builds). Merging is valid — use `--custom-id-prefix` if you worry about collisions, though dedup is automatic.
- **Context length:** Prompts are 2–12 k chars (placements with history are longest). Batch JSONL for 400 placements ≈ 4.4 MB; high decisions ≈ 5.0 MB; merged ≈ 9.4 MB — well under 200 MB limit.
- **Scoring:** Batch results: join `custom_id` → `metadata._expected_completion`, parse `choices[0].message.content` with same `parse_predicted_index` logic; single mode does this for you.
- **`--strict` when?** Only if a validator or older Batch importer rejects extra top-level keys. Otherwise leave off to keep ground truth alongside requests.
- **`--extra-body-json` quoting:** Use single quotes around JSON: `--extra-body-json '{"response_format":{"type":"json_object"}}'`.
- **Determinism:** `--temperature 0` + `--seed` for `--random` gives reproducible runs.
- **Full prompt inspection:** `--show-prompt-full | less` or redirect: `--dry-run --show-prompt-full > prompt.txt`.
- **Module path:** Always run as `python -m catan_llm.llm.openai_batch` (package-aware) or `PYTHONPATH=src python -m ...` if not installed.

---

## See Also

- Source: `src/catan_llm/openai_batch.py` (docstring + `_cli`)
- `python -m catan_llm.llm.openai_batch --help` — canonical flag list
- Corpus builders: `src/catan_llm/collect_corpus.py` / `collect_placements.py`
- Prompt docs: `src/catan_llm/format/prompts.py` (`get_complete_prompt` six-section order)

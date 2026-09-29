# Project Guidelines for AI Agents

## Backwards Compatibility

**We do not need backwards compatibility.**

When refactoring code or making changes, do not worry about maintaining backwards compatibility with previous versions. Focus on:
- Clean, maintainable code
- Proper separation of concerns
- Comprehensive test coverage
- Modern best practices

This allows for more aggressive refactoring and cleaner code architecture without being constrained by legacy requirements.

## Data Layout — Do Not Scan Every JSONL

All corpora live under `data/` — **no symlinks in repo root**. Read `data/README.md` (the data catalog) instead of globbing `*.jsonl`.

```
data/initial_placements/raw/      # 8 rec/game — initial_placements_2000.jsonl (2000 games, 16k, annotated placement_round) is PRIMARY
data/initial_placements/raw/*_sampled2.jsonl  # 2 random players/game (seed 12345) → 8k (4000+4000) — batch source
data/initial_placements/phase1/{batch,results}/  # 4000 first settlements (R1) — batch as sent + Batch output
data/initial_placements/phase2/{batch,results,augmented}/  # 4000 second settlements with injected [CURRENT STRATEGY] + combined 8k
data/sft/
  high_decision_moves.jsonl        # winner high decisions (AlphaBeta depth2, friendly_robber=False)
  trajectory_manifest.jsonl        # provenance per game (trajectory_id, game_seed, game_end_turn, seat_order)
  sft_runs.db                      # SQLite run store — source of truth (runs/trajectories/checkpoints/batch_chunks/outputs + SFT side table)
  batches/<run_id>/ckpt<N>_chunk*.batch.jsonl  # per-epoch batch files (~100 req/chunk, /v1/responses)
  results/<run_id>_ckpt<N>_*.jsonl           # raw Batch outputs (hashed in SQLite)
  sft/sft_*.jsonl                             # accepted SFT exports (full reasoning, split 80/10/10)
data/executor/
  batches/<run_id>/ckpt<N>_chunk*.batch.jsonl  # generic executor batch files (pipeline-agnostic)
  results/<run_id>_ckpt<N>_*.jsonl             # generic results
data/luna/                                    # 10-game eval subsets
src/catan_llm/llm/sft/            # LLM SFT pipeline (selection/validation/provenance/cli + side table/export)
  selection.py, validation.py, provenance.py, cli.py, side_table.py
src/catan_llm/executor/           # generic executor (store/runner/spec — pipeline-agnostic)
  store.py, runner.py, spec.py
batch_sft.py                      # CLI wrapper: prepare/submit/poll/resume/export/status (SFT)
```

## Deep Module Layout

The package is organized into deep modules (small interface, hidden implementation), each with a clean seam:

| Module | Interface | Hides |
|--------|-----------|-------|
| `catan_llm/format` | renderers for board/players/history/moves/prompts | Catanatron public-state details, ActionType registry, section ordering |
| `catan_llm/llm/decision` | `DecisionSurface.present()/.parse()/.plan()` + `MoveExecutor` | LLM-facing move numbering, compound moves, `AUTO_ROAD` resolution, parsing |
| `catan_llm/llm/prompt` | `PromptBuilder.build(PromptContext) -> PromptArtifact`; strategy block utilities | LLM prompt composition and strategy blocks |
| `catan_llm/llm/teacher` | `TeacherGateway`, request builders, response parsing | model registry, chat/Responses payloads, transport and parsing |
| `catan_llm/llm/strategy` | `resolve_lineage(...) -> LineageDecision` | LLM strategy-chain policy |
| `catan_llm/llm/dataset` | `build_sft_record(...)` | accepted SFT record and split construction |
| `catan_llm/llm/evaluation` | `evaluation.metrics` + `evaluation.report` | LLM/SFT acceptance, accuracy, and run reporting |
| `catan_llm/executor` | `RunStore` + `RunExecutor` + `RunSpec` | generic durable executor and transport orchestration |
| `catan_llm/llm/sft` | `cli` + `selection`/`validation`/`provenance` + `side_table` | LLM SFT policy using the generic executor |

Dependency rules (enforced by review, not by tooling):
- `format` never imports `llm`/SQLite.
- `llm.decision`/`llm.prompt` depend on shared `format` + `domain`.
- LLM pipeline modules use `llm.teacher` and `llm.prompt.strategy` for request/response handling and strategy blocks.
- `llm.openai_batch` is a facade + CLI over `catan_llm.llm.teacher`.
- `RunStore` is generic execution infrastructure; the LLM SFT side table joins it and owns SFT policy.

The top-level `catan_llm/__init__.py` re-exports the formatting surface for ergonomics; all new code should import from the deep modules directly.

## Existing Scripts

- `batch_two_phase.py` — LLM placement strategy carryover (uses `catan_llm.llm.prompt` + `teacher`).
- `batch_sft.py` — wrapper for `catan_llm.llm.sft.cli`.
- `catan_llm.llm.collect_corpus` / `collect_1000_placements.py` — AlphaBeta simulation + corpus JSONL (omitted from coverage; rerun via `--help`).

* Initial placements: `decision_id 0-3` = R1 first (`placement_round=1`), `4-7` = R2 second reverse. Per player one of each. `collect_1000_placements.py` defaults to `data/initial_placements/raw/...`, annotates `placement_round`/`is_first_placement`, and supports `--sample-players 2 --sample-seed 12345`.
* Two-phase batch: `batch_two_phase.py` builds `phase1` from sampled R1 (`[CURRENT STRATEGY] None` normalized), sends to `POST /v1/batches` (`endpoint /v1/responses`, `model gpt-5.6-luna`, `reasoning.effort=medium`, `max_output_tokens=4096`, requires `<think><strategy><action>`), parses `<strategy>` per `(game_id,color)` and injects for that player's R2 before `[RECENT TURNS]`, then builds `phase2` batch. Phase files are derived from the corpus stem via `phase_paths_for()` so 1000 vs 2000 stay isolated.

## Testing & Coverage

Run tests with `pytest` (see `pyproject.toml` for `pythonpath = ["src"]`):

```bash
venv/bin/python -m pytest -q          # all tests
venv/bin/python -m pytest tests/format -q  # format package only
```

Coverage is enforced in CI (`.github/workflows/ci.yml`) and locally via `make`:

```bash
make test              # pytest -q
make coverage          # html + xml reports
make coverage-check    # gate: --cov-fail-under=80 (mirrors CI)
open htmlcov/index.html
```

Config: `pyproject.toml` `[tool.coverage.*]` — `branch = true`, `source = ["src/catan_llm"]`, omits `collect_corpus.py` + deprecated shim, `fail_under = 80`. The gate (`make coverage-check` / CI) runs the full suite against `src/catan_llm` plus a `tests/format`-only check against `src/catan_llm/format`. Current baseline: full suite ~82% combined (`format` alone ~97.5%).

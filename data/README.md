# Data Catalog

> **For future agents:** Read this file instead of scanning every JSONL. It partitions the corpus so you can pick the slice you need without loading everything.

## Layout
```
data/
  initial_placements/          # 2-placement opening draft (settlement+road bundled as one Move)
    raw/                       # Full AlphaBeta self-play corpora (8 records/game)
      initial_placements.jsonl                 # LEGACY 50 games, 400 rec, no round annotation (pre-strategy)
      initial_placements_1000.jsonl            # 1000 games, 8000 rec, annotated (placement_round, is_first_placement)
      initial_placements_2000.jsonl            # ★ PRIMARY 2000 games, 16000 rec, annotated — source for batches
      initial_placements_1000_sampled2.jsonl   # 1000 games sampled 2 players/game → 4000 rec (2000+2000) — legacy sample
      initial_placements_2000_sampled2.jsonl   # ★ SAMPLED 2000 games sampled 2 players/game (seed 12345) → 8000 rec (4000+4000) — batch source
    phase1/
      batch/
        initial_placements_2000_phase1.batch.jsonl              # ★ 4000 first settlements (R1, decision 0-3, sampled) as sent to OpenAI
        initial_placements_2000_phase1.batch.jsonl.mapping.json # custom_id → {game_id,color,expected_completion}
        initial_placements_1000_phase1.batch.jsonl              # legacy 1000 subset (280-4000, high vs medium)
      results/
        initial_placements_2000_phase1_results.jsonl            # ★ 4000 Batch outputs for phase1 (gpt-5.6-luna, medium, /v1/responses, max 4096)
        initial_placements_1000_phase1_mock_results.jsonl       # mock/demo only
    phase2/
      batch/
        initial_placements_2000_phase2.batch.jsonl              # ★ 4000 second settlements (R2, decision 4-7) with injected [CURRENT STRATEGY]
        initial_placements_1000_phase2.batch.jsonl              # legacy 1000 subset
      results/
        initial_placements_2000_phase2_results.jsonl            # 4000 outputs for phase2 (strategy-refined, medium) — completed 4000/4000 via 10×400 + 24×100
      augmented/
        initial_placements_2000_with_strategy.jsonl             # ★ 8000 combined (4000 first + 4000 second) — second prompts already contain injected strategy + original_prompt
        initial_placements_1000_with_strategy.jsonl             # legacy 1000 subset
  sft/
    high_decision_moves.jsonl                  # Winner high-decision corpus (initial+robber+build+ambiguous <0.05), AlphaBeta depth2, friendly_robber=False
    trajectory_manifest.jsonl                  # Provenance: trajectory_id, game_seed, game_end_turn, seat_order, winner (generated alongside high_decision_moves)
    sft_runs.db                                # SQLite run store (runs, trajectories, checkpoints, batch_chunks, outputs + SFT side table) — not committed, regenerated via batch_sft.py
    batches/<run_id>/ckpt<N>_chunk*.batch.jsonl # Per-epoch batch request files (~100/ chunk, Responses API, max_output_tokens 4096, reasoning medium)
    results/<run_id>_ckpt<N>_*_results.jsonl   # Raw Batch outputs (immutable artifacts, hashed in SQLite)
    sft/sft_*.jsonl                             # Accepted SFT exports (full <think><strategy><action> + engine_completion + strategy_in/out + split)
  executor/
    batches/<run_id>/ckpt<N>_chunk*.batch.jsonl # Generic executor batch files (pipeline-agnostic, used via data/executor/)
    results/<run_id>_ckpt<N>_*_results.jsonl   # Generic results
  luna/
    luna_first_settlements.jsonl               # 10 games, 40 first settlements, gpt-5.6-luna eval (no high reasoning)
    luna_high_first_settlements.jsonl          # same 40, reasoning high (8192 tokens)

src/catan_llm/llm/sft/                         # LLM SFT checkpoint pipeline (see AGENTS.md)
  selection.py        # collapse (traj,seat,turn,phase) + K-by-length (4/5/6) + band midpoint selection, skip empty
  validation.py       # gates: <think>/<strategy>/<action> + range + engine match + single [CURRENT STRATEGY] + lineage
  provenance.py       # derive_trajectory_id, seat_index_for_color, sample_winning_trajectories (stratified)
  side_table.py       # SFT side table (band/progress) + export 80/10/10 via JOIN
  cli.py              # prepare / submit / poll / resume / export / status (SFT)
src/catan_llm/executor/                        # generic executor (see AGENTS.md)
  store.py            # RunStore — generic durable store (runs/trajectories/checkpoints/batch_chunks/outputs)
  runner.py           # RunExecutor — ~100/chunk, concurrent submit, poll/resume, epoch gate
  spec.py             # RunSpec / RetryPolicy
batch_sft.py                                   # wrapper for cli.py (PYTHONPATH=src)
```

## Key Concepts

**Initial placements are 8 per game, 2 per player.**
* `decision_id 0-3` = first round (forward draft, `placement_round=1`, `is_first_placement=True`)
* `decision_id 4-7` = second round (reverse draft, `placement_round=2`, `is_first_placement=False`)
* Per player: one first + one second (e.g. WHITE appears at decision 0 and 7 in game 0). `placement_number_for_player` mirrors `placement_round`.

**Sampling for batches:** To get 2000 games → 4000+4000 while staying under OpenAI's 5M enqueued-token limit, we randomly sample **2 players per game** (`Random(12345+game_id).sample(colors,2)`, deterministic).  
* Full 2000 → 16k rec, sampled → 8k rec (4000 R1 + 4000 R2).  
* `sampled=True`, `sampled_players_for_game`, `sample_seed` added per record.  
* Batch files are built from the sampled R1 (phase1) and later augmented R2 (phase2).

**Two-phase pipeline (strategy carryover):**
1. Phase1 prompts are normalized to contain `[CURRENT STRATEGY]\nNone` (legacy corpus lacked this block — `batch_two_phase.py:inject_strategy_into_prompt` inserts it before `[RECENT TURNS]`).
2. Those 4000 R1 prompts are sent to `POST /v1/batches` (`endpoint /v1/responses`, `model gpt-5.6-luna`, `reasoning.effort=medium`, `max_output_tokens=4096`, system prompt requires `<think><strategy><action>`).
3. Batch outputs are parsed for `<strategy>…</strategy>` per `(game_id,color)` and injected as `[CURRENT STRATEGY]\n<strategy text>` for that player's R2 prompt (again before `[RECENT TURNS]`).  
4. Phase2 prompts (now with real strategy) are sent as a second batch. The strategy must *refine/pivot* the prior plan and declare the locked-in archetype (OWS/Dev, Wood/Brick, Port Monopoly).

**What to read:**
* **To understand the task:** `data/initial_placements/raw/initial_placements_2000_sampled2.jsonl` (one record shows `prompt` + `completion` + `placement_round` + `color` + `game_id`).
* **To reproduce batch:** `data/initial_placements/phase1/batch/*.batch.jsonl` (the exact JSONL sent to `POST /v1/files` → `POST /v1/batches`) and `…/results/*.jsonl` (the Batch `output_file_id` download, each line `{custom_id, response:{status_code, body:{output_text, output:[…]}}}`).
* **To train/eval without re-running batches:** `data/initial_placements/phase2/augmented/initial_placements_2000_with_strategy.jsonl` (8000, second half already has `injected_strategy` and `original_prompt`).
* **SFT is separate** — `data/sft/high_decision_moves.jsonl` is winner-only, high-value decisions (not initial placements). Don't mix with `initial_placements` unless you need full trajectory.
* **SFT checkpoints (strategy chaining):** See `docs/sft_pipeline.md`. Pipeline: `batch_sft.py` / `python -m catan_llm.llm.sft.cli prepare --corpus data/sft/high_decision_moves.jsonl --phase2-strategies data/initial_placements/phase2/augmented/... --sample-size 500` creates SQLite run, stratifies 500 winners by seat (≈125/seat, fixed seed), collapses same `(traj,seat,turn,phase)`, computes K=4/5/6 by `game_end_turn`, divides progress into K bands and picks nearest to midpoint (skipping empty bands), chains `checkpoint1←phase2` and `checkpoint N←N-1 accepted` (invalid chains stop, no stale reuse). Batch orchestration: ~100/chunk, all chunks/epoch concurrent, persists `request_path/hash`, `batch_id`, `result_path/hash`, polls without duplicates, dry-run without API key, and blocks next epoch until validated. Validation: `<think>`/`<strategy>` non-empty, `<action>` parseable + in range + equals AlphaBeta, exactly one `[CURRENT STRATEGY]`, lineage same traj/seat and `strategy_source_checkpoint < target`. Export preserves raw teacher output + `strategy_in/out` + lineage metadata + `split` (80/10/10, seed 42).

## Quick Checks
```bash
wc -l data/initial_placements/raw/initial_placements_2000.jsonl # 16000
wc -l data/initial_placements/raw/initial_placements_2000_sampled2.jsonl # 8000
wc -l data/initial_placements/phase1/results/initial_placements_2000_phase1_results.jsonl # 4000
wc -l data/initial_placements/phase2/results/initial_placements_2000_phase2_results.jsonl # 4000
# prompt contains strategy block?
grep -c "\[CURRENT STRATEGY\]" data/initial_placements/phase1/batch/initial_placements_2000_phase1.batch.jsonl # 4000
```

No symlinks in repo root — all data lives under `data/`; update any legacy `initial_placements.jsonl` or `high_decision_moves.jsonl` root references to the `data/...` paths above. Scripts default to the `data/` locations (e.g. `collect_1000_placements.py --out data/initial_placements/raw/...`, `batch_two_phase.py --corpus data/initial_placements/raw/...`).

## Midgame Provenance Prerequisite

Existing `high_decision_moves.jsonl` (10-game legacy, 292 rec) lacks `trajectory_id`/`game_seed`/`game_end_turn`/`seat_index`. The pipeline synthesizes deterministic provenance (`derive_trajectory_id(seed, seat_order)`, `seed=1000+game_id`) for backward compat, but production should regenerate via `python -m catan_llm.llm.collect_corpus --help` / `run_simulation(..., base_seed=..., manifest_path=...)` which now emits `trajectory_manifest.jsonl` alongside the corpus and annotates each record. The 4,000-game source for the 500-winner pilot should be regenerated before sampling, or a manifest mapping must be supplied via `--phase2-strategies` with trajectory-aware keys.

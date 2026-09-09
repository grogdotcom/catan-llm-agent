# Midgame SFT Strategy Checkpoints

**Status:** Approved design specification

**Purpose:** Generate a cost-controlled midgame SFT corpus by querying the LLM at
sparse, high-decision checkpoints while carrying forward the player's previous
strategy as synthetic strategic memory.

## 1. Context

The current midgame corpus is produced from AlphaBeta self-play. It contains
high-decision snapshots rather than every game tick, but its prompts do not yet
contain a propagated `[CURRENT STRATEGY]` block. The initial-placement pipeline
already generates phase-2 strategies that can seed a player's midgame strategy
chain.

The LLM must not be called for every simulated action. The simulator continues
to play the complete trajectory with the engine, while LLM calls are limited to
selected checkpoint decisions.

The initial pilot samples 500 winning player trajectories from the existing
4,000-game source population. The corpus may be expanded after measuring the
accepted record count and output quality.

## 2. Goals

- Generate a high-quality midgame reasoning corpus at substantially lower cost
  than querying every tick.
- Carry each player's previous strategy into the next selected checkpoint.
- Keep strategy lineage deterministic, auditable, and isolated per player.
- Preserve the full `<think><strategy><action>` teacher response for SFT.
- Run sequential checkpoint dependencies through durable OpenAI Batch API state.
- Allow the pilot to resume after process interruption without duplicate batches.

## 3. Non-goals

- This corpus is not intended to measure gameplay generalization.
- This corpus does not model a true LLM policy trajectory. Game actions come from
  the AlphaBeta simulation, and strategy text is sparse teacher-generated memory.
- The pipeline does not require an event-triggered strategy refresh in the first
  version.
- The pipeline does not query the LLM for ordinary, low-value ticks.

## 4. Terminology

### Trajectory

A complete simulated game, identified independently of the local JSONL row
number. A trajectory includes the random seed, final turn count, player seat
mapping, and game outcome.

### Winning player trajectory

The trajectory view for the player who won the simulated game. The initial pilot
uses winner-only decisions, matching the current `high_decision_moves.jsonl`
corpus.

### Decision opportunity

A unique high-decision opportunity for one player at one completed turn and one
phase. Raw records that represent the same `(trajectory, player, turn, phase)`
opportunity are collapsed for checkpoint-density calculations.

### Checkpoint

One selected decision opportunity at which the LLM receives the current board,
the previous strategy, and the selected AlphaBeta action target.

### Strategy chain

The ordered sequence of strategies for one `(trajectory_id, seat_index)` pair:

```text
initial phase-2 strategy -> checkpoint 1 -> checkpoint 2 -> ...
```

### Epoch

One checkpoint position across all sampled trajectories. All requests in an epoch
can run concurrently. The next epoch cannot start until the current epoch has
completed and passed validation.

## 5. Data Provenance

Initial-placement phase-2 strategies and midgame snapshots must come from the
same simulated trajectories. A local `game_id` is not sufficient as a join key.

The simulation must persist trajectory metadata in the run store and export a
trajectory manifest with at least:

```json
{
  "trajectory_id": "seed-123456-seat-order-hash",
  "game_seed": 123456,
  "game_end_turn": 72,
  "seat_order": ["RED", "BLUE", "ORANGE", "WHITE"],
  "winner": "BLUE",
  "players": [
    {"seat_index": 0, "color": "RED"},
    {"seat_index": 1, "color": "BLUE"},
    {"seat_index": 2, "color": "ORANGE"},
    {"seat_index": 3, "color": "WHITE"}
  ]
}
```

Every decision record must include or be joinable to:

```text
trajectory_id
game_seed
game_id                 # local display/index only
seat_index
color
game_end_turn
turn
trajectory_index
```

The current JSONL records contain local `game_id` and `color`, but do not expose
the required immutable seed and seat metadata. The implementation must either
add a manifest mapping for the existing 4,000 trajectories or regenerate the
trajectory metadata before building the midgame batches.

## 6. Storage Architecture

Use SQLite as the durable control plane. Keep JSONL for OpenAI Batch API
interchange and immutable raw-result artifacts. Generate Parquet only as an
optional analytics export; it is not the workflow store.

The SQLite database is the source of truth for selection, strategy lineage,
batch progress, validation, and dataset membership. Raw JSONL files are
reproducible artifacts associated with database rows, not mutable state.

### 6.1 Store Interface

The implementation should expose one small `MidgameRunStore` interface. The
SQLite implementation is used in production; an in-memory SQLite database is
used by tests.

The interface should support:

```text
create_run(config)
upsert_trajectory(metadata)
upsert_decision_opportunity(record)
select_checkpoints(run_id)
pending_batch_chunks(run_id, checkpoint_index)
record_batch_submission(chunk_id, batch_id)
import_batch_results(chunk_id, result_path)
advance_checkpoint(run_id)
export_dataset(run_id, output_path)
```

Callers must not update checkpoint or batch status by editing JSONL. State
transitions occur through the store and are committed transactionally.

### 6.2 Logical Tables

The database should contain at least:

```text
runs
  run_id, source_version, sample_seed, model, reasoning_effort,
  chunk_size, status, created_at, updated_at, config_json

trajectories
  trajectory_id, game_seed, game_id, game_end_turn, winner,
  seat_order_json, source_hash

players
  trajectory_id, seat_index, color, is_winner

decision_opportunities
  opportunity_id, trajectory_id, seat_index, color, turn, phase,
  trajectory_index, normalized_progress, prompt_json, engine_completion,
  source_record_json

checkpoints
  checkpoint_id, run_id, trajectory_id, seat_index, checkpoint_index,
  checkpoint_count, band, target_progress, opportunity_id, strategy_in,
  strategy_source_checkpoint, status, skip_reason

batch_chunks
  chunk_id, run_id, checkpoint_index, chunk_index, request_path,
  request_hash, openai_batch_id, result_path, status, submitted_at,
  completed_at, accepted_count, rejected_count

outputs
  checkpoint_id, raw_output_path, raw_output_hash, think_text,
  strategy_out, predicted_action, validation_status, rejection_reason,
  parsed_at

dataset_membership
  checkpoint_id, split, split_seed, export_status
```

Large prompt and raw response text may remain in JSON fields or immutable
artifacts referenced by paths. The store must retain hashes for referenced raw
files so artifacts can be verified after a resume or export.

### 6.3 Transaction Boundaries

The following operations must be atomic:

- Creating a run and its sampled trajectory set.
- Persisting a selected checkpoint and its `strategy_in` lineage.
- Recording an OpenAI batch submission and its request-file hash.
- Importing and validating a result chunk.
- Marking an epoch complete and making the next epoch eligible.
- Assigning final train/validation/test membership.

An incomplete transaction must leave the prior state intact. A process restart
must be recoverable from SQLite without scanning every JSONL file.

## 7. Pilot Sampling

The pilot samples 500 winning player trajectories from the 4,000-game source
population.

Sampling requirements:

- Use a fixed random seed.
- Stratify approximately evenly by `seat_index`.
- Target approximately 125 winning trajectories per seat for four seats.
- Preserve `trajectory_id`, `seat_index`, `color`, and `game_seed` in the sample
  manifest.
- Sample only winners for the first version.

The sample size is a starting point, not a hard final corpus size. Depending on
game lengths and accepted responses, the pilot is expected to produce roughly
2,000-3,000 midgame records. Expand the sampled winner population if the final
accepted corpus is below the desired SFT size.

## 8. Candidate Construction

Start with the existing high-decision eligibility filter. Do not redefine the
engine's high-decision semantics for the pilot.

Exclude records from the initial-placement phase when constructing midgame
candidate opportunities. Initial placement is used only to provide the phase-2
strategy seed.

Collapse raw records by:

```text
(trajectory_id, seat_index, turn, phase)
```

This prevents clusters such as several discard actions at the same turn from
artificially inflating high-decision density.

When a collapsed group contains multiple concrete records, choose the retained
record deterministically:

1. Prefer a strategic action over a discard action.
2. Prefer the record with the smallest available AlphaBeta top-action value gap.
3. Prefer the earliest `trajectory_index`.
4. Break any remaining tie by local `decision_id`.

The retained record remains a normal concrete training example. Collapsing only
controls checkpoint density; it does not merge or rewrite the prompt.

## 9. Dynamic Checkpoint Count

Let `L` be the completed game length in turns:

```text
L = game_end_turn
```

Determine the maximum checkpoint count from game length:

```text
L <= 70       K_by_length = 4
71 <= L < 85  K_by_length = 5
L >= 85       K_by_length = 6
```

Let `N` be the number of collapsed midgame decision opportunities for the
winning player. The final checkpoint count is:

```text
K = min(K_by_length, N)
```

If `N == 0`, the player has no midgame checkpoint records. If the game ends
early or a band has no eligible opportunity, emit fewer than `K` checkpoints.
Do not substitute a distant decision from another part of the game.

## 10. Checkpoint Placement

Normalize each candidate's position in its trajectory:

```text
normalized_progress = turn / game_end_turn
```

Divide normalized progress into `K` equal-width bands:

```text
band 0: [0/K, 1/K)
```

For each band:

1. Compute the band midpoint.
2. Find candidate opportunities inside the band.
3. Select the candidate closest to the midpoint.
4. Break ties using the candidate ranking from Section 8.
5. Skip the band if it contains no candidate.

This gives game-length-normalized checkpoints while allowing long games to
receive more strategy updates than short games. It also avoids pretending that
turn 30 has the same strategic meaning in a 57-turn game and a 96-turn game.

Each selected record receives:

```json
{
  "checkpoint_index": 1,
  "checkpoint_count": 5,
  "checkpoint_band": 0,
  "normalized_progress": 0.21,
  "selection_reason": "nearest_midpoint"
}
```

## 11. Strategy Propagation

The first checkpoint for a player receives the phase-2 initial-placement strategy
from the same trajectory and seat.

For checkpoint `i`:

```text
strategy_in(i) = strategy_out(i - 1)
```

The first checkpoint uses:

```text
strategy_in(1) = initial_phase2_strategy
```

The strategy generated from checkpoint `i` is not injected into checkpoint `i`.
It becomes the input only for checkpoint `i+1`.

Strategy state is isolated per:

```text
(trajectory_id, seat_index)
```

Other players' board actions remain visible in the prompt, but their strategy
outputs never replace the current player's strategy state.

If the initial phase-2 strategy is unavailable, the player's midgame chain is
marked `missing_bootstrap_strategy` and is excluded from dependent checkpoint
requests.

## 12. Prompt and Output Contract

Every checkpoint prompt must contain exactly one strategy block:

```text
[CURRENT STRATEGY]
<strategy_in>
```

The model must return tags in this order:

```xml
<think>...</think>
<strategy>...</strategy>
<action>N</action>
```

The strategy prompt must describe the strategy as a prior checkpoint plan that
may be refined or pivoted based on the current state. It must not imply that the
strategy was generated on every intervening game tick.

The resulting augmented record must preserve both the engine target and the raw
teacher output:

```json
{
  "prompt": "...",
  "completion": "<think>...</think><strategy>...</strategy><action>4</action>",
  "engine_completion": "4",
  "chosen_label": "...",
  "predicted_action": 4,
  "strategy_in": "...",
  "strategy_out": "...",
  "strategy_source_turn": 18,
  "strategy_source_decision_id": 7,
  "trajectory_id": "...",
  "seat_index": 1,
  "color": "BLUE",
  "turn": 30,
  "game_end_turn": 72,
  "checkpoint_index": 2,
  "checkpoint_count": 5,
  "phase": "PLAY_TURN",
  "validation_status": "accepted"
}
```

The raw teacher output is the SFT completion for the reasoning dataset. An
action-only derived dataset may later use `<action>N</action>` while retaining
the full response in metadata.

## 13. Batch Workflow

The workflow is a durable state machine. The long-running process is convenient,
but the SQLite run store is the source of truth and must support recovery after
process exit.

### 13.1 Preparation

1. Load the trajectory manifest and initial phase-2 strategy map into SQLite.
2. Sample the 500 winners with the fixed seed.
3. Build collapsed high-decision opportunities.
4. Select dynamic checkpoints.
5. Write checkpoint rows with `strategy_in` and no teacher result.

### 13.2 Submission

For each checkpoint epoch:

1. Partition requests into batches of approximately 100 prompts.
2. Persist each chunk and its request mapping before submission.
3. Submit all chunks for the epoch concurrently.
4. Persist every returned OpenAI batch ID immediately.
5. Poll all chunks until each reaches a terminal state.
6. Download each result file.
7. Validate and parse every response.
8. Write accepted results and rejected-result diagnostics.
9. Advance to the next checkpoint only after all current-epoch chunks are
   terminal and the strategy map for valid chains is complete.

### 13.3 Resume Behavior

The SQLite run store is the source of truth. Resuming a run must:

- Reuse submitted batch IDs instead of submitting duplicates.
- Detect completed chunks whose results are already downloaded.
- Revalidate existing parsed results.
- Submit only missing future work.
- Never advance an epoch based on an unvalidated response.

Recommended lifecycle states:

```text
prepared
submitted
polling
downloaded
validated
advanced
failed
```

The corresponding run and batch-chunk fields are:

Run row:

```text
run_id, source_version, sample_seed, model, reasoning_effort,
chunk_size, status, current_checkpoint, config_json
```

Batch-chunk rows:

```text
chunk_id, run_id, checkpoint_index, chunk_index, request_path,
request_hash, openai_batch_id, result_path, status,
accepted_count, rejected_count
```

The database may export this state as a JSON manifest for inspection, but the
export is diagnostic and must not replace the SQLite state.

The OpenAI Batch API's 24-hour completion window is expected. The runner should
poll at a configurable interval and write progress after every state transition.

## 14. Validation Gates

A response is accepted only when all gates pass:

1. The response contains a non-empty `<think>` block.
2. The response contains a non-empty `<strategy>` block.
3. The response contains a parseable `<action>` integer.
4. The action is within the prompt's move range.
5. The action equals the original AlphaBeta `engine_completion`.
6. The prompt has exactly one `[CURRENT STRATEGY]` block.
7. The strategy source belongs to the same trajectory and seat.
8. The strategy source checkpoint precedes the target checkpoint.

Track rejection reasons separately:

```text
missing_think
missing_strategy
missing_action
invalid_action_range
wrong_alpha_beta_action
duplicate_strategy_block
broken_strategy_lineage
api_error
```

If a checkpoint is rejected, mark that player's strategy chain invalid from that
checkpoint onward. Do not silently reuse an older strategy. The rejected record
may remain in diagnostics but must not feed the next epoch or the accepted SFT
corpus.

## 15. Dataset Split

The first version is a language/reasoning baseline rather than a gameplay
generalization benchmark.

Use a reproducible record-level random split:

```text
80% train
10% validation
10% test
```

Persist the split seed and assignment in every final record. A future gameplay
generalization evaluation should add a trajectory-level split as a separate
benchmark.

## 16. Expected Pilot Size and Expansion

With 500 winners and four to six checkpoints per winner, the expected accepted
midgame corpus is approximately 2,000-3,000 records before validation losses.

After the pilot:

- Measure accepted records per winner.
- Measure checkpoint coverage by game-length bucket and seat position.
- Measure tag/action validation rates.
- Expand the winner sample if the accepted corpus is too small.

The expansion should reuse the same selection and batch workflow with a new
`run_id` and sample manifest. It must not silently mix different trajectory
provenance versions.

## 17. Acceptance Criteria

The implementation is complete when:

- The sampled population contains approximately 500 winners with reproducible
  seat stratification.
- Every sampled trajectory has a stable provenance manifest.
- Every selected checkpoint has normalized progress and a documented selection
  reason.
- Same-turn/same-phase raw records do not inflate checkpoint density.
- The first checkpoint uses the exact same trajectory's phase-2 strategy.
- Later checkpoints use only the immediately preceding accepted strategy.
- Batch chunks are persisted, resumable, and never duplicated after restart.
- No next epoch starts before the previous epoch is validated.
- Rejected responses cannot contaminate dependent strategy chains.
- The final accepted records pass all validation gates.
- The resulting JSONL can be split into full-reasoning and action-only SFT
  variants without another API call.

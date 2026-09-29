"""
Collect *only* initial placements as an LLM supervision corpus.

This is a placements-only companion to ``collect_corpus``: it records every
initial settlement→road decision from *all* players (not just the winner),
ignoring the high-decision filters.  Each 4-player game yields 8 records
(4 players × 2 placements), with settlement+road bundled as one prompt — the
same prompting used for the main corpus so the data is directly comparable.

Records are JSONL with the same schema as ``collect_corpus.decision_to_record``:

  {
    "prompt": <six-section prompt via LLMObservationAgent.build_full_prompt>,
    "completion": "3",
    "chosen_index": 3,
    "chosen_label": "...",
    "phase": "BUILD_INITIAL_SETTLEMENT",
    "color": "RED",
    "num_moves": 54,
    "move_labels": ["1. ...", ...],
    "selected_action": {...},
    "next_road": {...},          # grouped road
    "grouped": true,
    "winner": "BLUE",            # game winner for metadata only (not a filter)
    "game_id": 0,
    "decision_id": 2
  }

Usage::

    # 1) As a module (recommended)
    venv/bin/python -m catan_llm.llm.collect_placements -n 100 -o initial_placements.jsonl
    venv/bin/python -m catan_llm.llm.collect_placements 500 placements.jsonl

    # 2) Via the main corpus with flag
    venv/bin/python -m catan_llm.llm.collect_corpus --placements -n 100 -o initial_placements.jsonl

    # 3) Programmatically
    from catan_llm.llm.collect_placements import run_placements_simulation
    run_placements_simulation(num_games=100, output_file="placements.jsonl")
"""

from catan_llm.llm.collect_corpus import (
    PlacementsAccumulator,
    CorpusCollectionPlayer,
    decision_to_record,
    run_placements_simulation,
)

__all__ = [
    "PlacementsAccumulator",
    "CorpusCollectionPlayer",
    "decision_to_record",
    "run_placements_simulation",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect initial-placement prompts (all players, 8 per game) to JSONL"
    )
    parser.add_argument("num_games", nargs="?", type=int, default=1000, help="Number of games to simulate")
    parser.add_argument("output_file", nargs="?", default="initial_placements.jsonl", help="Output JSONL path")
    parser.add_argument("-n", "--num-games", type=int, dest="num_games_flag", help="Number of games (flag form)")
    parser.add_argument("-o", "--output", "--out", type=str, dest="output_flag", help="Output JSONL path (flag form)")
    args = parser.parse_args()

    n = args.num_games_flag if args.num_games_flag is not None else args.num_games
    out = args.output_flag if args.output_flag is not None else args.output_file

    if isinstance(n, str) and n.endswith(".jsonl"):
        n, out = out, n  # type: ignore  # swapped positional

    if not out.endswith(".jsonl"):
        print(f"Note: output {out} does not end with .jsonl — using as-is, but JSONL is recommended.")

    run_placements_simulation(n, output_file=out)

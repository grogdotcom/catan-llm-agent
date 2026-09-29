#!/usr/bin/env python3
"""
Collect 1000 games of initial placements (8000 records), marking first vs second
placement per player.

Each game yields 8 records (4 players x 2 placements):
  - decision_id 0-3 : first round (forward draft)
  - decision_id 4-7 : second round (reverse draft)

Annotations added per record:
  - placement_round: 1 or 2
  - is_first_placement: bool (True for round 1, False for round 2)
  - placement_number_for_player: 1 or 2 (same as round but explicit)
  - draft_order: forward/reverse position

Resumes from existing file if present (checks max game_id).

Output: data/initial_placements/raw/initial_placements_1000.jsonl (and also updates data/initial_placements/raw/initial_placements.jsonl if desired)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from catan_llm.llm.collect_corpus import CorpusCollectionPlayer, PlacementsAccumulator, _write_jsonl, decision_to_record
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.enums import ActionPrompt, ActionType

OUT_FILE = "data/initial_placements/raw/initial_placements_1000.jsonl"
LEGACY_FILE = "data/initial_placements/raw/initial_placements.jsonl"
TARGET_GAMES = 1000  # default; override via --games / --out / first positional arg

def annotate_record(rec: dict) -> dict:
    did = rec["decision_id"]
    # decision 0-3 = first draft round, 4-7 = second (reverse)
    placement_round = 1 if did < 4 else 2
    rec["placement_round"] = placement_round
    rec["is_first_placement"] = placement_round == 1
    rec["placement_number_for_player"] = placement_round  # per player 1 or 2
    # draft_position_global is same as decision_id
    rec["draft_position"] = did
    # is_first vs second per player: can also derive via color count, but round suffices
    return rec


def sample_two_players_per_game(records, seed: int = 12345, players_per_game: int = 2):
    """Randomly sample 2 players' pairs per game (keeps both settlements per sampled player).

    For 2000 games → 4000 first + 4000 second = 8000 records.
    Deterministic per game_id: Random(seed + game_id).sample(colors, k).
    """
    import random
    from collections import defaultdict

    by_game = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    sampled = []
    for gid in sorted(by_game):
        recs = by_game[gid]
        # distinct colors in this game (should be 4)
        colors = sorted(set(r["color"] for r in recs))
        rng = random.Random(seed + int(gid))
        # if game has fewer than k colors (shouldn't), take all
        k = min(players_per_game, len(colors))
        picked = set(rng.sample(colors, k))
        for r in recs:
            if r["color"] in picked:
                rr = dict(r)
                rr["sampled"] = True
                rr["sampled_players_for_game"] = sorted(picked)
                rr["sample_seed"] = seed
                sampled.append(rr)
    sampled.sort(key=lambda x: (x["game_id"], x["decision_id"]))
    return sampled

def load_existing():
    existing = []
    max_gid = -1
    for path in [OUT_FILE, LEGACY_FILE]:
        p = Path(path)
        if p.exists():
            with p.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    # ensure annotation exists
                    annotate_record(r)
                    existing.append(r)
                    if r.get("game_id", -1) > max_gid:
                        max_gid = r["game_id"]
            print(f"Loaded {len(existing)} from {path} (max_gid {max_gid})")
            if existing:
                break
    # dedup by (game_id, decision_id)
    seen = {}
    for r in existing:
        seen[(r["game_id"], r["decision_id"])] = r
    deduped = sorted(seen.values(), key=lambda x: (x["game_id"], x["decision_id"]))
    return deduped, max_gid

def run():
    existing, max_gid = load_existing()
    start_game = max_gid + 1 if max_gid >= 0 else 0
    needed = TARGET_GAMES - (max_gid + 1 if max_gid >= 0 else 0)
    print(f"Target {TARGET_GAMES} games; start at {start_game}; need {needed} more")
    if needed <= 0:
        # just re-annotate and write
        print("Already at target, rewriting annotated file...")
        for r in existing:
            annotate_record(r)
        _write_jsonl(OUT_FILE, existing)
        print(f"Wrote {len(existing)} to {OUT_FILE}")
        return

    all_records = list(existing)  # keep existing annotated
    player_instances = [
        CorpusCollectionPlayer(Color.RED, prunning=False),
        CorpusCollectionPlayer(Color.BLUE, prunning=False),
        CorpusCollectionPlayer(Color.ORANGE, prunning=False),
        CorpusCollectionPlayer(Color.WHITE, prunning=False),
    ]
    # incremental save every 10 games
    for gi in range(start_game, TARGET_GAMES):
        for p in player_instances:
            p.reset_state()
        acc = PlacementsAccumulator(player_instances, game_id=gi)
        game = Game(player_instances, friendly_robber=False)
        game.play(accumulators=[acc])
        # annotate acc.corpus
        for rec in acc.corpus:
            annotate_record(rec)
        all_records.extend(acc.corpus)
        print(f"Game {gi+1}/{TARGET_GAMES} done: {len(acc.corpus)} placements (total {len(all_records)})")
        if (gi + 1) % 10 == 0 or gi == start_game:
            _write_jsonl(OUT_FILE, sorted(all_records, key=lambda x: (x["game_id"], x["decision_id"])))
            print(f"  checkpoint saved to {OUT_FILE}")

    # final sort + write
    all_records = sorted(all_records, key=lambda x: (x["game_id"], x["decision_id"]))
    _write_jsonl(OUT_FILE, all_records)
    print(f"Finished. {len(all_records)} placements ({TARGET_GAMES} games) -> {OUT_FILE}")
    # summary
    rounds = [r["placement_round"] for r in all_records]
    import collections
    print(collections.Counter(rounds))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect initial placements with round marking")
    parser.add_argument("--games", "-n", type=int, default=None, help="Target number of games (default 1000, use 2000 for double)")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--sample-players", type=int, default=None, help="If set, randomly sample this many players per game (e.g. 2) — keeps both settlements per sampled player. Produces 4*sample vs 8 per game.")
    parser.add_argument("--sample-seed", type=int, default=12345, help="Seed for player sampling (deterministic per game_id)")
    parser.add_argument("--sample-out", type=str, default=None, help="Optional second file to write sampled subset (e.g. data/initial_placements/raw/initial_placements_2000_sampled2.jsonl)")
    parser.add_argument("games_pos", nargs="?", default=None, help="Positional games target (shorthand)")
    args, _ = parser.parse_known_args()
    if args.games is not None:
        TARGET_GAMES = args.games
    elif args.games_pos is not None and args.games_pos.isdigit():
        TARGET_GAMES = int(args.games_pos)
    if args.out is not None:
        OUT_FILE = args.out
    # If targeting 2000, also support resuming from 1000 file transparently
    # (load_existing already checks OUT_FILE then LEGACY_FILE; for 2000 we check both 2000 and 1000)
    if TARGET_GAMES == 2000 and OUT_FILE == "data/initial_placements/raw/initial_placements_1000.jsonl":
        # User asked for 2000 but left default out — switch to 2000 filename
        OUT_FILE = "data/initial_placements/raw/initial_placements_2000.jsonl"
        # Ensure we can resume from 1000 file if 2000 not yet exists
        LEGACY_FILE = "data/initial_placements/raw/initial_placements_1000.jsonl"

    # If sampling requested, wrap run to also emit sampled file
    if args.sample_players is not None:
        # Run normal collection first
        run()
        # Then create sampled subset from the just-written OUT_FILE
        import json as _json
        _recs = []
        with open(OUT_FILE) as _f:
            for _line in _f:
                if _line.strip():
                    _recs.append(_json.loads(_line))
        _sampled = sample_two_players_per_game(_recs, seed=args.sample_seed, players_per_game=args.sample_players)
        _sample_path = args.sample_out or OUT_FILE.replace(".jsonl", f"_sampled{args.sample_players}.jsonl")
        # For 2000 with 2 players, also alias to _sampled2 expected by batch pipeline
        if args.sample_players == 2 and "2000" in OUT_FILE and _sample_path == OUT_FILE.replace(".jsonl", "_sampled2.jsonl"):
            pass
        with open(_sample_path, "w") as _out:
            for _r in _sampled:
                _out.write(_json.dumps(_r, ensure_ascii=False) + "\n")
        print(f"Sampled {len(_recs)} -> {len(_sampled)} (players_per_game={args.sample_players}, seed={args.sample_seed}) -> {_sample_path}")
        # Also print counts per round
        from collections import Counter as _Counter
        print(_Counter(r["placement_round"] for r in _sampled))
    else:
        run()

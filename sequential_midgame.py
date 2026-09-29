#!/usr/bin/env python3
"""
Next midgame checkpoints from the 2000 placement games, via OpenCode Muse Spark.

Uses the 4000 placement strategies you already have
(data/initial_placements/phase2/augmented/initial_placements_2000_with_strategy.jsonl)
as lineage for the *next* midgame checkpoint per game.

Picks ~500 games (deterministic sample) and runs their first midgame
decision after placements (≈500 requests, or K per game if --checkpoints all).

Flow:
  1. Load phase2 augmented map (game_id,color)->strategy (3999 entries for 2000 games)
  2. Load trajectory_manifest + high_decision_moves for those game_ids
  3. Sample 500 trajectory_ids (seeded) — stratified by game_end_turn K or simple random
  4. For each sampled trajectory: collapse (traj,seat,turn,phase), divide progress
     into K bands (4/5/6 by game_end_turn via sft.selection), select checkpoints.
     Default --checkpoint-mode first picks band 0 (earliest midgame after placements);
     use --checkpoint-mode all to run all K bands (~2500 req for 500 games).
  5. Inject phase2 strategy into each selected prompt's [CURRENT STRATEGY] block
     and call muse-spark-1.2-contributor via TeacherGateway (responses API, 8192 tokens,
     x-opencode-session header) in parallel (concurrency 8 default).

Outputs -> data/opencode_midgame/
  midgame_results.jsonl  (one per checkpoint, response_text + parsed <think>/<strategy>/<action>)
  run_meta.json
  selected_checkpoints.jsonl (the SFT selection metadata before injection)

Usage:
  venv/bin/python sequential_midgame.py --dry-run --sample-games 10
  venv/bin/python sequential_midgame.py --sample-games 500 --concurrency 8 --checkpoint-mode first
  venv/bin/python sequential_midgame.py --sample-games 500 --checkpoint-mode all --concurrency 16 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "src")

from catan_llm.llm.teacher import TeacherGateway, spec_for
from catan_llm.llm.teacher.models import DEFAULT_SYSTEM_PROMPT, normalize_base_url, _extract_response_text
from catan_llm.llm.teacher.parsing import parse_teacher_response
from catan_llm.llm.prompt.strategy import ensure_strategy_block
from catan_llm.llm.sft.selection import (
    collapse_opportunities,
    filter_midgame_candidates,
    select_checkpoints_for_trajectory,
)
from catan_llm.llm.sft.provenance import seat_index_for_color

DEFAULT_AUGMENTED = "data/initial_placements/phase2/augmented/initial_placements_2000_with_strategy.jsonl"
DEFAULT_HIGH = "data/sft/high_decision_moves.jsonl"
DEFAULT_MANIFEST = "data/sft/trajectory_manifest.jsonl"
DEFAULT_OUTPUT_DIR = "data/opencode_midgame"
DEFAULT_MODEL = "muse-spark-1.2-contributor"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SAMPLE_GAMES = 500
DEFAULT_CONCURRENCY = 8
MAX_RETRIES = 3


def load_augmented_strategy_map(path: str) -> Dict[Tuple[int, str], str]:
    m: Dict[Tuple[int, str], str] = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        j = json.loads(line)
        # Phase2 records carry injected_strategy; Phase1 has none
        strat = j.get("injected_strategy")
        if strat:
            m[(j["game_id"], j["color"])] = strat
    return m


def load_manifest(path: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for line in Path(path).open():
        if not line.strip():
            continue
        j = json.loads(line)
        out[j["game_id"]] = j
    return out


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    seen: Dict[str, Dict[str, Any]] = {}
    for line in path.open():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if cid:
                seen[cid] = obj
        except Exception:
            continue
    return seen


def custom_id_for_mid(rec: Dict[str, Any], sel: Dict[str, Any]) -> str:
    # stable: traj-game-decision-color-turn-phase-ckpt
    gid = rec.get("game_id", "?")
    did = rec.get("decision_id", "?")
    color = rec.get("color", "?")
    turn = rec.get("turn", "?")
    phase = rec.get("phase", "?")
    ckpt = sel.get("checkpoint_index", "?")
    traj = rec.get("trajectory_id", "?")[:12]
    return f"game-{gid}-dec-{did}--{color}--t{turn}-{phase}--ckpt{ckpt}--{traj}"


def sample_trajectory_ids(all_ids: List[str], n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    if n >= len(all_ids):
        return sorted(all_ids)
    return sorted(rng.sample(all_ids, n))


def main():
    p = argparse.ArgumentParser(description="Next midgame checkpoints via Muse Spark (parallel)")
    p.add_argument("--augmented", default=DEFAULT_AUGMENTED, help="Phase2 augmented JSONL with injected_strategy")
    p.add_argument("--corpus", default=DEFAULT_HIGH, help="High-decision moves JSONL")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Trajectory manifest JSONL")
    p.add_argument("--sample-games", type=int, default=DEFAULT_SAMPLE_GAMES, help="Number of distinct game_ids/trajectories to sample (~500)")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--checkpoint-mode", choices=["first", "all"], default="first", help="first = earliest band per trajectory (1 per game); all = all K bands")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--delay", type=float, default=0.0, help="Delay between calls (only concurrency=1)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Debug: limit selected checkpoints")
    p.add_argument("--base-url", default=None)
    args = p.parse_args()

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    spec = spec_for(args.model, base_url)
    print(f"Midgame via {args.model}  base_url={normalize_base_url(base_url)}  spec={spec.api}/{spec.default_max_tokens}")
    if not args.dry_run and not api_key:
        print("WARN: OPENAI_API_KEY not set — will fail (use --dry-run)")

    # 1. Load strategy map
    strat_map = load_augmented_strategy_map(args.augmented)
    print(f"Loaded placement strategies: {len(strat_map)} (game,color) from {args.augmented}")
    if not strat_map:
        print("ERROR: no strategies — check augmented path")
        sys.exit(1)
    # quick: how many distinct games covered?
    strat_games = len({gid for gid, _ in strat_map})
    print(f"  covering {strat_games} distinct games")

    # 2. Load manifest + corpus
    manifest = load_manifest(args.manifest)
    print(f"Loaded manifest: {len(manifest)} trajectories from {args.manifest}")

    # Load high decisions filtered to those games that have a strategy for at least one color
    # Build game_id set that has at least one strategy (should be ~2000)
    strat_game_ids = {gid for gid, _ in strat_map}
    all_recs: List[Dict[str, Any]] = []
    for line in Path(args.corpus).open():
        if not line.strip():
            continue
        j = json.loads(line)
        gid = j.get("game_id")
        if gid not in strat_game_ids:
            continue
        # Exclude initial placements — we want midgame candidates only
        if j.get("phase") == "BUILD_INITIAL_SETTLEMENT":
            continue
        # Annotate seat_index for collapsing (needed by selection)
        # Use manifest seat_order or seat_index_for_color fallback
        traj = j.get("trajectory_id")
        color = j.get("color")
        # Try to derive seat_index via manifest
        seat_idx = j.get("seat_index")
        if seat_idx is None:
            m = manifest.get(gid)
            if m and "seat_order" in m:
                try:
                    seat_idx = m["seat_order"].index(color)
                except Exception:
                    seat_idx = seat_index_for_color(color) if color else 0
            else:
                seat_idx = seat_index_for_color(color) if color else 0
            j["seat_index"] = seat_idx
        # Ensure trajectory_index for determinism (use decision_id as tie-break)
        if "trajectory_index" not in j:
            j["trajectory_index"] = j.get("decision_id", 0)
        all_recs.append(j)
    print(f"Loaded midgame candidates: {len(all_recs)} records (phase != BUILD_INITIAL_SETTLEMENT, filtered to {len(strat_game_ids)} games)")

    # 3. Sample trajectory_ids — only those with placement lineage when possible
    all_tids = sorted({r["trajectory_id"] for r in all_recs if r.get("trajectory_id")})
    print(f"  distinct trajectory_ids in candidates: {len(all_tids)}")
    # Filter to trajectories where winner's color has a phase2 strategy (so we can chain lineage)
    # Build game_id -> winner color from manifest / high_decisions
    winner_for_tid: Dict[str, str] = {}
    for r in all_recs:
        tid = r.get("trajectory_id")
        if tid not in winner_for_tid:
            winner_for_tid[tid] = r.get("color")  # high_decisions are winner-only, one color per tid
    stratified_pool = [tid for tid in all_tids if (manifest.get(next((x.get("game_id") for x in all_recs if x.get("trajectory_id")==tid), -1), {}).get("game_id", -1), winner_for_tid.get(tid)) in strat_map or any((r.get("game_id"), r.get("color")) in strat_map for r in all_recs if r.get("trajectory_id")==tid)]
    # Simpler: require winner lineage directly
    winner_lineage_tids = [tid for tid in all_tids if winner_for_tid.get(tid) and any((r.get("game_id"), winner_for_tid[tid]) in strat_map for r in all_recs if r.get("trajectory_id")==tid)]
    if len(winner_lineage_tids) >= args.sample_games:
        print(f"  winner-lineage pool: {len(winner_lineage_tids)}/2000 have placement strategy for winner color — sampling from this pool for clean lineage")
        sampled_tids = sample_trajectory_ids(winner_lineage_tids, args.sample_games, seed=args.sample_seed)
    else:
        print(f"  winner-lineage pool only {len(winner_lineage_tids)}, need {args.sample_games} — sampling from all and will use None where missing")
        sampled_tids = sample_trajectory_ids(all_tids, args.sample_games, seed=args.sample_seed)
    print(f"Sampled {len(sampled_tids)} trajectory_ids (seed {args.sample_seed}) e.g. {sampled_tids[:3]}")

    # Filter to sampled trajectories
    sampled_recs = [r for r in all_recs if r.get("trajectory_id") in set(sampled_tids)]
    print(f"  sampled records: {len(sampled_recs)} (≈{len(sampled_recs)/len(sampled_tids):.1f}/game)")

    # Group by trajectory+seat for checkpoint selection
    # Actually selection is per (trajectory_id, seat_index) — but high_decisions are winner-only? Let's check.
    # In high_decision_moves, each game appears to be winner's decisions only (one seat per game).
    # So per trajectory_id we have one seat_index. Group accordingly.
    by_traj_seat: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in sampled_recs:
        by_traj_seat[(r["trajectory_id"], r["seat_index"])].append(r)

    selected: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []  # (opportunity, selection_meta)
    selected_meta_list: List[Dict[str, Any]] = []
    for (tid, seat), recs in sorted(by_traj_seat.items()):
        # pick one game_id to get manifest game_end_turn
        gid = recs[0].get("game_id")
        m = manifest.get(gid, {})
        game_end_turn = m.get("game_end_turn", 84)
        collapsed = collapse_opportunities(recs)
        # normalized_progress already set by selection; but ensure
        sels = select_checkpoints_for_trajectory(
            collapsed=collapsed,
            game_end_turn=game_end_turn,
            trajectory_id=tid,
            seat_index=seat,
        )
        if not sels:
            continue
        if args.checkpoint_mode == "first":
            # earliest band = smallest checkpoint_band
            sels = [min(sels, key=lambda s: s["checkpoint_band"])]
        for sel in sels:
            opp = sel["opportunity"]
            selected.append((opp, sel))
            # also keep meta for output file
            meta = dict(sel)
            # embed opportunity custom fields for debugging
            meta["opportunity"] = {k: opp.get(k) for k in ["game_id", "decision_id", "color", "phase", "turn", "trajectory_id", "seat_index", "normalized_progress", "completion", "chosen_label"]}
            meta["game_end_turn"] = game_end_turn
            selected_meta_list.append(meta)

    print(f"Selected checkpoints: {len(selected)} (mode {args.checkpoint_mode}, K per traj 4-6 → avg {len(selected)/len(sampled_tids):.1f}/game)")

    if args.limit:
        selected = selected[: args.limit]
        selected_meta_list = selected_meta_list[: args.limit]
        print(f"Limited to {args.limit} for debug")

    if not selected:
        print("No checkpoints selected — check filters")
        sys.exit(1)

    # Show phase breakdown
    from collections import Counter
    phase_ctr = Counter(opp.get("phase") for opp, _ in selected)
    print(f"  phase breakdown: {dict(phase_ctr)}")
    # Show lineage coverage
    missing_strat = sum(1 for opp, _ in selected if (opp.get("game_id"), opp.get("color")) not in strat_map)
    print(f"  lineage: {len(selected)-missing_strat}/{len(selected)} have phase2 strategy; {missing_strat} missing (will use None)")

    # Prepare output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "midgame_results.jsonl"
    meta_path = out_dir / "selected_checkpoints.jsonl"
    run_meta_path = out_dir / "run_meta.json"

    # Write selection meta (always, even dry-run)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in selected_meta_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Wrote selection meta -> {meta_path} ({len(selected_meta_list)})")

    if args.dry_run:
        print("\n[dry-run] would inject strategies and call model for:")
        for i, (opp, sel) in enumerate(selected[:5]):
            gid, color = opp.get("game_id"), opp.get("color")
            strat = strat_map.get((gid, color), "None")[:80]
            print(f"  {i+1}. game {gid} {color} turn {opp.get('turn')} phase {opp.get('phase')} ckpt {sel.get('checkpoint_index')}/{sel.get('checkpoint_count')} strat={strat!r}...")
        print(f"\n[dry-run] total {len(selected)} — no API calls. Re-run without --dry-run to execute.")
        # still write run_meta
        with run_meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": args.model,
                    "base_url": normalize_base_url(base_url),
                    "spec_api": spec.api,
                    "augmented": args.augmented,
                    "corpus": args.corpus,
                    "manifest": args.manifest,
                    "sample_games": args.sample_games,
                    "sample_seed": args.sample_seed,
                    "checkpoint_mode": args.checkpoint_mode,
                    "concurrency": args.concurrency,
                    "dry_run": True,
                    "selected": len(selected),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )
        return

    # Build gateway
    gateway = TeacherGateway(
        api_key=api_key,
        model=args.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=args.max_tokens,
        base_url=base_url,
    )

    existing = _load_existing_results(out_path) if args.resume else {}
    if existing:
        print(f"Resume: {out_path} has {len(existing)} — skipping those custom_ids")

    # Build work items: (custom_id, prompt_text, opp, sel)
    work: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
    for opp, sel in selected:
        cid = custom_id_for_mid(opp, sel)
        if cid in existing:
            continue
        key = (opp.get("game_id"), opp.get("color"))
        strat = strat_map.get(key)
        # Inject strategy before [RECENT TURNS]; ensure block exists
        prompt = ensure_strategy_block(opp.get("prompt", ""), strat if strat else "None")
        work.append((cid, prompt, opp, sel))

    print(f"To execute: {len(work)} (skipped {len(selected)-len(work)} resume), concurrency={args.concurrency}")

    if not work:
        print("Nothing to do — all checkpoints already have results")
        return

    file_lock = threading.Lock()
    map_lock = threading.Lock()  # not needed but keeps parity
    f_append = out_path.open("a", encoding="utf-8")
    results: List[Dict[str, Any]] = []

    def _call_one(item: Tuple[str, str, Dict[str, Any], Dict[str, Any]]) -> Dict[str, Any]:
        cid, prompt, opp, sel = item
        call_rec = dict(opp)
        call_rec["prompt"] = prompt
        last_err = None
        response_text = ""
        usage = None
        latency_ms = None
        raw = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                raw = gateway.chat_completion(
                    call_rec,
                    model=gateway.model,
                    system_prompt=gateway.system_prompt,
                    temperature=gateway.temperature,
                    max_tokens=gateway.max_tokens,
                )
                latency_ms = int((time.time() - t0) * 1000)
                response_text = _extract_response_text(raw) or ""
                try:
                    usage = getattr(raw, "usage", None) or (raw.get("usage") if isinstance(raw, dict) else None)
                    if usage is not None and hasattr(usage, "model_dump"):
                        usage = usage.model_dump()
                    elif usage is not None and hasattr(usage, "dict"):
                        usage = usage.dict()
                except Exception:
                    usage = str(usage) if usage is not None else None
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                is_retriable = "429" in str(e) or "rate" in str(e).lower() or "timeout" in str(e).lower() or "500" in str(e) or "502" in str(e) or "503" in str(e)
                if attempt < MAX_RETRIES and is_retriable:
                    backoff = (2 ** (attempt - 1)) + random.random()
                    print(f"    retry {attempt}/{MAX_RETRIES} after {backoff:.1f}s — {last_err} ({cid})")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"    failed {cid}: {last_err}")
                    break

        parsed = parse_teacher_response(response_text or "")
        num_moves = opp.get("num_moves")
        predicted = parsed.action
        valid = predicted is not None and (num_moves is None or 1 <= predicted <= num_moves)
        try:
            expected_int = int(str(opp.get("completion", "")).strip())
            correct = predicted == expected_int
        except Exception:
            correct = None

        # Extract lineage strategy that was injected (for provenance)
        strat_in = strat_map.get((opp.get("game_id"), opp.get("color")))

        obj = {
            "custom_id": cid,
            "trajectory_id": opp.get("trajectory_id"),
            "game_id": opp.get("game_id"),
            "game_end_turn": sel.get("game_end_turn"),
            "decision_id": opp.get("decision_id"),
            "turn": opp.get("turn"),
            "phase": opp.get("phase"),
            "color": opp.get("color"),
            "seat_index": opp.get("seat_index"),
            "checkpoint_index": sel.get("checkpoint_index"),
            "checkpoint_count": sel.get("checkpoint_count"),
            "checkpoint_band": sel.get("checkpoint_band"),
            "normalized_progress": sel.get("normalized_progress"),
            "expected_completion": str(opp.get("completion", "")),
            "expected_label": opp.get("chosen_label"),
            "num_moves": num_moves,
            "strategy_in": strat_in,
            "strategy_in_preview": (strat_in or "")[:120],
            "response_text": response_text,
            "think": parsed.think_text,
            "strategy_out": parsed.strategy,
            "action": parsed.action,
            "valid": valid,
            "correct": correct,
            "latency_ms": latency_ms,
            "model": gateway.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "usage": usage,
            "error": last_err if not response_text else None,
            "prompt_chars": len(prompt),
        }
        with file_lock:
            f_append.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f_append.flush()
        status = "✓" if correct else ("?" if correct is None else "✗")
        strat_out_prev = (parsed.strategy or "")[:50].replace("\n", " ")
        print(f"  {status} {cid} action={parsed.action} exp={opp.get('completion')} turn={opp.get('turn')} strat_out={strat_out_prev!r} {latency_ms}ms")
        return obj

    t0 = time.time()
    if args.concurrency <= 1:
        for item in work:
            _call_one(item)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(_call_one, it): it[0] for it in work}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"unhandled {futs[fut]}: {e}")

    f_append.close()
    elapsed = time.time() - t0
    # Count results
    all_results = _load_existing_results(out_path)
    correct = sum(1 for v in all_results.values() if v.get("correct"))
    valid = sum(1 for v in all_results.values() if v.get("valid"))
    print(f"\nDone {len(all_results)}/{len(selected)} in {elapsed/60:.1f}m  valid {valid}/{len(all_results)} correct {correct}/{len(all_results)}")

    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "base_url": normalize_base_url(base_url),
                "spec_api": spec.api,
                "augmented": args.augmented,
                "corpus": args.corpus,
                "manifest": args.manifest,
                "sample_games": args.sample_games,
                "sample_seed": args.sample_seed,
                "checkpoint_mode": args.checkpoint_mode,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "elapsed_seconds": round(elapsed, 1),
                "selected": len(selected),
                "results": len(all_results),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    print(f"Meta -> {run_meta_path}  results -> {out_path}")


if __name__ == "__main__":
    main()

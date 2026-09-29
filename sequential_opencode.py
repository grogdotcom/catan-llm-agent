#!/usr/bin/env python3
"""
Sequential two-phase placement pipeline via OpenCode API.

Unlike `batch_two_phase.py` which uses OpenAI Batch (long queue), this
runs 100 games *sequentially* against the OpenCode gateway with model
`muse-spark-1.2-contributor` (no batch API).

Flow for 100 games (mirrors the two-phase strategy-carryover):

  1. Load corpus, slice first N distinct game_ids (default 100),
     optionally sample 2 players/game (deterministic seed).
  2. Phase 1: for each first-settlement (placement_round==1) call
     the model sequentially, parse <strategy>, persist line-by-line
     so the run is resumable.
  3. Phase 2: inject each player's phase-1 <strategy> into that
     player's second-settlement prompt ([CURRENT STRATEGY] before
     [RECENT TURNS]) and call sequentially again.
  4. Write an augmented corpus (800-ish records for 100 games sampled-2)
     compatible with the batch pipeline's augmented file.

Usage:
  # Dry run — no API call, just slice+preview prompts:
  venv/bin/python sequential_opencode.py --dry-run --games 5

  # Real run — 100 games, 2 sampled players/game, sequential via OpenCode:
  export OPENAI_API_KEY=...            # OpenCode key
  export OPENAI_BASE_URL=https://opencode.ai/zen/go/v1   # or your gateway
  venv/bin/python sequential_opencode.py --games 100 --sample-players 2

  # Resume after interrupt (skips already-written custom_ids):
  venv/bin/python sequential_opencode.py --games 100 --resume

  # Custom model / tokens / delay:
  venv/bin/python sequential_opencode.py --model muse-spark-1.2-contributor --max-tokens 4096 --delay 0.3

Outputs (under data/opencode_sequential/ by default):
  phase1_results.jsonl   — one JSON per phase-1 call (response_text + parsed)
  phase2_results.jsonl   — same for phase 2
  augmented.jsonl        — combined records with injected_strategy/original_prompt
  run_meta.json          — args + timing + counts + model spec
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
from catan_llm.llm.teacher.models import DEFAULT_SYSTEM_PROMPT, normalize_base_url
from catan_llm.llm.teacher.parsing import parse_teacher_response
from catan_llm.llm.prompt.strategy import ensure_strategy_block

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CORPUS = "data/initial_placements/raw/initial_placements_2000.jsonl"
DEFAULT_MODEL = "muse-spark-1.2-contributor"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_GAMES = 100
DEFAULT_SAMPLE_PLAYERS = 2
DEFAULT_SAMPLE_SEED = 12345
DEFAULT_OUTPUT_DIR = "data/opencode_sequential"
DEFAULT_DELAY = 0.2  # seconds between calls to be nice to the gateway
DEFAULT_CONCURRENCY = 1
MAX_RETRIES = 3


def _game_sort_key(r: Dict[str, Any]):
    return (r.get("game_id", 0), r.get("decision_id", 0))


# ---------------------------------------------------------------------------
# Corpus helpers (mirrors batch_two_phase helpers)
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"corpus not found: {p} (tried {path})")
    recs = []
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            recs.append(json.loads(line))
    for r in recs:
        did = r.get("decision_id", 0)
        if "placement_round" not in r:
            r["placement_round"] = 1 if did < 4 else 2
        if "is_first_placement" not in r:
            r["is_first_placement"] = r["placement_round"] == 1
        if "placement_number_for_player" not in r:
            r["placement_number_for_player"] = r["placement_round"]
    return recs


def slice_first_n_games(recs: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n is None or n <= 0:
        return recs
    game_ids = sorted({r["game_id"] for r in recs})
    keep = set(game_ids[:n])
    out = [r for r in recs if r["game_id"] in keep]
    out.sort(key=_game_sort_key)
    return out


def sample_two_players_per_game(
    records: List[Dict[str, Any]], seed: int = 12345, players_per_game: int = 2
) -> List[Dict[str, Any]]:
    by_game: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    out: List[Dict[str, Any]] = []
    for gid in sorted(by_game):
        recs = by_game[gid]
        colors = sorted({r["color"] for r in recs})
        rng = random.Random(seed + int(gid))
        k = min(players_per_game, len(colors))
        picked = set(rng.sample(colors, k))
        for r in recs:
            if r["color"] in picked:
                rr = dict(r)
                rr["sampled"] = True
                rr["sampled_players_for_game"] = sorted(picked)
                rr["sample_seed"] = seed
                out.append(rr)
    out.sort(key=_game_sort_key)
    return out


def split_rounds(recs: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    first = [r for r in recs if r.get("placement_round") == 1]
    second = [r for r in recs if r.get("placement_round") == 2]
    first.sort(key=_game_sort_key)
    second.sort(key=_game_sort_key)
    return first, second


def custom_id_for(rec: Dict[str, Any], idx: int, phase: str) -> str:
    gid = rec.get("game_id", "?")
    did = rec.get("decision_id", idx)
    color = rec.get("color", "")
    round_ = rec.get("placement_round", "?")
    base = f"game-{gid}-decision-{did}--{color}--r{round_}--{phase}"
    # dedup not needed for this slice, but guard
    return base


def _extract_text(raw: Any) -> str:
    # Reuse TeacherGateway's private extractor via model helper, or inline
    from catan_llm.llm.teacher.models import _extract_response_text

    return _extract_response_text(raw) or ""


def _load_existing_results(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    seen: Dict[str, Dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
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


# ---------------------------------------------------------------------------
# Sequential runner
# ---------------------------------------------------------------------------

def _build_injected_prompt(rec: Dict[str, Any], phase_label: str, strat_map: Dict[Tuple[int, str], str]) -> str:
    prompt = rec["prompt"]
    if phase_label == "phase1":
        return ensure_strategy_block(prompt, "None")
    key = (rec["game_id"], rec["color"])
    strat = strat_map.get(key)
    if strat:
        return ensure_strategy_block(prompt, strat)
    if "[CURRENT STRATEGY]" not in prompt:
        return ensure_strategy_block(prompt, "None")
    return prompt


def _call_with_retries(
    call_rec: Dict[str, Any],
    gateway: TeacherGateway,
    cid: str,
) -> Tuple[str, Optional[int], Any, Optional[str]]:
    last_err: Optional[str] = None
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
            response_text = _extract_text(raw)
            try:
                usage = getattr(raw, "usage", None) or (raw.get("usage") if isinstance(raw, dict) else None)
                if usage is not None and hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                elif usage is not None and hasattr(usage, "dict"):
                    usage = usage.dict()
            except Exception:
                usage = str(usage) if usage is not None else None
            return response_text, latency_ms, usage, last_err
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
                return "", None, None, last_err
    return response_text, latency_ms, usage, last_err


def run_phase(
    records: List[Dict[str, Any]],
    phase_label: str,
    gateway: TeacherGateway,
    out_path: Path,
    strat_map: Optional[Dict[Tuple[int, str], str]] = None,
    resume: bool = True,
    delay: float = DEFAULT_DELAY,
    dry_run: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Tuple[Dict[Tuple[int, str], str], List[Dict[str, Any]]]:
    """Run one phase, sequentially (concurrency==1) or in parallel."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing_results(out_path) if resume else {}
    if existing:
        print(f"  resume: {out_path} already has {len(existing)} records — skipping those custom_ids")

    phase_strat_map: Dict[Tuple[int, str], str] = dict(strat_map or {})
    results: List[Dict[str, Any]] = []
    skipped = 0

    # Pre-populate results with existing
    for idx, rec in enumerate(records):
        cid = custom_id_for(rec, idx, phase_label)
        if cid in existing:
            obj = existing[cid]
            strat = obj.get("strategy")
            if strat:
                phase_strat_map[(rec["game_id"], rec["color"])] = strat
            results.append(obj)
            skipped += 1

    if skipped:
        print(f"  phase {phase_label}: skipped {skipped} already-done, {len(records)-skipped} to execute (concurrency={concurrency})")
    else:
        print(f"  phase {phase_label}: {len(records)} to execute (concurrency={concurrency})")

    # Build work list
    work: List[Tuple[int, Dict[str, Any], str, str]] = []
    for idx, rec in enumerate(records):
        cid = custom_id_for(rec, idx, phase_label)
        if cid in existing:
            continue
        prompt = _build_injected_prompt(rec, phase_label, phase_strat_map)
        call_rec = dict(rec)
        call_rec["prompt"] = prompt
        work.append((idx, rec, cid, prompt))

    # Dry-run path (respect concurrency for preview but no API)
    if dry_run:
        for idx, rec, cid, prompt in work:
            preview = prompt[:220].replace("\n", " ")
            print(f"  [dry] {cid} prompt_preview={preview[:120]!r}...")
            fake_text = f"<think>dry-run think for {cid}</think><strategy>dry-run strategy favoring ore/wheat center</strategy><action>{rec.get('completion')}</action>"
            parsed = parse_teacher_response(fake_text)
            obj = {
                "custom_id": cid,
                "game_id": rec.get("game_id"),
                "decision_id": rec.get("decision_id"),
                "color": rec.get("color"),
                "placement_round": rec.get("placement_round"),
                "phase": phase_label,
                "expected_completion": str(rec.get("completion", "")),
                "expected_label": rec.get("chosen_label"),
                "response_text": fake_text,
                "think": parsed.think_text,
                "strategy": parsed.strategy,
                "action": parsed.action,
                "valid": True,
                "correct": parsed.action == int(str(rec.get("completion", "0")).strip()) if str(rec.get("completion", "")).isdigit() else None,
                "latency_ms": 0,
                "model": gateway.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dry_run": True,
            }
            if parsed.strategy:
                phase_strat_map[(rec["game_id"], rec["color"])] = parsed.strategy
            results.append(obj)
        # keep results in original order for reporting
        results.sort(key=lambda x: (x.get("game_id", 0), x.get("decision_id", 0)))
        return phase_strat_map, results

    if not work:
        return phase_strat_map, results

    # Thread-safe file append
    file_lock = threading.Lock()
    map_lock = threading.Lock()
    f_append = out_path.open("a", encoding="utf-8")
    completed = 0
    total_work = len(work)

    def _execute_one(item: Tuple[int, Dict[str, Any], str, str]) -> Dict[str, Any]:
        idx, rec, cid, prompt = item
        call_rec = dict(rec)
        call_rec["prompt"] = prompt
        response_text, latency_ms, usage, last_err = _call_with_retries(call_rec, gateway, cid)
        parsed = parse_teacher_response(response_text or "")
        num_moves = rec.get("num_moves")
        predicted = parsed.action
        valid = predicted is not None and (num_moves is None or 1 <= predicted <= num_moves)
        try:
            expected_int = int(str(rec.get("completion", "")).strip())
            correct = predicted == expected_int
        except Exception:
            correct = None
        obj = {
            "custom_id": cid,
            "game_id": rec.get("game_id"),
            "decision_id": rec.get("decision_id"),
            "color": rec.get("color"),
            "placement_round": rec.get("placement_round"),
            "phase": phase_label,
            "expected_completion": str(rec.get("completion", "")),
            "expected_label": rec.get("chosen_label"),
            "num_moves": num_moves,
            "response_text": response_text,
            "think": parsed.think_text,
            "strategy": parsed.strategy,
            "action": parsed.action,
            "valid": valid,
            "correct": correct,
            "latency_ms": latency_ms,
            "model": gateway.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "usage": usage,
            "error": last_err if not response_text else None,
            "prompt_chars": len(prompt),
            "response_wrapper": {
                "custom_id": cid,
                "response": {"status_code": 200 if response_text else 500, "body": {"output_text": response_text}},
            },
        }
        if parsed.strategy:
            with map_lock:
                phase_strat_map[(rec["game_id"], rec["color"])] = parsed.strategy
        elif response_text:
            print(f"    warn: no <strategy> parsed for {cid} — text len {len(response_text)} preview={response_text[:180]!r}")
        # persist immediately
        with file_lock:
            f_append.write(json.dumps(obj, ensure_ascii=False) + "\n")
            f_append.flush()
        status = "✓" if correct else ("?" if correct is None else "✗")
        strat_preview = (parsed.strategy or "")[:60].replace("\n", " ")
        print(f"  [{idx+1}/{len(records)}] {status} {cid} action={parsed.action} expected={rec.get('completion')} strat={strat_preview!r} {latency_ms}ms")
        if delay and concurrency == 1:
            time.sleep(delay)
        return obj

    if concurrency <= 1:
        for item in work:
            obj = _execute_one(item)
            results.append(obj)
            completed += 1
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_execute_one, item): item for item in work}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    obj = fut.result()
                    results.append(obj)
                    completed += 1
                except Exception as e:
                    idx, rec, cid, _ = futures[fut]
                    print(f"  !! unhandled exception for {cid}: {e}")
                    # synthesize error record so resume doesn't loop
                    obj = {
                        "custom_id": cid,
                        "game_id": rec.get("game_id"),
                        "decision_id": rec.get("decision_id"),
                        "color": rec.get("color"),
                        "placement_round": rec.get("placement_round"),
                        "phase": phase_label,
                        "expected_completion": str(rec.get("completion", "")),
                        "response_text": "",
                        "strategy": None,
                        "action": None,
                        "valid": False,
                        "correct": False,
                        "error": f"unhandled: {e}",
                        "model": gateway.model,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    with file_lock:
                        f_append.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        f_append.flush()
                    results.append(obj)

    f_append.close()
    # Keep results sorted deterministically for valid/correct summaries
    results_sorted = sorted(results, key=lambda x: (x.get("game_id", 0), x.get("decision_id", 0)))
    # Return map + sorted results (existing+new). For duplicated handling, re-derive map from all
    return phase_strat_map, results_sorted


def main():
    parser = argparse.ArgumentParser(description="Sequential two-phase via OpenCode (Muse Spark)")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="Input placements JSONL")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="First N distinct game_ids (0=all)")
    parser.add_argument("--sample-players", type=int, default=DEFAULT_SAMPLE_PLAYERS, help="Sample k players/game (0 or None = no sampling)")
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id (e.g. muse-spark-1.2-contributor)")
    parser.add_argument("--base-url", default=None, help="Gateway base URL (or OPENAI_BASE_URL env)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between calls (only for concurrency=1)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Parallel requests within a phase (1=sequential, 4-8 recommended)")
    parser.add_argument("--resume", action="store_true", help="Skip custom_ids already in output files")
    parser.add_argument("--dry-run", action="store_true", help="No API call — synthesize placeholders and test wiring")
    parser.add_argument("--limit", type=int, default=None, help="Debug: limit records per phase (after sampling)")
    parser.add_argument("--phase", choices=["both", "phase1", "phase2"], default="both", help="Run only one phase (phase2 needs phase1 results)")
    args = parser.parse_args()

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    print(f"Sequential OpenCode two-phase — model={args.model} corpus={args.corpus}")
    if base_url:
        print(f"  base_url={normalize_base_url(base_url)}")
    else:
        print(f"  base_url=(default OpenAI) — set OPENAI_BASE_URL for OpenCode zen gateway")
    if not args.dry_run and not api_key:
        print("  WARN: OPENAI_API_KEY not set — will fail on first call (use --dry-run to test wiring)")
    spec = spec_for(args.model, base_url)
    print(f"  resolved spec: api={spec.api} supports_temperature={spec.supports_temperature} default_max_tokens={spec.default_max_tokens}")

    # Load + slice + sample
    recs = load_corpus(args.corpus)
    print(f"Loaded {len(recs)} total records from {args.corpus}")
    if args.games and args.games > 0:
        before = len(recs)
        recs = slice_first_n_games(recs, args.games)
        print(f"Sliced to first {args.games} games: {before} -> {len(recs)} (distinct games {len({r['game_id'] for r in recs})})")
    if args.sample_players and args.sample_players > 0:
        before = len(recs)
        recs = sample_two_players_per_game(recs, seed=args.sample_seed, players_per_game=args.sample_players)
        print(f"Sampled {args.sample_players} players/game (seed {args.sample_seed}): {before} -> {len(recs)}")
    first, second = split_rounds(recs)
    print(f"Split: {len(first)} first (round1) + {len(second)} second (round2) = {len(recs)}")
    if args.limit:
        first = first[: args.limit]
        second = second[: args.limit]
        print(f"Limited to {args.limit} per phase for debug")

    if not first and not second:
        print("No records to run — check --games / --sample-players")
        sys.exit(1)

    # Prepare gateway
    gateway = TeacherGateway(
        api_key=api_key,
        model=args.model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=base_url,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1_path = out_dir / "phase1_results.jsonl"
    p2_path = out_dir / "phase2_results.jsonl"
    aug_path = out_dir / "augmented.jsonl"
    meta_path = out_dir / "run_meta.json"

    print(f"\nOutputs -> {out_dir}/")
    print(f"  phase1: {p1_path}")
    print(f"  phase2: {p2_path}")
    print(f"  augmented: {aug_path}")

    strat_map: Dict[Tuple[int, str], str] = {}
    t_start = time.time()

    # -- Phase 1 --
    if args.phase in ("both", "phase1"):
        print(f"\n=== Phase 1: {len(first)} first settlements (concurrency={args.concurrency}) ===")
        strat_map, p1_results = run_phase(first, "phase1", gateway, p1_path, resume=args.resume, delay=args.delay, dry_run=args.dry_run, concurrency=args.concurrency)
        # If we resumed, strat_map may be partial from skipped lines — reload full p1 to fill it
        if args.resume and p1_path.exists():
            # Rebuild strat_map from file to ensure phase2 sees all
            for obj in _load_existing_results(p1_path).values():
                s = obj.get("strategy")
                if s:
                    strat_map[(obj["game_id"], obj["color"])] = s
        print(f"Phase1 done: {len(p1_results)} results, {len(strat_map)} strategies parsed")
        # quick accuracy preview
        correct = sum(1 for r in p1_results if r.get("correct"))
        valid = sum(1 for r in p1_results if r.get("valid"))
        print(f"  valid {valid}/{len(p1_results)} correct {correct}/{len(p1_results)}")
    else:
        # Load phase1 strategies from file for phase2-only run
        if p1_path.exists():
            print(f"Loading phase1 strategies from {p1_path} for phase2-only run")
            for obj in _load_existing_results(p1_path).values():
                s = obj.get("strategy")
                if s:
                    strat_map[(obj["game_id"], obj["color"])] = s
            print(f"  loaded {len(strat_map)} strategies")
        else:
            print(f"WARN: --phase phase2 but {p1_path} not found — phase2 will use None strategies")

    # -- Phase 2 --
    if args.phase in ("both", "phase2"):
        print(f"\n=== Phase 2: {len(second)} second settlements (with injected strategy, concurrency={args.concurrency}) ===")
        # Note: run_phase for phase2 expects strat_map already populated
        strat_map, p2_results = run_phase(second, "phase2", gateway, p2_path, strat_map=strat_map, resume=args.resume, delay=args.delay, dry_run=args.dry_run, concurrency=args.concurrency)
        correct2 = sum(1 for r in p2_results if r.get("correct"))
        valid2 = sum(1 for r in p2_results if r.get("valid"))
        print(f"Phase2 done: {len(p2_results)} results, {len(strat_map)} total strategies, valid {valid2}/{len(p2_results)} correct {correct2}/{len(p2_results)}")

        if not args.dry_run:
            # Build augmented corpus (mirrors batch_two_phase augmented file)
            print(f"\nBuilding augmented corpus -> {aug_path}")
            augmented: List[Dict[str, Any]] = []
            for rec in recs:
                if rec["placement_round"] == 1:
                    augmented.append(rec)
                else:
                    key = (rec["game_id"], rec["color"])
                    strat = strat_map.get(key)
                    new_rec = dict(rec)
                    if strat:
                        new_rec["prompt"] = ensure_strategy_block(rec["prompt"], strat)
                        new_rec["injected_strategy"] = strat
                        new_rec["injected_from"] = f"game-{key[0]}-{key[1]}-first"
                        new_rec["original_prompt"] = rec["prompt"]
                    else:
                        new_rec["injected_strategy"] = None
                        new_rec["original_prompt"] = rec["prompt"]
                    augmented.append(new_rec)
            augmented.sort(key=_game_sort_key)
            with aug_path.open("w", encoding="utf-8") as f:
                for r in augmented:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Wrote {aug_path} ({len(augmented)} records)")
        else:
            print("\n[dry-run] skipping augmented.jsonl write")

    elapsed = time.time() - t_start
    meta = {
        "model": args.model,
        "base_url": normalize_base_url(base_url) if base_url else None,
        "spec_api": spec.api,
        "corpus": args.corpus,
        "games": args.games,
        "sample_players": args.sample_players,
        "sample_seed": args.sample_seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "output_dir": str(out_dir),
        "phase": args.phase,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "delay": args.delay,
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {"first": len(first), "second": len(second), "total": len(recs)},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone in {elapsed/60:.1f}m — meta {meta_path}")
    print(f"To inspect: wc -l {out_dir}/*.jsonl && head -1 {out_dir}/phase1_results.jsonl | python -m json.tool | head -40")


if __name__ == "__main__":
    main()

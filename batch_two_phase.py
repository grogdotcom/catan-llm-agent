#!/usr/bin/env python3
"""
Two-phase initial-placement batch pipeline.

Phase 1: Collect first placements (placement_round==1) across 1000 games,
         build Batch JSONL for OpenAI (Responses API via Zen), submit and poll.

Phase 2: Parse <strategy> from phase-1 outputs, inject into each player's
         second placement prompt ([CURRENT STRATEGY] section), producing
         augmented records + second-phase Batch JSONL.

Usage:
  # Prepare only (no API call) — builds JSONL files for inspection:
  venv/bin/python batch_two_phase.py --prepare-only

  # Full flow — submit phase1 to OpenAI, poll, parse, build phase2:
  venv/bin/python batch_two_phase.py --submit

  # After a batch is running, poll an existing batch id:
  venv/bin/python batch_two_phase.py --poll BATCH_ID

  # Inject from an already-downloaded results file:
  venv/bin/python batch_two_phase.py --inject results.jsonl

Outputs:
  data/initial_placements/raw/initial_placements_1000.jsonl etc. (full corpora)
  data/initial_placements/phase1/batch/initial_placements_2000_phase1.batch.jsonl    -> upload for OpenAI Batch
  data/initial_placements/phase1/results/initial_placements_2000_phase1_results.jsonl  <- downloaded results
  data/initial_placements/phase2/augmented/initial_placements_2000_with_strategy.jsonl   (8000 with augmented phase2 prompts)
  data/initial_placements/phase2/batch/initial_placements_2000_phase2.batch.jsonl    -> ready for second batch submit

All prompts carry explicit placement_round / is_first_placement so sorting is
unambiguous.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "src")
from catan_llm.llm.openai_batch import (
    DEFAULT_SYSTEM_PROMPT,
    _load_dotenv,
    _custom_id_for,
    spec_for,
    normalize_base_url,
)

_load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CORPUS_1000 = "data/initial_placements/raw/initial_placements_1000.jsonl"
CORPUS_2000 = "data/initial_placements/raw/initial_placements_2000.jsonl"
FALLBACK_CORPUS = "data/initial_placements/raw/initial_placements.jsonl"

MODEL = "gpt-5.6-luna"
# Zen gateway model id routed via OPENAI_BASE_URL=https://opencode.ai/zen/go/v1
ENDPOINT_RESPONSES = "/v1/responses"
ENDPOINT_CHAT = "/v1/chat/completions"

# Luna via responses needs larger budget (reasoning + output)
MAX_OUTPUT_TOKENS = 4096
REASONING_EFFORT = "medium"  # default medium for gpt-5.6-luna per latest plan; override via --reasoning-effort

# Phase file basing — will be derived from corpus stem (supports 1000 & 2000)
PHASE1_BATCH = "data/initial_placements/phase1/batch/initial_placements_1000_phase1.batch.jsonl"
PHASE1_RESULTS = "data/initial_placements/phase1/results/initial_placements_1000_phase1_results.jsonl"
AUGMENTED_CORPUS = "data/initial_placements/phase2/augmented/initial_placements_1000_with_strategy.jsonl"
PHASE2_BATCH = "data/initial_placements/phase2/batch/initial_placements_1000_phase2.batch.jsonl"


def phase_paths_for(corpus_path: str):
    """Derive phase file paths from corpus stem (supports 1000 & 2000).

    All phase files live under data/initial_placements/phase1|2/... regardless of
    corpus location — stem is used to distinguish 1000 vs 2000.
    """
    stem = Path(corpus_path).stem  # e.g. initial_placements_2000
    return {
        "phase1_batch": f"data/initial_placements/phase1/batch/{stem}_phase1.batch.jsonl",
        "phase1_results": f"data/initial_placements/phase1/results/{stem}_phase1_results.jsonl",
        "augmented": f"data/initial_placements/phase2/augmented/{stem}_with_strategy.jsonl",
        "phase2_batch": f"data/initial_placements/phase2/batch/{stem}_phase2.batch.jsonl",
    }


def sample_two_players_per_game(records, seed: int = 12345, players_per_game: int = 2):
    """Randomly sample k players per game, keeping both settlements per player.

    For 2000 games sampled 2 per game → 4000 first + 4000 second.
    Deterministic per game_id via Random(seed + game_id).
    """
    import random
    from collections import defaultdict

    by_game: dict = defaultdict(list)
    for r in records:
        by_game[r["game_id"]].append(r)
    out = []
    for gid in sorted(by_game):
        recs = by_game[gid]
        colors = sorted(set(r["color"] for r in recs))
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
    out.sort(key=lambda x: (x["game_id"], x["decision_id"]))
    return out

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists() and path == CORPUS_1000:
        # fallback to legacy 50-game file for dev
        p = Path(FALLBACK_CORPUS)
        print(f"{path} not ready, falling back to {p}")
    recs = []
    with p.open() as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    # Ensure annotations present
    for r in recs:
        did = r.get("decision_id", 0)
        if "placement_round" not in r:
            r["placement_round"] = 1 if did < 4 else 2
        if "is_first_placement" not in r:
            r["is_first_placement"] = r["placement_round"] == 1
        if "placement_number_for_player" not in r:
            r["placement_number_for_player"] = r["placement_round"]
    return recs

def split_rounds(recs: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    first = [r for r in recs if r.get("placement_round") == 1]
    second = [r for r in recs if r.get("placement_round") == 2]
    # sort deterministic
    first.sort(key=lambda x: (x["game_id"], x["decision_id"]))
    second.sort(key=lambda x: (x["game_id"], x["decision_id"]))
    return first, second

def parse_strategy(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<strategy>(.*?)</strategy>", text, re.S | re.I)
    if m:
        s = m.group(1).strip()
        # collapse whitespace but keep sentences
        s = re.sub(r"\s+", " ", s).strip()
        return s if s else None
    return None

def parse_think(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<think>(.*?)</think>", text, re.S | re.I)
    return m.group(1).strip() if m else None

def parse_action(text: str) -> Optional[int]:
    m = re.search(r"<action>\s*(\d+)\s*</action>", text or "", re.I)
    return int(m.group(1)) if m else None

def _extract_text_from_batch_response(resp_obj) -> str:
    """Extract assistant text from Batch output response.body"""
    # Batch result line: {custom_id, response: {status_code, body: {...}}}
    # Body varies: chat -> choices[0].message.content, responses -> output_text / output[]
    try:
        if isinstance(resp_obj, dict):
            body = resp_obj.get("body") or resp_obj.get("response", {}).get("body") or resp_obj
            # Responses shape inside body
            if body.get("output_text"):
                return body["output_text"]
            if body.get("output"):
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        c = item.get("content", [])
                        if c and c[0].get("text"):
                            return c[0].get("text")
            # Chat shape
            if body.get("choices"):
                ch = body["choices"][0]
                msg = ch.get("message", {})
                if msg.get("content"):
                    return msg["content"]
                if msg.get("reasoning_content"):
                    return msg["reasoning_content"]
            # Already flat?
            if resp_obj.get("output_text"):
                return resp_obj["output_text"]
        else:
            # object shape
            text = getattr(resp_obj, "output_text", None)
            if text:
                return text
            out = getattr(resp_obj, "output", None)
            if out:
                for item in out:
                    if getattr(item, "type", None) == "message":
                        c = getattr(item, "content", None)
                        if c and len(c) > 0:
                            t = getattr(c[0], "text", None)
                            if t:
                                return t
            # chat fallback
            try:
                msg = resp_obj.choices[0].message
                if getattr(msg, "content", None):
                    return msg.content
                rc = getattr(msg, "reasoning_content", None)
                if rc:
                    return rc
            except Exception:
                pass
    except Exception:
        pass
    return ""

def inject_strategy_into_prompt(prompt: str, strategy: str) -> str:
    """Replace ``[CURRENT STRATEGY]`` value with injected strategy string.

    Delegates to the canonical ``prompt.strategy`` helpers so the two-phase
    pipeline and the SFT pipeline share one deterministic implementation
    (no regexes over arbitrary prose).
    """
    from catan_llm.llm.prompt.strategy import ensure_strategy_block

    return ensure_strategy_block(prompt, strategy or "None")

def build_batch_line(record: Dict, custom_id: str, model: str, endpoint: str, system_prompt: Optional[str], max_tokens: int) -> Dict:
    prompt = record["prompt"]
    # Legacy corpus prompts lack [CURRENT STRATEGY] — normalize so first settlements show None
    if "[CURRENT STRATEGY]" not in prompt:
        prompt = inject_strategy_into_prompt(prompt, "None")
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if endpoint == ENDPOINT_RESPONSES:
        # Responses API shape
        body: Dict = {
            "model": model,
            "input": messages,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": REASONING_EFFORT},
        }
        # temperature not supported for luna via responses — drop
        # Keep metadata inside top-level req for local join
    else:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if REASONING_EFFORT and "reasoning" not in body:
            # chat models ignore reasoning; keep clean
            pass

    req = {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": body,
        "metadata": {
            "_expected_completion": str(record.get("completion", "")),
            "_game_id": record.get("game_id"),
            "_decision_id": record.get("decision_id"),
            "_color": record.get("color"),
            "_phase": record.get("phase"),
            "_placement_round": record.get("placement_round"),
        },
    }
    # Clean None metadata
    req["metadata"] = {k: v for k, v in req["metadata"].items() if v is not None}
    return req

def build_batch_file(records: List[Dict], out_path: str, model: str, endpoint: str, system_prompt: Optional[str], max_tokens: int) -> Dict:
    lines = []
    for idx, rec in enumerate(records):
        cid = _custom_id_for(rec, idx)
        # Ensure uniqueness for batch — include game/dec/color/round
        # _custom_id_for already encodes game-decision + color-phase
        # suffix with round to be extra safe
        cid = f"{cid}--r{rec.get('placement_round', 1)}"
        line = build_batch_line(rec, cid, model, endpoint, system_prompt, max_tokens)
        lines.append(line)
    # Dedup (same as openai_batch)
    seen = {}
    for r in lines:
        cid = r["custom_id"]
        if cid in seen:
            seen[cid] += 1
            r["custom_id"] = f"{cid}--{seen[cid]}"
        else:
            seen[cid] = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in lines:
            # remove metadata if strict? we keep metadata for join (OpenAI ignores but useful)
            # For actual upload, OpenAI will ignore unknown top-level keys? keep metadata as per spec? Actually spec says batch file lines only allow custom_id/method/url/body — metadata top-level may be rejected. Safer to keep separate map locally and strip before upload.
            # We'll write two versions: with metadata for local, but actual upload should be strict.
            # For now write strict (no metadata) + save mapping separately.
            # But to keep it compatible with Zen which is lenient, we strip metadata.
            r_strict = {k: v for k, v in r.items() if k != "metadata"}
            f.write(json.dumps(r_strict, ensure_ascii=False) + "\n")
    # also save mapping for join
    mapping_path = out_path + ".mapping.json"
    with open(mapping_path, "w") as mf:
        json.dump({r["custom_id"]: r["metadata"] for r in lines}, mf, indent=2)
    total_bytes = out.stat().st_size
    return {"output_path": str(out), "mapping_path": mapping_path, "count": len(lines), "bytes": total_bytes, "model": model, "endpoint": endpoint}

def poll_batch(client, batch_id: str, interval: int = 20, timeout: int = 3600):
    start = time.time()
    while True:
        # client is OpenAIBatchClient wrapper — use its retrieve_batch helper
        try:
            batch = client.retrieve_batch(batch_id)
        except Exception:
            # fallback to raw SDK
            batch = client._get_openai_client().batches.retrieve(batch_id)
        status = getattr(batch, "status", None) or (batch.get("status") if isinstance(batch, dict) else "?")
        print(f"[{time.strftime('%H:%M:%S')}] batch {batch_id} status={status} elapsed {(time.time()-start)/60:.1f}m")
        if status in ("completed", "failed", "cancelled", "expired"):
            return batch
        if time.time() - start > timeout:
            print("poll timeout")
            return batch
        time.sleep(interval)

def download_results(client, batch, out_path: Optional[str] = None) -> Path:
    # batch.output_file_id
    out_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
    if not out_id:
        # maybe error file?
        out_id = getattr(batch, "error_file_id", None) or (batch.get("error_file_id") if isinstance(batch, dict) else None)
        if out_id:
            print(f"Batch has no output_file_id, has error_file_id={out_id}")
        else:
            raise ValueError(f"Batch {getattr(batch,'id', batch)} has no output_file_id yet (status {getattr(batch,'status', '??')})")
    print(f"Downloading results file {out_id} ...")
    target = Path(out_path) if out_path else Path(PHASE1_RESULTS)
    # SDK: client is OpenAIBatchClient — get raw client for files
    raw = client._get_openai_client() if hasattr(client, "_get_openai_client") else client
    # SDK: raw.files.content(file_id) -> response with .text or .read()
    try:
        resp = raw.files.content(out_id)
        # New SDK: resp.text, older: resp.read().decode()
        text = None
        if hasattr(resp, "text"):
            text = resp.text
        elif hasattr(resp, "read"):
            try:
                text = resp.read().decode("utf-8")
            except Exception:
                text = resp.read()
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
        else:
            text = str(resp)
        # If response is bytes-like streaming, handle
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        # Write to file
        target.write_text(text, encoding="utf-8")
        print(f"Wrote {target} ({target.stat().st_size} bytes, {len(text.splitlines())} lines)")
        return target
    except Exception as e:
        print(f"Failed to download via files.content: {e}")
        # Try alternative: raw.files.retrieve_content?
        try:
            content = raw.files.retrieve_content(out_id)  # type: ignore
            text = content if isinstance(content, str) else content.decode("utf-8") if isinstance(content, bytes) else str(content)
            target.write_text(text, encoding="utf-8")
            return target
        except Exception as e2:
            raise RuntimeError(f"Both download methods failed: {e} / {e2}") from e

def inject_strategy_and_build_phase2(corpus_path: str, results_path: str):
    recs = load_corpus(corpus_path)
    first, second = split_rounds(recs)
    print(f"Loaded {len(recs)} total: {len(first)} first, {len(second)} second")
    paths = phase_paths_for(corpus_path)

    # Parse results file into strategy map (game_id, color) -> strategy
    strat_map: Dict[Tuple[int, str], str] = {}
    raw_map: Dict[str, str] = {}
    results_p = Path(results_path)
    if not results_p.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")
    with results_p.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            # body may be nested: response.body
            resp = obj.get("response", obj)
            body = resp.get("body", resp)
            text = _extract_text_from_batch_response(body if isinstance(body, dict) else resp)
            # also try direct extraction from batch wrapper
            if not text:
                text = _extract_text_from_batch_response(obj)
            strat = parse_strategy(text)
            # recover game+color from custom_id (format game-0-decision-2--WHITE-BUILD_INITIAL_SETTLEMENT--r1)
            # or from error mapping
            # Try to parse via metadata mapping if exists
            # But we can also parse from custom_id parts
            # custom_id encodes game and decision; we need game+color
            # Instead we rely on the original first records mapping: match by custom_id built same way
            # Build cid->(game,color) lookup
            raw_map[cid] = text
            # store strategy raw for debug, will map later via cid lookup
    # Now we need cid -> (game, color) map by rebuilding same cid generation
    cid_to_key: Dict[str, Tuple[int, str]] = {}
    for idx, rec in enumerate(first):
        cid_base = _custom_id_for(rec, idx)
        cid = f"{cid_base}--r{rec.get('placement_round',1)}"
        # dedup handling same as build: if dup, suffix --1 etc.
        # For now assume unique (round suffix makes unique)
        # But batch builder may have deduped; mimic same dedup
        # For simplicity we check raw_map keys exactly; fallback to prefix match
        key = (rec["game_id"], rec["color"])
        cid_to_key[cid] = key
        # Also store variants with dedup suffix
        # We'll match by prefix later

    # Now map strategies
    for cid, text in raw_map.items():
        strat = parse_strategy(text)
        if not strat:
            continue
        # Find key: exact cid or prefix match without suffix
        key = cid_to_key.get(cid)
        if key is None:
            # Try prefix without dedup suffix
            for k_cid, k in cid_to_key.items():
                if cid.startswith(k_cid):
                    key = k
                    break
        if key is None:
            # Try parse from custom_id string directly
            m = re.search(r"game-(\d+).*--([A-Z]+)-", cid)
            if m:
                key = (int(m.group(1)), m.group(2))
        if key is not None and strat:
            strat_map[key] = strat

    print(f"Parsed {len(strat_map)} strategies from {len(raw_map)} results")
    # Show a few
    for k, v in list(strat_map.items())[:3]:
        print(f"  {k}: {v[:120]}...")

    # Inject into second placement prompts
    augmented_all: List[Dict] = []
    missing = 0
    for rec in recs:
        if rec["placement_round"] == 1:
            augmented_all.append(rec)
        else:
            key = (rec["game_id"], rec["color"])
            strat = strat_map.get(key)
            if strat:
                new_prompt = inject_strategy_into_prompt(rec["prompt"], strat)
                new_rec = dict(rec)
                new_rec["prompt"] = new_prompt
                new_rec["injected_strategy"] = strat
                new_rec["injected_from"] = f"game-{key[0]}-{key[1]}-first"
                # Keep original for diff
                new_rec["original_prompt"] = rec["prompt"]
                augmented_all.append(new_rec)
            else:
                missing += 1
                augmented_all.append(rec)
    print(f"Second-round augmented: {len(second)}; missing strategies: {missing}")

    # Sort and save augmented corpus
    augmented_all.sort(key=lambda x: (x["game_id"], x["decision_id"]))
    aug_path = paths["augmented"]
    with open(aug_path, "w") as f:
        for r in augmented_all:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {aug_path} ({len(augmented_all)} records)")

    # Build phase2 batch file from only second placements (augmented)
    second_aug = [r for r in augmented_all if r["placement_round"] == 2]
    # If some missing strategies, they still have prompt with None — fine
    p2_path = paths["phase2_batch"]
    info2 = build_batch_file(second_aug, p2_path, MODEL, ENDPOINT_RESPONSES, DEFAULT_SYSTEM_PROMPT, MAX_OUTPUT_TOKENS)
    print(f"Phase2 batch: {info2['count']} requests -> {info2['output_path']} ({info2['bytes']} bytes) endpoint {info2['endpoint']}")
    return augmented_all, strat_map

def main():
    parser = argparse.ArgumentParser(description="Two-phase batch for 1000/2000 initial placements")
    parser.add_argument("--corpus", default=CORPUS_1000, help="Input placements file (annotated); auto-prefers 2000 if exists when default")
    parser.add_argument("--prepare-only", action="store_true", help="Only build phase1 batch, don't call API")
    parser.add_argument("--submit", action="store_true", help="Build phase1 + upload + poll + inject phase2")
    parser.add_argument("--poll", type=str, default=None, help="Poll existing batch id (download + inject)")
    parser.add_argument("--inject", type=str, default=None, help="Inject from existing results file (no API)")
    parser.add_argument("--sample-players", type=int, default=None, help="Randomly sample k players per game (e.g. 2 → 2000 games => 4000 first + 4000 second, deterministic per game_id)")
    parser.add_argument("--sample-seed", type=int, default=12345, help="Seed for player sampling (per game_id offset)")
    parser.add_argument("--reasoning-effort", type=str, default="medium", choices=["low","medium","high","minimal"], help="Reasoning effort for gpt-5.6-luna (responses API)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--endpoint", default=ENDPOINT_RESPONSES, help="/v1/responses or /v1/chat/completions")
    parser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    # Auto-prefer 2000 corpus if default and 2000 file exists
    if args.corpus == CORPUS_1000 and Path(CORPUS_2000).exists() and not Path(args.corpus).exists():
        # edge: user left default but 1000 not yet finished, 2000 exists — keep 2000?
        pass
    if args.corpus == CORPUS_1000 and Path(CORPUS_2000).exists():
        # If 2000 exists and is larger, many users coming from 2000 plan will want it; but don't silently switch — keep default unless --corpus is explicit and 1000 is stale
        # We'll leave as is; user can pass --corpus explicitly
        pass

    _load_dotenv()
    # Apply reasoning effort globally for this run (affects build_batch_file)
    global REASONING_EFFORT
    REASONING_EFFORT = args.reasoning_effort

    if args.inject:
        # If sampling requested for inject, we need to apply it inside the called function
        # Monkey-patch inject to respect sampling flags via closure
        if args.sample_players is not None:
            orig_load = load_corpus
            def _sampled_load(path):
                recs = orig_load(path)
                return sample_two_players_per_game(recs, seed=args.sample_seed, players_per_game=args.sample_players)
            globals()["load_corpus"] = _sampled_load  # type: ignore
        inject_strategy_and_build_phase2(args.corpus, args.inject)
        return

    if args.poll:
        # need client
        from catan_llm.llm.openai_batch import OpenAIBatchClient
        base_url = None
        import os
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = OpenAIBatchClient(model=args.model, base_url=base_url, endpoint=args.endpoint, max_tokens=args.max_tokens)
        batch = client.retrieve_batch(args.poll)
        print(f"Polled {args.poll}: {getattr(batch,'status', batch.get('status') if isinstance(batch,dict) else '?')}")
        if getattr(batch,'status',None) == "completed" or (isinstance(batch, dict) and batch.get("status")=="completed"):
            paths_poll = phase_paths_for(args.corpus)
            path = download_results(client, batch, out_path=paths_poll["phase1_results"])
            inject_strategy_and_build_phase2(args.corpus, str(path))
        else:
            print(json.dumps(batch.model_dump() if hasattr(batch,'model_dump') else batch, indent=2, default=str)[:3000])
        return

    # Default / prepare flow
    recs = load_corpus(args.corpus)
    if args.sample_players is not None:
        before = len(recs)
        recs = sample_two_players_per_game(recs, seed=args.sample_seed, players_per_game=args.sample_players)
        print(f"Sampled {before} -> {len(recs)} via {args.sample_players} players/game (seed {args.sample_seed})")
    first, second = split_rounds(recs)
    print(f"Corpus {args.corpus}: {len(recs)} records = {len(first)} first + {len(second)} second")
    for rnd, cnt in [(1, len(first)), (2, len(second))]:
        print(f"  round {rnd}: {cnt} ({cnt/len(recs)*100:.1f}%)")
    # Show annotation example
    if recs:
        ex = recs[0]
        print(f"Example keys: {list(ex.keys())[:10]} ... placement_round={ex.get('placement_round')} is_first={ex.get('is_first_placement')} color={ex.get('color')} game={ex.get('game_id')}")

    # Derive phase paths from corpus stem (supports 1000 vs 2000 vs custom)
    paths = phase_paths_for(args.corpus)
    # Build phase1 batch
    p1_path = paths["phase1_batch"]
    info = build_batch_file(first, p1_path, args.model, args.endpoint, DEFAULT_SYSTEM_PROMPT, args.max_tokens)
    print(f"Phase1 batch built: {info['count']} requests -> {info['output_path']} ({info['bytes']} bytes) mapping {info['mapping_path']}")
    if info['count'] != len(first):
        print("WARN mismatch")

    if args.prepare_only:
        # Also show strategy injection preview (without real strategies - mock)
        print("\nPrepare-only: phase1 batch ready, second-phase will be built after collecting strategies.")
        print(f"To submit: venv/bin/python batch_two_phase.py --submit")
        # Demonstrate injection with dummy strategy
        demo_strategy = "Prioritize high-probability Ore/Wheat hub near center; retain Brick port option via Node 18 road toward coast for expansion."
        if second:
            demo_prompt = second[0]["prompt"]
            injected = inject_strategy_into_prompt(demo_prompt, demo_strategy)
            print("\n--- Demo injection (first second-placement prompt preview) ---")
            # Show strategy section snippet
            idx = injected.find("[CURRENT STRATEGY]")
            print(injected[idx: idx+600] + ("..." if len(injected[idx:])>600 else ""))
        return

    if args.submit:
        # Submit to OpenAI Zen
        from catan_llm.llm.openai_batch import OpenAIBatchClient
        import os
        base_url = os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY missing (check .env)")
            sys.exit(1)
        print(f"Using base_url={normalize_base_url(base_url) if base_url else 'default'} model={args.model} endpoint={args.endpoint}")
        client = OpenAIBatchClient(model=args.model, base_url=base_url, endpoint=args.endpoint, max_tokens=args.max_tokens)
        # Upload file
        print(f"Uploading {info['output_path']} ...")
        file_id = client.upload_file(info["output_path"])
        print(f"Uploaded file_id={file_id}")
        # Create batch
        batch = client.create_batch(file_id, endpoint=args.endpoint, completion_window="24h", metadata={"phase":"catan-phase1-first-placements"})
        batch_id = getattr(batch, "id", batch.get("id") if isinstance(batch, dict) else str(batch))
        print(f"Created batch_id={batch_id} — polling...")
        # Persist batch id
        Path("phase1_batch_id.txt").write_text(batch_id)
        print(f"Wrote batch id to phase1_batch_id.txt")
        batch_done = poll_batch(client, batch_id, interval=30, timeout=3600)
        status = getattr(batch_done, "status", batch_done.get("status") if isinstance(batch_done, dict) else "?")
        print(f"Final status: {status}")
        if status == "completed":
            paths_submit = phase_paths_for(args.corpus)
            path = download_results(client, batch_done, out_path=paths_submit["phase1_results"])
            inject_strategy_and_build_phase2(args.corpus, str(path))
        else:
            print(f"Batch not completed: {status}")
            if hasattr(batch_done, "model_dump"):
                print(json.dumps(batch_done.model_dump(), indent=2, default=str)[:4000])
            elif isinstance(batch_done, dict):
                print(json.dumps(batch_done, indent=2, default=str)[:4000])

if __name__ == "__main__":
    main()

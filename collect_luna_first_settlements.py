#!/usr/bin/env python3
"""
Collect gpt-5.6-luna reasoning for every first initial settlement (decision_id < 4)
across 10 games (40 prompts). Stores raw + parsed reasoning and builds a
Lavish-ready JSON for the visualizer.

Uses the Responses API spec for luna (no temperature) via OpenAIBatchClient.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
from catan_llm.llm.openai_batch import OpenAIBatchClient, _extract_response_text, parse_predicted_index, evaluate_response

CORPUS = "data/initial_placements/raw/initial_placements.jsonl"
OUT_JSONL = "data/luna/luna_first_settlements.jsonl"
OUT_SUMMARY = ".lavish/luna_first_settlements_data.json"
MODEL = "gpt-5.6-luna"

NUM_GAMES = 10

def parse_reasoning(text: str):
    strat = re.search(r"<strategy>(.*?)</strategy>", text or "", re.S | re.I)
    think = re.search(r"<think>(.*?)</think>", text or "", re.S | re.I)
    action = re.search(r"<action>\s*(\d+)\s*</action>", text or "", re.I)
    return {
        "strategy": strat.group(1).strip() if strat else None,
        "think": think.group(1).strip() if think else None,
        "action_raw": action.group(0) if action else None,
        "action_id": int(action.group(1)) if action else None,
    }

def main():
    # Load 40 records: game 0-9, decision 0-3
    all_recs = []
    with open(CORPUS) as f:
        for line in f:
            all_recs.append(json.loads(line))
    targets = [r for r in all_recs if r.get("game_id", -1) < NUM_GAMES and r.get("decision_id", 99) < 4]
    # sort by game then decision
    targets.sort(key=lambda r: (r["game_id"], r["decision_id"]))
    print(f"Selected {len(targets)} placements (games 0-{NUM_GAMES-1}, first round only)")

    client = OpenAIBatchClient(model=MODEL)

    results = []
    existing = {}
    # Resume if file exists
    if Path(OUT_JSONL).exists():
        with open(OUT_JSONL) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec.get("game_id"), rec.get("decision_id"))
                existing[key] = rec
        print(f"Found {len(existing)} existing results on disk — will skip those")

    for idx, rec in enumerate(targets):
        key = (rec["game_id"], rec["decision_id"])
        if key in existing:
            print(f"[{idx+1}/40] SKIP game={key[0]} dec={key[1]} {rec['color']} (already collected)")
            results.append(existing[key])
            continue

        print(f"[{idx+1}/40] game={rec['game_id']} dec={rec['decision_id']} {rec['color']} moves={rec['num_moves']} expected={rec['completion']} …", flush=True)
        tries = 0
        while tries < 3:
            try:
                eval_res = client.evaluate_single(rec)
                raw_text = eval_res.get("predicted_text") or ""
                raw_obj = eval_res.get("raw_response")
                usage = eval_res.get("usage")
                # Parse reasoning
                parsed = parse_reasoning(raw_text)
                # Fallback: _extract_response_text for reasoning_content case already handled
                predicted = eval_res.get("predicted_index")
                # Also try parsed action id if parse_predicted missed
                if predicted is None and parsed["action_id"] is not None:
                    predicted = parsed["action_id"]
                    eval_res["predicted_index"] = predicted
                    eval_res["correct"] = predicted == int(rec["completion"])
                    eval_res["valid"] = 1 <= predicted <= rec["num_moves"]

                # Extract usage dict
                usage_dict = None
                try:
                    if usage is not None:
                        if hasattr(usage, "model_dump"):
                            usage_dict = usage.model_dump()
                        elif hasattr(usage, "dict"):
                            usage_dict = usage.dict()
                        elif isinstance(usage, dict):
                            usage_dict = usage
                        else:
                            usage_dict = {"raw": str(usage)}
                except Exception:
                    usage_dict = None

                # Try to capture output_text shape for debug
                output_text_len = len(raw_text) if raw_text else 0
                # Check incomplete_details
                incomplete_reason = None
                try:
                    inc = getattr(raw_obj, "incomplete_details", None)
                    if inc is not None:
                        incomplete_reason = getattr(inc, "reason", None) or (inc.get("reason") if isinstance(inc, dict) else str(inc))
                except Exception:
                    pass

                out_rec = {
                    "game_id": rec["game_id"],
                    "decision_id": rec["decision_id"],
                    "color": rec["color"],
                    "phase": rec["phase"],
                    "num_moves": rec["num_moves"],
                    "expected_index": int(rec["completion"]),
                    "expected_label": rec.get("chosen_label"),
                    "move_labels": rec.get("move_labels"),
                    "prompt": rec["prompt"],
                    "selected_action": rec.get("selected_action"),
                    "next_road": rec.get("next_road"),
                    "predicted_index": predicted,
                    "predicted_label": eval_res.get("predicted_label"),
                    "correct": eval_res.get("correct"),
                    "valid": eval_res.get("valid"),
                    "raw_text": raw_text,
                    "strategy": parsed["strategy"],
                    "think": parsed["think"],
                    "action_id_parsed": parsed["action_id"],
                    "output_text_len": output_text_len,
                    "incomplete_reason": incomplete_reason,
                    "usage": usage_dict,
                }
                results.append(out_rec)
                # Append incrementally
                with open(OUT_JSONL, "a") as out:
                    out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                status = "✓" if out_rec["correct"] else "✗"
                extra = f" | incomplete={incomplete_reason}" if incomplete_reason else ""
                print(f"  -> {status} predicted={predicted} expected={rec['completion']} len={output_text_len}{extra}")
                if parsed["strategy"]:
                    print(f"     strategy: {parsed['strategy'][:120]}...")
                elif raw_text:
                    print(f"     raw preview: {raw_text[:200]!r}")
                break
            except Exception as e:
                tries += 1
                print(f"  !! error try {tries}: {e}")
                time.sleep(2 * tries)
                if tries >= 3:
                    results.append({"game_id": rec["game_id"], "decision_id": rec["decision_id"], "color": rec["color"], "error": str(e), "prompt": rec["prompt"]})
        time.sleep(0.6)

    # Load full set from disk for summary (in case resumed)
    all_results = []
    if Path(OUT_JSONL).exists():
        with open(OUT_JSONL) as f:
            for line in f:
                all_results.append(json.loads(line))
    else:
        all_results = results

    # Summary
    correct = sum(1 for r in all_results if r.get("correct"))
    valid = sum(1 for r in all_results if r.get("valid"))
    print(f"\nDone: {len(all_results)} total, correct {correct}/{len(all_results)}, valid {valid}/{len(all_results)}")

    # Build Lavish data file (subset for rendering — prompt is large so include full but compress)
    # Keep complete prompt; frontend will show truncated sections.
    Path(".lavish").mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY, "w") as f:
        json.dump({"model": MODEL, "corpus": CORPUS, "results": all_results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_SUMMARY} ({Path(OUT_SUMMARY).stat().st_size} bytes)")
    print(f"Wrote {OUT_JSONL} ({Path(OUT_JSONL).stat().st_size} bytes)")

if __name__ == "__main__":
    main()

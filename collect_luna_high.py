#!/usr/bin/env python3
"""
Collect gpt-5.6-luna HIGH reasoning for first initial settlements (games 0-9, dec 0-3).
Mirrors collect_luna_first_settlements.py but with reasoning={"effort":"high"}.
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, "src")
from catan_llm.llm.openai_batch import OpenAIBatchClient

CORPUS = "data/initial_placements/raw/initial_placements.jsonl"
OUT_JSONL = "data/luna/luna_high_first_settlements.jsonl"
OUT_SUMMARY = ".lavish/luna_high_data.json"
MODEL = "gpt-5.6-luna"
NUM_GAMES = 10
MAX_TOKENS = 8192  # high reasoning needs headroom (reasoning 1k-2k + output)

def parse_reasoning(text: str):
    import re as r
    strat = r.search(r"<strategy>(.*?)</strategy>", text or "", r.S | r.I)
    think = r.search(r"<think>(.*?)</think>", text or "", r.S | r.I)
    action = r.search(r"<action>\s*(\d+)\s*</action>", text or "", r.I)
    return {
        "strategy": strat.group(1).strip() if strat else None,
        "think": think.group(1).strip() if think else None,
        "action_id": int(action.group(1)) if action else None,
    }

def test_single():
    print("Testing single high-reasoning call...")
    client = OpenAIBatchClient(model=MODEL)
    import json as j
    with open(CORPUS) as f:
        rec = j.loads(next(iter(f)))
        # ensure it's a first-round placement
        for line in f:
            pass
    # reload sorted targets
    with open(CORPUS) as f:
        all_recs=[j.loads(l) for l in f]
    targets=[r for r in all_recs if r["game_id"]<NUM_GAMES and r["decision_id"]<4]
    targets.sort(key=lambda r:(r["game_id"], r["decision_id"]))
    rec=targets[0]
    print(f"Testing G{rec['game_id']} D{rec['decision_id']} {rec['color']} moves={rec['num_moves']}")
    res=client.evaluate_single(rec, max_tokens=MAX_TOKENS, extra_body={"reasoning":{"effort":"high"}})
    txt=res.get("predicted_text") or ""
    usage=res.get("usage")
    try:
        if hasattr(usage,"model_dump"): usage=usage.model_dump()
        elif hasattr(usage,"dict"): usage=usage.dict()
    except: pass
    print(f" predicted={res.get('predicted_index')} correct={res.get('correct')} valid={res.get('valid')}")
    print(f" usage={usage}")
    print(f" len={len(txt)} preview={(txt[:500] if txt else '')!r}")
    parsed=parse_reasoning(txt)
    print(f" strategy? {bool(parsed['strategy'])} think? {bool(parsed['think'])} action {parsed['action_id']}")
    return True

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="only test single call")
    args=parser.parse_args()
    if args.test_only:
        test_single()
        sys.exit(0)
    # full run
    with open(CORPUS) as f:
        all_recs=[json.loads(l) for l in f]
    targets=[r for r in all_recs if r["game_id"]<NUM_GAMES and r["decision_id"]<4]
    targets.sort(key=lambda r:(r["game_id"], r["decision_id"]))
    print(f"Selected {len(targets)} placements, max_tokens={MAX_TOKENS}, reasoning=high")
    client=OpenAIBatchClient(model=MODEL)
    existing={}
    if Path(OUT_JSONL).exists():
        with open(OUT_JSONL) as f:
            for line in f:
                r=json.loads(line)
                existing[(r["game_id"], r["decision_id"])]=r
        print(f"Resuming: {len(existing)} existing")
        if existing:
            # estimate reasoning avg
            vals=[(r.get("usage") or {}).get("output_tokens_details",{}).get("reasoning_tokens",0) for r in existing.values() if r.get("usage")]
            if vals:
                print(f" existing avg reasoning {sum(vals)/len(vals):.0f}, max {max(vals)}")

    for idx, rec in enumerate(targets):
        key=(rec["game_id"], rec["decision_id"])
        if key in existing:
            print(f"[{idx+1}/40] SKIP G{key[0]} D{key[1]} {rec['color']}")
            continue
        print(f"[{idx+1}/40] G{rec['game_id']} D{rec['decision_id']} {rec['color']} moves={rec['num_moves']} exp={rec['completion']} ...", flush=True)
        tries=0
        while tries<3:
            try:
                eval_res=client.evaluate_single(rec, max_tokens=MAX_TOKENS, extra_body={"reasoning":{"effort":"high"}})
                raw=eval_res.get("predicted_text") or ""
                raw_obj=eval_res.get("raw_response")
                usage=eval_res.get("usage")
                try:
                    if hasattr(usage,"model_dump"): usage=usage.model_dump()
                    elif hasattr(usage,"dict"): usage=usage.dict()
                except: pass
                parsed=parse_reasoning(raw)
                predicted=eval_res.get("predicted_index")
                if predicted is None and parsed["action_id"] is not None:
                    predicted=parsed["action_id"]
                    eval_res["predicted_index"]=predicted
                    eval_res["correct"]=predicted==int(rec["completion"])
                    eval_res["valid"]=1<=predicted<=rec["num_moves"]
                incomplete=None
                try:
                    inc=getattr(raw_obj,"incomplete_details",None)
                    if inc is not None:
                        incomplete=getattr(inc,"reason",None) or (inc.get("reason") if isinstance(inc, dict) else str(inc))
                except: pass
                out_rec={
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
                    "raw_text": raw,
                    "strategy": parsed["strategy"],
                    "think": parsed["think"],
                    "action_id_parsed": parsed["action_id"],
                    "output_text_len": len(raw) if raw else 0,
                    "incomplete_reason": incomplete,
                    "usage": usage if isinstance(usage, dict) else None,
                    "reasoning_effort": "high",
                    "max_tokens": MAX_TOKENS,
                }
                with open(OUT_JSONL,"a") as out:
                    out.write(json.dumps(out_rec, ensure_ascii=False)+"\n")
                status="✓" if out_rec["correct"] else "✗"
                rt=(usage or {}).get("output_tokens_details",{}).get("reasoning_tokens","?")
                ot=(usage or {}).get("output_tokens","?")
                print(f"  -> {status} pred={predicted} exp={rec['completion']} len={len(raw)} reasoning={rt} output={ot} incomplete={incomplete}")
                break
            except Exception as e:
                tries+=1
                print(f"  !! try {tries} error: {e}")
                time.sleep(2*tries)
                if tries>=3:
                    print(f"  FAILED G{rec['game_id']} D{rec['decision_id']}")
        time.sleep(0.7)

    # finalize summary
    all_results=[]
    if Path(OUT_JSONL).exists():
        with open(OUT_JSONL) as f:
            for line in f:
                all_results.append(json.loads(line))
    correct=sum(1 for r in all_results if r.get("correct"))
    valid=sum(1 for r in all_results if r.get("valid"))
    print(f"\nDone high: {len(all_results)} total correct {correct}/{len(all_results)} valid {valid}/{len(all_results)}")
    if all_results:
        vals=[(r.get("usage") or {}).get("output_tokens_details",{}).get("reasoning_tokens",0) for r in all_results if (r.get("usage") or {}).get("output_tokens_details",{}).get("reasoning_tokens") is not None]
        if vals:
            print(f" reasoning avg {sum(vals)/len(vals):.0f}, min {min(vals)}, max {max(vals)}")
        lens=[r.get("output_text_len",0) for r in all_results]
        print(f" output_text_len avg {sum(lens)/len(lens):.0f}")
    Path(".lavish").mkdir(parents=True, exist_ok=True)
    with open(OUT_SUMMARY,"w") as f:
        json.dump({"model":MODEL,"corpus":CORPUS,"reasoning_effort":"high","max_tokens":MAX_TOKENS,"results":all_results}, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_SUMMARY} and {OUT_JSONL}")

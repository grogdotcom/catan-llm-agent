"""
Facade for the unified teacher pipeline.

All implementation now lives in :mod:`catan_llm.llm.teacher` (request construction,
transport gateway, response parsing). This module exists to keep the historical
public surface — ``build_batch_jsonl``, ``OpenAIBatchClient``, the CLI, and the
underscore helpers used by the placement scripts — importable from one place.

Prefer importing from :mod:`catan_llm.llm.teacher` in new code.
"""

from __future__ import annotations

import json
import os
import re  # noqa: F401  (kept for any external wildcard import usage)
from pathlib import Path  # noqa: F401
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union  # noqa: F401

from catan_llm.llm.teacher import OpenAIBatchClient
from catan_llm.llm.teacher.gateway import TeacherGateway
from catan_llm.llm.teacher.models import (
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    MAX_BATCH_BYTES,
    MAX_BATCH_REQUESTS,
    MODEL_SPECS,
    ModelSpec,
    _extract_response_text,
    _load_dotenv,
    normalize_base_url,
    spec_for,
)
from catan_llm.llm.teacher.requests import (
    build_batch_jsonl,
    build_batch_request,
    build_batch_requests,
    build_chat_messages,
    custom_id_for,
    evaluate_response,
    iter_corpus,
    load_single_record,
    parse_predicted_index,
    validate_record,
)

# -- underscore aliases kept for scripts that import them ---------------------
_custom_id_for = custom_id_for
_validate_record = validate_record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    import argparse

    _load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Build an OpenAI Batch JSONL from a Catan corpus. "
            "Works for either initial_placements.jsonl or high_decision_moves.jsonl "
            "(or a merged list). Also supports --single for immediate evaluation of one decision."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input JSONL path(s) — placements and/or high-decision corpora (merged in order)",
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        default=None,
        help="Output Batch JSONL path (required for batch mode)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Chat model for body.model")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt text. Default: expert Catan prompt. Use --no-system-prompt to omit.",
    )
    parser.add_argument("--no-system-prompt", action="store_true", help="Omit system message (send only user prompt)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--limit", type=int, default=None, help="Max requests to include (batch mode)")
    parser.add_argument("--offset", type=int, default=0, help="Skip N records before building (batch mode)")
    parser.add_argument("--custom-id-prefix", default=None)
    parser.add_argument("--strict", action="store_true", help="Omit metadata (strict Batch spec, only custom_id/method/url/body)")
    parser.add_argument("--extra-body-json", default=None, help='JSON string merged into body (e.g. \'{"response_format": {"type": "json_object"}}\')')
    parser.add_argument(
        "--create-batch",
        action="store_true",
        help="Also upload and create the batch via the OpenAI API (requires OPENAI_API_KEY / --api-key)",
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY / .env OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="Custom API base URL (e.g. https://opencode.ai/zen/go) or set OPENAI_BASE_URL / .env OPENAI_BASE_URL")
    parser.add_argument("--env-file", default=None, help="Path to .env file to load (default: .env in CWD or repo root)")
    parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW)
    parser.add_argument("--metadata-json", default=None, help='JSON string for batch metadata (e.g. \'{"project": "catan"}\')')

    single_group = parser.add_argument_group("single-request immediate evaluation")
    single_group.add_argument("--single", action="store_true", help="Send a single decision immediately via chat.completions (not batch)")
    single_group.add_argument("--index", type=int, default=None, help="Positional index of the decision to evaluate (0-based)")
    single_group.add_argument("--game-id", type=int, default=None, help="Filter by game_id (use with --decision-id)")
    single_group.add_argument("--decision-id", type=int, default=None, help="Filter by decision_id")
    single_group.add_argument("--phase", type=str, default=None, help="Filter by phase (e.g. BUILD_INITIAL_SETTLEMENT, MOVE_ROBBER)")
    single_group.add_argument("--random", action="store_true", help="Pick a random decision from the corpus")
    single_group.add_argument("--seed", type=int, default=None, help="Seed for --random")
    single_group.add_argument("--dry-run", action="store_true", help="Single mode: show the request and expected answer without calling the API")
    single_group.add_argument("--show-prompt", action="store_true", help="Single mode: print the full prompt (truncated to 2000 chars by default)")
    single_group.add_argument("--show-prompt-full", action="store_true", help="Single mode: print the entire prompt without truncation")

    args = parser.parse_args()

    if args.env_file:
        _load_dotenv(args.env_file)

    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")
    if args.base_url is None:
        args.base_url = os.environ.get("OPENAI_BASE_URL")

    if args.no_system_prompt:
        system_prompt = None
    elif args.system_prompt is not None:
        system_prompt = args.system_prompt
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    extra_body = None
    if args.extra_body_json:
        extra_body = json.loads(args.extra_body_json)

    corpus_arg: Union[str, List[str]] = args.input[0] if len(args.input) == 1 else args.input

    if args.single:
        try:
            record, pos = load_single_record(
                corpus_arg,
                index=args.index,
                game_id=args.game_id,
                decision_id=args.decision_id,
                phase=args.phase,
                random_pick=args.random,
                seed=args.seed,
            )
        except Exception as e:
            parser.error(str(e))
            return

        custom_id = _custom_id_for(record, pos)

        if args.dry_run:
            messages = build_chat_messages(record, system_prompt=system_prompt)
            dry = {
                "custom_id": custom_id,
                "positional_index": pos,
                "model": args.model,
                "messages_preview": [
                    {"role": m["role"], "content": m["content"][:500] + ("..." if len(m["content"]) > 500 else "")}
                    for m in messages
                ],
                "expected_index": record.get("chosen_index", record.get("completion")),
                "expected_completion": str(record.get("completion", "")),
                "expected_label": record.get("chosen_label"),
                "phase": record.get("phase"),
                "color": record.get("color"),
                "game_id": record.get("game_id"),
                "decision_id": record.get("decision_id"),
                "num_moves": record.get("num_moves"),
            }
            if args.show_prompt or args.show_prompt_full:
                if args.show_prompt_full:
                    dry["prompt"] = record["prompt"]
                else:
                    full = record["prompt"]
                    dry["prompt"] = full[:2000] + ("... [truncated]" if len(full) > 2000 else "")
            print(json.dumps(dry, indent=2, ensure_ascii=False))
            if args.show_prompt or args.show_prompt_full:
                print("\n--- PROMPT ---")
                print(record["prompt"][:2000] + ("... [truncated]" if not args.show_prompt_full and len(record["prompt"]) > 2000 else record["prompt"] if args.show_prompt_full else ""))
            return

        client = OpenAIBatchClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
        )
        if args.no_system_prompt:
            result = client.evaluate_single(
                record,
                model=args.model,
                system_prompt=None,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
                use_default_when_none=False,
            )
        else:
            result = client.evaluate_single(
                record,
                model=args.model,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body=extra_body,
                use_default_when_none=True,
            )
        out = {k: v for k, v in result.items() if k not in ("raw_response", "record")}
        if "usage" in result and result["usage"] is not None:
            try:
                u = result["usage"]
                if hasattr(u, "model_dump"):
                    out["usage"] = u.model_dump()
                elif hasattr(u, "dict"):
                    out["usage"] = u.dict()
                else:
                    out["usage"] = u if isinstance(u, dict) else str(u)
            except Exception:
                out["usage"] = str(result["usage"])
        out["positional_index"] = pos
        out["custom_id"] = custom_id
        if args.show_prompt or args.show_prompt_full:
            out["prompt"] = record["prompt"] if args.show_prompt_full else record["prompt"][:2000]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        status = "✓ CORRECT" if out.get("correct") else "✗ WRONG"
        print(f"\n{status} — predicted {out.get('predicted_index')} vs expected {out.get('expected_index')} "
              f"(phase={out.get('phase')} color={out.get('color')} valid={out.get('valid')})")
        if out.get("expected_label"):
            print(f"  expected: {out['expected_label']}")
        if out.get("predicted_label"):
            print(f"  predicted: {out['predicted_label']}")
        return

    if args.output is None:
        parser.error("--output is required for batch mode (or use --single)")
        return

    if args.create_batch:
        client = OpenAIBatchClient(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            system_prompt=system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
            completion_window=args.completion_window,
        )
        metadata = json.loads(args.metadata_json) if args.metadata_json else None
        if args.no_system_prompt:
            prepare_info = build_batch_jsonl(
                corpus_arg, args.output,
                model=args.model, system_prompt=None,
                temperature=args.temperature, max_tokens=args.max_tokens,
                endpoint=args.endpoint, extra_body=extra_body,
                limit=args.limit, offset=args.offset,
                custom_id_prefix=args.custom_id_prefix, strict=args.strict,
            )
            file_id = client.upload_file(args.output)
            batch = client.create_batch(file_id, endpoint=args.endpoint, completion_window=args.completion_window, metadata=metadata)
            batch_id = getattr(batch, "id", batch.get("id") if isinstance(batch, dict) else str(batch))
            result = {"file_id": file_id, "batch": batch, "batch_id": batch_id, "prepare_info": prepare_info}
        else:
            result = client.create_batch_from_corpus(
                corpus_arg,
                output_path=args.output,
                model=args.model,
                system_prompt=system_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                endpoint=args.endpoint,
                extra_body=extra_body,
                limit=args.limit,
                offset=args.offset,
                custom_id_prefix=args.custom_id_prefix,
                strict=args.strict,
                metadata=metadata,
                completion_window=args.completion_window,
            )
        print(json.dumps({"prepare_info": result["prepare_info"], "file_id": result["file_id"], "batch_id": result["batch_id"]}, indent=2))
        if result["prepare_info"].get("warnings"):
            for w in result["prepare_info"]["warnings"]:
                print(f"warning: {w}")
    else:
        info = build_batch_jsonl(
            corpus_arg, args.output,
            model=args.model, system_prompt=system_prompt,
            temperature=args.temperature, max_tokens=args.max_tokens,
            endpoint=args.endpoint, extra_body=extra_body,
            limit=args.limit, offset=args.offset,
            custom_id_prefix=args.custom_id_prefix, strict=args.strict,
        )
        print(json.dumps(info, indent=2))
        if info.get("warnings"):
            for w in info["warnings"]:
                print(f"warning: {w}")


if __name__ == "__main__":
    _cli()

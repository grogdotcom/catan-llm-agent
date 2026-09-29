"""Corpus iteration and Batch request construction.

Turns corpus records (shared ``{prompt, completion, ...}`` schema, used by both
placements and SFT corpora) into OpenAI Batch lines or single-evaluation
messages. The transport itself lives in :mod:`catan_llm.llm.teacher.gateway`.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from catan_llm.llm.teacher.models import (
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    MAX_BATCH_BYTES,
    MAX_BATCH_REQUESTS,
)


def iter_corpus(
    corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
) -> Iterable[Dict[str, Any]]:
    """Yield records from one or more JSONL corpus files.

    Blank lines are skipped; invalid JSON raises ``ValueError`` with file+line.
    """
    if isinstance(corpus_path, (str, os.PathLike)):
        paths: List[Path] = [Path(corpus_path)]
    else:
        paths = [Path(p) for p in corpus_path]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"corpus file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"invalid JSON at {p}:{lineno}: {e}") from e
                if not isinstance(rec, dict):
                    raise ValueError(f"expected JSON object at {p}:{lineno}, got {type(rec).__name__}")
                yield rec


def validate_record(rec: Dict[str, Any], context: str = "") -> None:
    """Raise ValueError if rec lacks required prompt/completion fields."""
    if "prompt" not in rec or not isinstance(rec["prompt"], str) or not rec["prompt"].strip():
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}record missing non-empty string field 'prompt' (keys={list(rec.keys())})")
    if "completion" not in rec:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}record missing field 'completion' (keys={list(rec.keys())})")
    comp = rec["completion"]
    if comp is None or (isinstance(comp, str) and not comp.strip()):
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}record has empty 'completion'")


def custom_id_for(rec: Dict[str, Any], index: int) -> str:
    """Stable custom_id: game-decision--COLOR-PHASE, falling back to index."""
    game_id = rec.get("game_id")
    decision_id = rec.get("decision_id")
    color = rec.get("color", "")
    phase = rec.get("phase", "")
    if game_id is not None and decision_id is not None:
        base = f"game-{game_id}-decision-{decision_id}"
    elif game_id is not None:
        base = f"game-{game_id}-idx-{index}"
    elif decision_id is not None:
        base = f"decision-{decision_id}-idx-{index}"
    else:
        base = f"request-{index}"
    suffix_parts: List[str] = []
    if color:
        suffix_parts.append(str(color))
    if phase:
        suffix_parts.append(str(phase))
    if suffix_parts:
        return f"{base}--{'-'.join(suffix_parts)}"
    return base


def build_batch_request(
    record: Dict[str, Any],
    *,
    custom_id: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    endpoint: str = DEFAULT_ENDPOINT,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a single corpus record into one Batch API JSONL line."""
    validate_record(record, context=custom_id)
    prompt = record["prompt"]
    completion = str(record["completion"]).strip()

    messages: List[Dict[str, str]] = []
    if system_prompt is not None and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        body.update(extra_body)

    metadata: Dict[str, Any] = {
        "_expected_completion": completion,
        "_expected_index": record.get("chosen_index", completion),
        "_phase": record.get("phase"),
        "_color": record.get("color"),
        "_game_id": record.get("game_id"),
        "_decision_id": record.get("decision_id"),
        "_num_moves": record.get("num_moves"),
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    req: Dict[str, Any] = {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": body,
    }
    if metadata:
        req["metadata"] = metadata
    return req


def dedupe_custom_ids(requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Suffix duplicate custom_ids (defensive for merged corpora)."""
    seen: Dict[str, int] = {}
    for r in requests:
        cid = r["custom_id"]
        if cid in seen:
            seen[cid] += 1
            r["custom_id"] = f"{cid}--{seen[cid]}"
        else:
            seen[cid] = 0
    return requests


def build_batch_requests(
    corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    endpoint: str = DEFAULT_ENDPOINT,
    extra_body: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    custom_id_prefix: Optional[str] = None,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Build all Batch requests for a corpus in memory."""
    requests: List[Dict[str, Any]] = []
    for idx, rec in enumerate(iter_corpus(corpus_path)):
        if idx < offset:
            continue
        if limit is not None and len(requests) >= limit:
            break
        cid = custom_id_for(rec, idx)
        if custom_id_prefix:
            cid = f"{custom_id_prefix}{cid}"
        req = build_batch_request(
            rec,
            custom_id=cid,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint=endpoint,
            extra_body=extra_body,
        )
        if strict:
            req.pop("metadata", None)
        requests.append(req)
    return dedupe_custom_ids(requests)


def build_batch_jsonl(
    corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    output_path: Union[str, os.PathLike],
    *,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    endpoint: str = DEFAULT_ENDPOINT,
    extra_body: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    custom_id_prefix: Optional[str] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """Write a Batch JSONL file from a corpus.

    Returns ``{"output_path", "count", "bytes", "model", "endpoint", warnings}``.
    Raises ``ValueError`` if no requests were built.
    """
    requests = build_batch_requests(
        corpus_path,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        endpoint=endpoint,
        extra_body=extra_body,
        limit=limit,
        offset=offset,
        custom_id_prefix=custom_id_prefix,
        strict=strict,
    )
    if not requests:
        raise ValueError(
            "no batch requests built — corpus is empty or offset/limit excluded all records "
            f"(corpus_path={corpus_path!r}, offset={offset}, limit={limit})"
        )

    warnings: List[str] = []
    if len(requests) > MAX_BATCH_REQUESTS:
        warnings.append(f"batch has {len(requests)} requests exceeds OpenAI max {MAX_BATCH_REQUESTS} — split into multiple batches")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with out.open("w", encoding="utf-8") as f:
        for r in requests:
            line = json.dumps(r, ensure_ascii=False)
            total_bytes += len(line.encode("utf-8")) + 1
            f.write(line + "\n")

    if total_bytes > MAX_BATCH_BYTES:
        warnings.append(f"batch file is {total_bytes} bytes exceeds OpenAI max {MAX_BATCH_BYTES} — split or reduce max_tokens/prompt")

    return {
        "output_path": str(out),
        "count": len(requests),
        "bytes": total_bytes,
        "model": model,
        "endpoint": endpoint,
        "warnings": warnings,
    }


# -- Single-request helpers (immediate evaluation, not batch) -----------------

def build_chat_messages(
    record: Dict[str, Any],
    *,
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> List[Dict[str, str]]:
    """Build ``messages`` array for a single chat completion from a record."""
    validate_record(record, context="build_chat_messages")
    msgs: List[Dict[str, str]] = []
    if system_prompt is not None and system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": record["prompt"]})
    return msgs


def parse_predicted_index(
    response_text: str,
    *,
    num_moves: Optional[int] = None,
) -> Optional[int]:
    """Extract the Move ID from model text.

    Priority: ``<action>56</action>`` tag, then first integer in ``1..num_moves``.
    """
    if not response_text:
        return None
    text = str(response_text)
    m = re.search(r"<action>\s*(\d+)\s*</action>", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    candidates = re.findall(r"\d+", text)
    if not candidates:
        return None
    ints = [int(c) for c in candidates]
    if num_moves is not None:
        for v in ints:
            if 1 <= v <= num_moves:
                return v
        return ints[0]
    return ints[0]


def evaluate_response(
    response_text: str,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare a raw model response to a corpus record's ground truth."""
    validate_record(record, context="evaluate_response")
    expected = int(str(record["completion"]).strip())
    num_moves = record.get("num_moves")
    predicted = parse_predicted_index(response_text, num_moves=num_moves)

    valid = predicted is not None and (num_moves is None or 1 <= predicted <= num_moves)
    correct = predicted == expected

    result: Dict[str, Any] = {
        "expected_index": expected,
        "expected_completion": str(record["completion"]).strip(),
        "expected_label": record.get("chosen_label"),
        "predicted_text": response_text,
        "predicted_index": predicted,
        "correct": correct,
        "valid": valid,
        "num_moves": num_moves,
        "phase": record.get("phase"),
        "color": record.get("color"),
        "game_id": record.get("game_id"),
        "decision_id": record.get("decision_id"),
    }
    if predicted is not None and record.get("move_labels"):
        try:
            labels = record["move_labels"]
            if 1 <= predicted <= len(labels):
                result["predicted_label"] = labels[predicted - 1]
        except Exception:
            pass
    return result


def load_single_record(
    corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
    *,
    index: Optional[int] = None,
    game_id: Optional[int] = None,
    decision_id: Optional[int] = None,
    phase: Optional[str] = None,
    random_pick: bool = False,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """Load a single record by selector (index / game+decision / phase / random)."""
    if random_pick:
        all_recs = list(iter_corpus(corpus_path))
        if not all_recs:
            raise ValueError(f"corpus is empty: {corpus_path!r}")
        rng = random.Random(seed)
        idx = rng.randrange(len(all_recs))
        return all_recs[idx], idx

    if game_id is not None or decision_id is not None:
        for idx, rec in enumerate(iter_corpus(corpus_path)):
            if (game_id is not None and decision_id is not None and rec.get("game_id") == game_id and rec.get("decision_id") == decision_id):
                return rec, idx
            if game_id is not None and decision_id is None and rec.get("game_id") == game_id:
                return rec, idx
            if decision_id is not None and game_id is None and rec.get("decision_id") == decision_id:
                return rec, idx
        raise ValueError(f"no record found for game_id={game_id} decision_id={decision_id} in {corpus_path!r}")

    if phase is not None:
        for idx, rec in enumerate(iter_corpus(corpus_path)):
            if rec.get("phase") == phase:
                return rec, idx
        raise ValueError(f"no record found for phase={phase!r} in {corpus_path!r}")

    target = 0 if index is None else index
    if target < 0:
        all_recs = list(iter_corpus(corpus_path))
        try:
            return all_recs[target], len(all_recs) + target if target < 0 else target
        except IndexError:
            raise IndexError(f"corpus index {target} out of range (size {len(all_recs)})") from None
    for idx, rec in enumerate(iter_corpus(corpus_path)):
        if idx == target:
            return rec, idx
    raise IndexError(f"corpus index {target} out of range (size {idx + 1})")

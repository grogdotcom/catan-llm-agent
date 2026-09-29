"""Teacher pipeline: transport, requests, and response parsing.

The unified seam for everything that talks to a teacher LLM over the Batch or
single-inference API, shared by generic batches, two-phase placements, and
SFT checkpoints.
"""

from __future__ import annotations

from catan_llm.llm.teacher.gateway import OpenAIBatchClient, TeacherGateway
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
    normalize_base_url,
    spec_for,
)
from catan_llm.llm.teacher.parsing import (
    extract_action,
    extract_block,
    extract_strategy,
    extract_text_from_batch_body,
    extract_think,
    parse_teacher_response,
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

__all__ = [
    "TeacherGateway",
    "OpenAIBatchClient",
    "ModelSpec",
    "MODEL_SPECS",
    "spec_for",
    "normalize_base_url",
    "_extract_response_text",
    "DEFAULT_MODEL",
    "DEFAULT_ENDPOINT",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_COMPLETION_WINDOW",
    "MAX_BATCH_REQUESTS",
    "MAX_BATCH_BYTES",
    "iter_corpus",
    "validate_record",
    "custom_id_for",
    "build_batch_request",
    "build_batch_requests",
    "build_batch_jsonl",
    "build_chat_messages",
    "parse_predicted_index",
    "evaluate_response",
    "load_single_record",
    "parse_teacher_response",
    "extract_think",
    "extract_strategy",
    "extract_action",
    "extract_block",
    "extract_text_from_batch_body",
]

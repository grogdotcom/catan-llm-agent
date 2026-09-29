"""Batch transport gateway.

One thin adapter over the ``openai`` SDK that every pipeline (generic batch,
two-phase placements, SFT checkpoints) uses for upload, batch creation,
polling, downloads, and single-record inference. Hides Chat vs Responses API
divergence and the ``openai`` optional dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from catan_llm.llm.teacher.models import (
    DEFAULT_COMPLETION_WINDOW,
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    ModelSpec,
    _extract_response_text,
    _load_dotenv,
    normalize_base_url,
    spec_for,
)
from catan_llm.llm.teacher.requests import (
    build_batch_jsonl,
    build_chat_messages,
    evaluate_response,
    load_single_record,
)

import uuid as _uuid


class TeacherGateway:
    """Batch + single inference transport for a model/gateway combination.

    File preparation (``prepare_batch_file``) needs no API key; upload, batch
    creation, polling and single inference require the ``openai`` SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        endpoint: str = DEFAULT_ENDPOINT,
        completion_window: str = DEFAULT_COMPLETION_WINDOW,
        base_url: Optional[str] = None,
        client: Optional[Any] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.endpoint = endpoint
        self.completion_window = completion_window
        self.base_url = base_url
        self._api_key = api_key
        self._client = client
        # OpenCode Go requires x-opencode-session for efficient routing/cache.
        # Use explicit session_id, env OPENCODE_SESSION / X_OPENCODE_SESSION, or stable random.
        self._session_id = (
            session_id
            or os.environ.get("OPENCODE_SESSION")
            or os.environ.get("X_OPENCODE_SESSION")
            or os.environ.get("OPENCODE_SESSION_ID")
            or _uuid.uuid4().hex[:16]
        )

    # -- SDK access -----------------------------------------------------------

    def _session_headers(self) -> Dict[str, str]:
        """Return OpenCode session header when talking to zen/go, else empty."""
        effective = self.base_url or os.environ.get("OPENAI_BASE_URL") or ""
        if "opencode.ai/zen" in effective or "zen/go" in effective:
            return {"x-opencode-session": self._session_id}
        return {}

    def _get_openai_client(self) -> Any:
        """Return an ``openai.OpenAI`` client or raise with an install hint."""
        if self._client is not None:
            return self._client
        _load_dotenv()
        try:
            import openai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "the 'openai' package is required for upload/batch creation. "
                "Install it with `pip install openai` or use prepare_batch_file() "
                "to just build the JSONL without uploading."
            ) from e
        kwargs: Dict[str, Any] = {}
        api_key = self._api_key
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is not None:
            kwargs["api_key"] = api_key
        base_url = self.base_url
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url is not None:
            kwargs["base_url"] = normalize_base_url(base_url)
        # Attach default session header at client level (also sent per-request)
        hdrs = self._session_headers()
        if hdrs:
            kwargs["default_headers"] = hdrs
        self._client = openai.OpenAI(**kwargs)
        return self._client

    # -- File preparation (no API key) ----------------------------------------

    def prepare_batch_file(
        self,
        corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        output_path: Union[str, os.PathLike],
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        endpoint: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        custom_id_prefix: Optional[str] = None,
        strict: bool = False,
        use_defaults_when_none: bool = True,
    ) -> Dict[str, Any]:
        """Build a Batch JSONL file from a corpus (offline, no API key)."""
        if use_defaults_when_none:
            final_model = self.model if model is None else model
            final_system = self.system_prompt if system_prompt is None else system_prompt
            final_temp = self.temperature if temperature is None else temperature
            final_max = self.max_tokens if max_tokens is None else max_tokens
            final_endpoint = self.endpoint if endpoint is None else endpoint
        else:
            final_model = model if model is not None else self.model
            final_system = system_prompt
            final_temp = self.temperature if temperature is None else temperature
            final_max = self.max_tokens if max_tokens is None else max_tokens
            final_endpoint = self.endpoint if endpoint is None else endpoint

        return build_batch_jsonl(
            corpus_path,
            output_path,
            model=final_model,
            system_prompt=final_system,
            temperature=final_temp,
            max_tokens=final_max,
            endpoint=final_endpoint,
            extra_body=extra_body,
            limit=limit,
            offset=offset,
            custom_id_prefix=custom_id_prefix,
            strict=strict,
        )

    # -- Upload + batch creation ----------------------------------------------

    def upload_file(self, batch_file_path: Union[str, os.PathLike], *, purpose: str = "batch") -> str:
        """Upload a Batch JSONL file via the Files API. Returns file ID."""
        p = Path(batch_file_path)
        if not p.exists():
            raise FileNotFoundError(f"batch file not found: {p}")
        client = self._get_openai_client()
        with p.open("rb") as f:
            result = client.files.create(file=f, purpose=purpose)
        return getattr(result, "id", result.get("id") if isinstance(result, dict) else str(result))

    def create_batch(
        self,
        input_file_id: str,
        *,
        endpoint: Optional[str] = None,
        completion_window: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Create a Batch from an uploaded file. Returns the Batch object."""
        client = self._get_openai_client()
        return client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint or self.endpoint,
            completion_window=completion_window or self.completion_window,
            metadata=metadata,
        )

    def create_batch_from_corpus(
        self,
        corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        *,
        output_path: Union[str, os.PathLike, None] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        endpoint: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        custom_id_prefix: Optional[str] = None,
        strict: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        purpose: str = "batch",
        completion_window: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One-shot: build JSONL -> upload -> create batch."""
        if output_path is None:
            if isinstance(corpus_path, (str, os.PathLike)):
                stem = Path(corpus_path).stem
                output_path = Path(f"{stem}.batch.jsonl")
            else:
                output_path = Path("merged.batch.jsonl")

        prepare_info = self.prepare_batch_file(
            corpus_path,
            output_path,
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
        file_id = self.upload_file(prepare_info["output_path"], purpose=purpose)
        batch = self.create_batch(
            file_id,
            endpoint=endpoint,
            completion_window=completion_window,
            metadata=metadata,
        )
        batch_id = getattr(batch, "id", batch.get("id") if isinstance(batch, dict) else str(batch))
        return {
            "file_id": file_id,
            "batch": batch,
            "batch_id": batch_id,
            "prepare_info": prepare_info,
        }

    # -- Polling / download ---------------------------------------------------

    def retrieve_batch(self, batch_id: str) -> Any:
        """Retrieve a Batch by ID."""
        return self._get_openai_client().batches.retrieve(batch_id)

    def cancel_batch(self, batch_id: str) -> Any:
        """Cancel a Batch by ID."""
        return self._get_openai_client().batches.cancel(batch_id)

    def list_batches(self, limit: int = 20, after: Optional[str] = None) -> Any:
        """List batches."""
        return self._get_openai_client().batches.list(limit=limit, after=after)

    def download_batch_output(self, batch: Any, out_path: Union[str, os.PathLike]) -> Path:
        """Download a batch's output file to ``out_path`` and return the path."""
        out_file_id = getattr(batch, "output_file_id", None) or (batch.get("output_file_id") if isinstance(batch, dict) else None)
        if not out_file_id:
            raise ValueError(f"batch has no output_file_id: {batch}")
        raw = self._get_openai_client().files.content(out_file_id)
        if hasattr(raw, "text"):
            text = raw.text
        elif hasattr(raw, "read"):
            text = raw.read()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
        else:
            text = str(raw)
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    # -- Single-request immediate evaluation ----------------------------------

    def build_chat_messages(
        self,
        record: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        use_default_when_none: bool = True,
    ) -> List[Dict[str, str]]:
        """Build messages for a single immediate chat completion."""
        if use_default_when_none:
            sp = self.system_prompt if system_prompt is None else system_prompt
        else:
            sp = system_prompt
        return build_chat_messages(record, system_prompt=sp)

    def chat_completion(
        self,
        record: Dict[str, Any],
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        use_default_when_none: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Send a single record's prompt immediately (chat or responses).

        Returns the raw SDK object; use :meth:`evaluate_single` for scoring.
        """
        if use_default_when_none:
            final_model = self.model if model is None else model
            final_system = self.system_prompt if system_prompt is None else system_prompt
            final_temp = self.temperature if temperature is None else temperature
            final_max = self.max_tokens if max_tokens is None else max_tokens
        else:
            final_model = self.model if model is None else model
            final_system = system_prompt
            final_temp = self.temperature if temperature is None else temperature
            final_max = self.max_tokens if max_tokens is None else max_tokens
        messages = build_chat_messages(record, system_prompt=final_system)
        client = self._get_openai_client()
        effective_base = self.base_url or os.environ.get("OPENAI_BASE_URL") or ""
        spec = spec_for(final_model, effective_base)
        if os.environ.get("OPENAI_USE_RESPONSES") == "0":
            spec = ModelSpec(name=spec.name, api="chat", supports_temperature=True, default_max_tokens=spec.default_max_tokens)

        if final_max is not None and final_max == DEFAULT_MAX_TOKENS and spec.default_max_tokens != DEFAULT_MAX_TOKENS:
            final_max = spec.default_max_tokens

        # OpenCode Go session header (required for both chat and responses routing)
        session_headers = self._session_headers()

        if spec.api == "responses":
            body: Dict[str, Any] = {
                "model": final_model,
                "input": messages,
            }
            if spec.supports_temperature:
                if extra_body and "temperature" in extra_body:
                    body["temperature"] = extra_body.pop("temperature")
                elif "temperature" in kwargs:
                    body["temperature"] = kwargs.pop("temperature")
                elif final_temp is not None and final_temp != DEFAULT_TEMPERATURE:
                    body["temperature"] = final_temp
            else:
                extra_body = {k: v for k, v in (extra_body or {}).items() if k != "temperature"}
                kwargs.pop("temperature", None)
            if final_max is not None:
                body["max_output_tokens"] = final_max
            if extra_body:
                body.update(extra_body)
            body.update(kwargs)
            body.pop("max_tokens", None)
            body.pop("messages", None)
            if session_headers:
                # SDK supports extra_headers per-request; keep as extra_headers kwarg
                kwargs_headers = body.pop("extra_headers", {})
                merged = {**session_headers, **kwargs_headers}
                return client.responses.create(extra_headers=merged, **body)
            return client.responses.create(**body)

        body = {
            "model": final_model,
            "messages": messages,
            "temperature": final_temp,
            "max_tokens": final_max,
        }
        if extra_body:
            body.update(extra_body)
        body.update(kwargs)
        if session_headers:
            kwargs_headers = body.pop("extra_headers", {})
            merged = {**session_headers, **kwargs_headers}
            return client.chat.completions.create(extra_headers=merged, **body)
        return client.chat.completions.create(**body)

    def evaluate_single(
        self,
        record: Dict[str, Any],
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        use_default_when_none: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Send one record immediately and evaluate vs ground truth."""
        raw = self.chat_completion(
            record,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            use_default_when_none=use_default_when_none,
            **kwargs,
        )
        content = _extract_response_text(raw)
        eval_result = evaluate_response(content or "", record)
        eval_result["custom_id"] = self._make_custom_id(record)
        eval_result["predicted_text"] = content
        eval_result["raw_response"] = raw
        try:
            eval_result["usage"] = getattr(raw, "usage", None) or (raw.get("usage") if isinstance(raw, dict) else None)
        except Exception:
            eval_result["usage"] = None
        eval_result["model"] = model or self.model
        return eval_result

    def evaluate_single_from_corpus(
        self,
        corpus_path: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        *,
        index: Optional[int] = None,
        game_id: Optional[int] = None,
        decision_id: Optional[int] = None,
        phase: Optional[str] = None,
        random_pick: bool = False,
        seed: Optional[int] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        use_default_when_none: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pick one record from a corpus and evaluate immediately."""
        record, pos = load_single_record(
            corpus_path,
            index=index,
            game_id=game_id,
            decision_id=decision_id,
            phase=phase,
            random_pick=random_pick,
            seed=seed,
        )
        result = self.evaluate_single(
            record,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            use_default_when_none=use_default_when_none,
            **kwargs,
        )
        result["positional_index"] = pos
        result["record"] = record
        return result

    def _make_custom_id(self, record: Dict[str, Any]) -> str:
        from catan_llm.llm.teacher.requests import custom_id_for

        return custom_id_for(record, 0)


# Backwards-compatible name for imports that used the old client class.
OpenAIBatchClient = TeacherGateway

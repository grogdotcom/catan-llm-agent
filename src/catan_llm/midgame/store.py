"""Compatibility shim — midgame store now delegates to generic executor.

The canonical implementation lives in ``catan_llm.executor.store``.
This module preserves the midgame-specific defaults (``data/midgame/...``)
and re-exports the executor implementation for staged migration.
New code should import from ``catan_llm.executor``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from catan_llm.executor.store import (
    SCHEMA as _EXEC_SCHEMA,
    RunStore as _RunStore,
    _build_midgame_request,
    _build_request,
    _hash_bytes,
    _hash_file,
    _now,
    _prompt_with_strategy,
)

SCHEMA = _EXEC_SCHEMA


class MidgameRunStore(_RunStore):
    """Midgame wrapper that preserves legacy midgame path defaults.

    Delegates all logic to ``RunStore``; overrides batch-path defaults to
    ``data/midgame`` so existing callers without an explicit ``request_dir``
    keep writing to the legacy location. The generic executor defaults to
    ``data/executor``.
    """

    def create_batch_chunks(
        self,
        *,
        run_id: str,
        checkpoint_index: int,
        chunk_size: int = 100,
        request_dir: str = "data/midgame/batches",
        base_output_dir: Optional[str] = None,
    ) -> List[str]:
        return super().create_batch_chunks(
            run_id=run_id,
            checkpoint_index=checkpoint_index,
            chunk_size=chunk_size,
            request_dir=request_dir,
            base_output_dir=base_output_dir,
        )


# Executor aliases (canonical)
RunStore = _RunStore

__all__ = [
    "SCHEMA",
    "MidgameRunStore",
    "RunStore",
    "_build_midgame_request",
    "_build_request",
    "_hash_bytes",
    "_hash_file",
    "_now",
    "_prompt_with_strategy",
]

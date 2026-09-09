"""Compatibility shim — midgame orchestrator now delegates to generic executor.

Canonical implementation lives in ``catan_llm.executor.runner``.
"""

from __future__ import annotations

from catan_llm.executor.runner import (
    MidgameRunStore,
    RunExecutor,
    _get_client,
    poll_epoch,
    prepare_epoch,
    resume_run,
    submit_epoch,
)

__all__ = ["_get_client", "prepare_epoch", "submit_epoch", "poll_epoch", "resume_run", "RunExecutor", "MidgameRunStore"]

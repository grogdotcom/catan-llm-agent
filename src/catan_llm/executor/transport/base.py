"""Transport protocol for executor — batch vs inline.

Batch uses OpenAI Files + Batches (24h window, /v1/responses).
Inline uses direct TeacherGateway.chat_completion parallel (opencode).
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol


class Transport(Protocol):
    """Pluggable execution transport."""

    def prepare(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> List[str]:
        """Offline preparation (e.g. create request files). Idempotent. Returns job ids."""
        ...

    def submit(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        """Submit jobs for an epoch. Returns {chunk_ids, batch_ids/job_ids, ...}."""
        ...

    def poll(self, store: Any, run_id: str, checkpoint_index: int, **kwargs: Any) -> Dict[str, Any]:
        """Poll until terminal. Download results and update Store. Returns status dict."""
        ...

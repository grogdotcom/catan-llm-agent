"""RunSpec for generic executor.

Pipeline-agnostic chain definition: the executor does not know about SFT
bands or checkpoint selection policy; it only knows how to walk a list of
chains each consisting of ordered steps that chain strategy_in via lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


@dataclass(frozen=True)
class RetryPolicy:
    """Per-run retry policy for rejected responses.

    Default: record failures, no auto-requeue.
    """

    max_retries: int = 0
    requeue_rejected: bool = False


@dataclass
class RunSpec:
    """Specification for a durable executor run.

    Generic: pipeline supplies chains + bootstrap strategies; executor
    handles storage, batch transport, and lineage gating.
    """

    run_id: Optional[str] = None
    source_version: str = "v1"
    sample_seed: int = 42
    model: str = "gpt-5.6-luna"
    reasoning_effort: str = "medium"
    chunk_size: int = 100
    transport: Literal["batch", "inline"] = "batch"
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    # Bootstrap strategies keyed by (trajectory_id, seat_index) or chain key.
    bootstrap_strategies: Dict[Any, str] = field(default_factory=dict)
    # Optional hook: given a response text + context, returns True if accepted.
    # Generic executor records failures by default; pipelines provide stricter hooks.
    acceptance_hook: Optional[Callable[..., bool]] = None
    # Additional pipeline-specific config forwarded into runs.config_json
    extra_config: Dict[str, Any] = field(default_factory=dict)
    # Chain definitions: each chain is an ordered list of opportunity_ids or
    # dict descriptors. For batch SFT, chains are per (trajectory, seat).
    chains: List[Dict[str, Any]] = field(default_factory=list)

    def to_run_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "run_id": self.run_id,
            "source_version": self.source_version,
            "sample_seed": self.sample_seed,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "chunk_size": self.chunk_size,
            "transport": self.transport,
            "retry_policy": {"max_retries": self.retry_policy.max_retries, "requeue_rejected": self.retry_policy.requeue_rejected},
            "bootstrap_strategies": self.bootstrap_strategies,
            "status": "prepared",
            "current_checkpoint": 0,
        }
        cfg.update(self.extra_config)
        return cfg

"""Teacher model registry and gateway configuration.

Centralises all model/gateway divergence (chat vs responses API, temperature
support, token budgets, base-URL normalisation, response-text extraction) so no
other module needs scattered ``if "luna" in model`` checks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _load_dotenv(dotenv_path: Optional[Union[str, os.PathLike]] = None) -> None:
    """Load ``.env`` into ``os.environ`` without overwriting existing vars.

    - Looks for ``.env`` in CWD and repo root if no path given.
    - Never overwrites an already-set env var.
    - Silently no-ops if the file does not exist.
    - Does not require ``python-dotenv``.
    """
    candidates: List[Path] = []
    if dotenv_path is not None:
        candidates = [Path(dotenv_path)]
    else:
        cwd_env = Path.cwd() / ".env"
        candidates.append(cwd_env)
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidates.append(parent / ".env")
            if parent.name == "catan-llm-agent":
                break
        seen: set[str] = set()
        uniq: List[Path] = []
        for p in candidates:
            k = str(p)
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        candidates = uniq

    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("export "):
                    stripped = stripped[len("export ") :].strip()
                if "=" not in stripped:
                    continue
                key, val = stripped.split("=", 1)
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            # Never fail startup because .env is unreadable
            continue
        break


_load_dotenv()

# -- Defaults ----------------------------------------------------------------

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_ENDPOINT = "/v1/chat/completions"
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_SYSTEM_PROMPT = """
You are an expert Catan AI analyst. You will be provided with a board state, legal moves, and current strategic memory. A Grandmaster engine has already selected the optimal action. Your task is to rationalize this choice as if you were making the decision yourself.

[REASONING FLOW & GUIDELINES]

1. <think> (Tactical Scratchpad):
   Perform your detailed tactical, mathematical, and spatial analysis first.
   * Strategic Alignment: Read [CURRENT STRATEGY] (if present) and evaluate whether the current board state supports or disrupts that prior intent. Consider what mirco level moves
   * Draft Foresight (1st Placement): If making your first placement, assess board depth to evaluate which resource types or regions are likely to remain viable when the draft loops back.
   * Board Density & Strategy Choice: Assess resource distribution across the map. Determine whether the board favors securing balanced resource access OR leaning into high-yield specialized production (e.g., heavy Ore/Wheat city engine, port specialization).
   * Opponent State & Expansion: Analyze how existing opponent placements alter node value, choke expansion routes, or create race conditions.
   * Expansion Evaluation: Assess the potential for future expansion from the selected position such as potential settlement locations and port access.
   * Win Condition Validation: Evaluate which prospective move aligns with the quickest path to 10 victory points, considering both immediate and long-term implications.

2. <strategy> (Executive Macro Plan):
   Synthesize the math and analysis from your <think> block into a concise executive summary.
   * IF [CURRENT STRATEGY] is "None": Formulate an initial high-level draft plan for this initial placement (e.g., primary resource targets and expansion direction). Consider potential win conditions and how this placement and potential second placements support that path to victory.
   * IF [CURRENT STRATEGY] contains a prior plan: Refine or pivot the plan based on your <think> calculations. Your strategy should state your primary objective, key resource priorities, and intended path to 10 victory points.

   <strategy> (Executive Macro Plan):
   Do NOT simply summarize your <think> block or explain why you chose the current move.
   Instead, output a STATEFUL BLUEPRINT that remains valid for future turns.

   Your strategy MUST contain:
   * Active Target(s): Specific node IDs or assets you are building toward over the next 3-8 turns.
   * Resource Focus: Which resources you are actively accumulating or trading for.
   * Win Condition Vectors: Your planned distribution to reach 10 VPs (e.g., 3 Settlements, 2 Cities, Largest Army).
   * Contingency / Pivot Trigger: A specific board condition that would force you to abandon this plan.

3. <action> (Execution):
   Output ONLY the literal integer ID of the selected Grandmaster move.

[CRITICAL OVERRIDE RULE]
You are NOT playing Catan to choose your own move. You are rationalizing the Grandmaster engine's FIXED move N given in [DECISION REQUIRED - FIXED] as though you were deciding to pick N yourself. Build your <think> and <strategy> as the proactive reasoning that leads to selecting N, then output <action>N</action> exactly — never substitute your own preferred move. Any other integer = incorrect, even if your analysis favors it. Copy N verbatim.

[RESPONSE FORMAT]
You must output the tags in this EXACT order:
<think>Detailed tactical and mathematical scratchpad evaluation.</think>
<strategy>A concise executive plan grounded in the calculations above.</strategy>
<action>THE_SELECTED_ACTION_INDEX</action>
The <action> value MUST equal the engine's Move ID N verbatim from [DECISION REQUIRED - FIXED]. Do not re-evaluate or change it.
"""
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0

# OpenAI Batch limits (soft-enforced with warnings, not hard failures)
MAX_BATCH_REQUESTS = 50_000
MAX_BATCH_BYTES = 200 * 1024 * 1024  # 200 MB


@dataclass(frozen=True)
class ModelSpec:
    """Declarative spec for a model/gateway combination.

    Replaces scattered ``if "luna" in model`` / ``if "zen" in base_url``
    checks with a single lookup.
    """

    name: str
    api: str  # "chat" -> /v1/chat/completions, "responses" -> /v1/responses
    supports_temperature: bool
    default_max_tokens: int


# Registry: add a new entry when a new model/gateway appears — no code changes elsewhere.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt-4o-mini": ModelSpec(name="gpt-4o-mini", api="chat", supports_temperature=True, default_max_tokens=16),
    "gpt-4o": ModelSpec(name="gpt-4o", api="chat", supports_temperature=True, default_max_tokens=16),
    "gpt-5.6-luna": ModelSpec(name="gpt-5.6-luna", api="responses", supports_temperature=False, default_max_tokens=1024),
    "grok-4.6": ModelSpec(name="grok-4.6", api="responses", supports_temperature=True, default_max_tokens=1024),
    "grok-4-6": ModelSpec(name="grok-4.6", api="responses", supports_temperature=True, default_max_tokens=1024),
    "glm-5.3": ModelSpec(name="glm-5.3", api="chat", supports_temperature=True, default_max_tokens=4096),
    "glm-5.3-flash": ModelSpec(name="glm-5.3-flash", api="chat", supports_temperature=True, default_max_tokens=4096),
    "glm-5.2": ModelSpec(name="glm-5.2", api="chat", supports_temperature=True, default_max_tokens=4096),
    "glm-5.1": ModelSpec(name="glm-5.1", api="chat", supports_temperature=True, default_max_tokens=4096),
    "muse-spark-1.2-contributor": ModelSpec(name="muse-spark-1.2-contributor", api="responses", supports_temperature=False, default_max_tokens=8192),
    "muse-spark-1.2": ModelSpec(name="muse-spark-1.2", api="responses", supports_temperature=False, default_max_tokens=8192),
    "muse-spark": ModelSpec(name="muse-spark", api="responses", supports_temperature=False, default_max_tokens=8192),
    "muse-spark-1.3-contributor": ModelSpec(name="muse-spark-1.3-contributor", api="responses", supports_temperature=False, default_max_tokens=8192),
    "muse-spark-1.3": ModelSpec(name="muse-spark-1.3", api="responses", supports_temperature=False, default_max_tokens=8192),
    "zen-default": ModelSpec(name="zen-default", api="responses", supports_temperature=False, default_max_tokens=1024),
}


def normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    """Normalize a base URL to SDK-expected form (ending in /v1).

    Handles inputs like:
    - https://opencode.ai/zen/go          -> .../zen/go/v1
    - https://opencode.ai/zen/go/v1       -> .../zen/go/v1
    - https://opencode.ai/zen/go/v1/responses -> .../zen/go/v1
    - https://api.openai.com/v1           -> .../v1 (unchanged)
    """
    if base_url is None:
        return None
    b = base_url.strip().rstrip("/")
    if b.endswith("/v1/responses"):
        b = b[: -len("/v1/responses")] + "/v1"
    elif b.endswith("/responses"):
        b = b[: -len("/responses")]
        if not b.endswith("/v1"):
            b = b + "/v1"
    elif b.endswith("/chat/completions"):
        b = b[: -len("/chat/completions")]
        if not b.endswith("/v1"):
            b = b + "/v1"
    elif "/v1" not in b:
        b = b + "/v1"
    return b


def spec_for(model: str, base_url: Optional[str] = None) -> ModelSpec:
    """Resolve a ModelSpec for (model, base_url).

    Priority: exact registry entry -> substring heuristics -> zen gateway
    fallback -> generic chat default.
    """
    if model in MODEL_SPECS:
        return MODEL_SPECS[model]
    low_m = (model or "").lower()
    low_b = (base_url or "").lower()
    if "grok" in low_m:
        return MODEL_SPECS["grok-4.6"]
    if "luna" in low_m or "gpt-5.6" in low_m:
        return MODEL_SPECS["gpt-5.6-luna"]
    if "muse" in low_m or "spark" in low_m:
        return MODEL_SPECS["muse-spark-1.2-contributor"]
    if "glm" in low_m:
        return MODEL_SPECS["glm-5.3"]
    if "opencode.ai/zen" in low_b or "zen/go" in low_b:
        return MODEL_SPECS["zen-default"]
    return ModelSpec(name=model, api="chat", supports_temperature=True, default_max_tokens=DEFAULT_MAX_TOKENS)


def _extract_response_text(raw: Any) -> str:
    """Extract assistant text from either Chat or Responses shape.

    Handles chat ``choices[0].message.content`` + ``reasoning_content`` (GLM)
    and responses ``output_text`` / ``output[].type==message`` shapes, for both
    SDK objects and plain dicts.
    """
    try:
        text = getattr(raw, "output_text", None)
        if text:
            return text
        out = getattr(raw, "output", None)
        if out:
            for item in out:
                if getattr(item, "type", None) == "message":
                    c = getattr(item, "content", None)
                    if c and len(c) > 0 and getattr(c[0], "text", None):
                        return c[0].text
            for item in out:
                c = getattr(item, "content", None)
                if c and len(c) > 0 and getattr(c[0], "text", None):
                    return c[0].text
    except Exception:
        pass
    try:
        msg = raw.choices[0].message
        content = getattr(msg, "content", None) or ""
        if content and content.strip():
            return content
        rc = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if rc and isinstance(rc, str) and rc.strip():
            return rc
    except Exception:
        pass
    try:
        msgd = raw["choices"][0]["message"]
        if msgd.get("content"):
            return msgd["content"] or ""
        if msgd.get("reasoning_content"):
            return msgd["reasoning_content"] or ""
    except Exception:
        pass
    try:
        if isinstance(raw, dict):
            if raw.get("output_text"):
                return raw.get("output_text") or ""
            for item in raw.get("output", []):
                if item.get("type") == "message":
                    c = item.get("content", [])
                    if c and c[0].get("text"):
                        return c[0].get("text") or ""
            ch = raw.get("choices", [{}])[0].get("message", {})
            if ch.get("content"):
                return ch["content"] or ""
            if ch.get("reasoning_content"):
                return ch["reasoning_content"] or ""
    except Exception:
        pass
    return ""

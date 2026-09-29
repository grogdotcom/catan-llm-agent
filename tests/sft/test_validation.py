"""Validation gates §14."""

from catan_llm.llm.sft.validation import validate_response, count_strategy_blocks


def _prompt(strat="My plan"):
    return f"[CURRENT STRATEGY]\n{strat}\n\n[RECENT TURNS]\nblah\n\n[PLAYABLE MOVES]\n1. a\n2. b"


def _resp(think="t", strat="s", action=2):
    return f"<think>{think}</think><strategy>{strat}</strategy><action>{action}</action>"


def test_accepted():
    v = validate_response(prompt=_prompt(), response_text=_resp(think="ok", strat="ok", action=2), engine_completion="2", num_moves=3)
    assert v.accepted
    assert v.think_text == "ok"
    assert v.strategy_out == "ok"
    assert v.predicted_action == 2


def test_missing_think():
    v = validate_response(prompt=_prompt(), response_text="<strategy>s</strategy><action>1</action>", engine_completion="1", num_moves=2)
    assert not v.accepted and v.rejection_reason == "missing_think"


def test_missing_strategy():
    v = validate_response(prompt=_prompt(), response_text="<think>t</think><action>1</action>", engine_completion="1", num_moves=2)
    assert not v.accepted and v.rejection_reason == "missing_strategy"


def test_missing_action():
    v = validate_response(prompt=_prompt(), response_text="<think>t</think><strategy>s</strategy>", engine_completion="1", num_moves=2)
    assert not v.accepted and v.rejection_reason == "missing_action"


def test_invalid_action_range():
    v = validate_response(prompt=_prompt(), response_text=_resp(action=5), engine_completion="5", num_moves=3)
    assert not v.accepted and v.rejection_reason == "invalid_action_range"


def test_wrong_alpha_beta_action():
    v = validate_response(prompt=_prompt(), response_text=_resp(action=2), engine_completion="3", num_moves=5)
    assert not v.accepted and v.rejection_reason == "wrong_alpha_beta_action"


def test_duplicate_strategy_block():
    prompt = "[CURRENT STRATEGY]\nA\n\n[CURRENT STRATEGY]\nB\n\n[RECENT TURNS]"
    v = validate_response(prompt=prompt, response_text=_resp(), engine_completion="2", num_moves=3)
    assert not v.accepted and v.rejection_reason == "duplicate_strategy_block"


def test_missing_strategy_block():
    v = validate_response(prompt="[PLAYABLE MOVES]\n1. a", response_text=_resp(), engine_completion="2", num_moves=3)
    assert not v.accepted and v.rejection_reason == "duplicate_strategy_block"


def test_broken_lineage():
    v = validate_response(prompt=_prompt(), response_text=_resp(), engine_completion="2", num_moves=3, strategy_lineage_ok=False)
    assert not v.accepted and v.rejection_reason == "broken_strategy_lineage"


def test_count_strategy_blocks():
    assert count_strategy_blocks(_prompt()) == 1
    assert count_strategy_blocks("[CURRENT STRATEGY]\nA\n\n[CURRENT STRATEGY]\nB") == 2
    assert count_strategy_blocks("nope") == 0


def test_empty_think_rejected():
    v = validate_response(prompt=_prompt(), response_text="<think>   </think><strategy>s</strategy><action>1</action>", engine_completion="1", num_moves=2)
    assert not v.accepted and v.rejection_reason == "missing_think"


def test_empty_strategy_rejected():
    v = validate_response(prompt=_prompt(), response_text="<think>t</think><strategy>   </strategy><action>1</action>", engine_completion="1", num_moves=2)
    assert not v.accepted and v.rejection_reason == "missing_strategy"

"""Strategy-chain lineage policy."""

from catan_llm.llm.strategy import LineageDecision, resolve_lineage


def test_first_checkpoint_bootstraps_from_phase2():
    d = resolve_lineage(1, phase2_strategy="Go ore", prev_accepted_strategy=None)
    assert d.can_run
    assert d.strategy_in == "Go ore"
    assert d.strategy_source is None
    assert d.skip_reason is None


def test_first_checkpoint_missing_bootstrap_skipped():
    d = resolve_lineage(1, phase2_strategy=None)
    assert not d.can_run
    assert d.status == "skipped"
    assert d.skip_reason == "missing_bootstrap_strategy"
    assert d.strategy_in is None


def test_later_checkpoint_chains_from_accepted():
    d = resolve_lineage(2, phase2_strategy=None, prev_accepted_strategy="Prev", prev_checkpoint_id="c1")
    assert d.can_run
    assert d.strategy_in == "Prev"
    assert d.strategy_source == "c1"
    assert d.skip_reason is None


def test_later_checkpoint_broken_chain_skipped():
    d = resolve_lineage(3, phase2_strategy="X", prev_accepted_strategy=None)
    assert not d.can_run
    assert d.status == "skipped"
    assert d.skip_reason == "broken_strategy_lineage"


def test_later_checkpoint_ignores_phase2():
    # phase2 only seeds the first checkpoint, never later ones
    d = resolve_lineage(2, phase2_strategy="Bootstrap", prev_accepted_strategy=None)
    assert d.skip_reason == "broken_strategy_lineage"
    assert d.strategy_in is None


def test_lineage_decision_frozen():
    d = LineageDecision("s", None, "pending", None)
    assert d.can_run

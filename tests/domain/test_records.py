"""Domain records round-trip and defaults."""

from catan_llm.domain import (
    BatchRequest,
    DecisionRecord,
    PromptArtifact,
    PromptContext,
    TeacherResponse,
    Trajectory,
    ValidationResult,
)


def test_decision_record_round_trip():
    rec = DecisionRecord(
        prompt="hello",
        completion="3",
        chosen_index=3,
        phase="PLAY_TURN",
        color="RED",
        game_id=7,
        decision_id=2,
        trajectory_id="seed-1",
        extra={"custom": 1},
    )
    d = rec.as_dict()
    assert d["prompt"] == "hello"
    assert d["custom"] == 1
    again = DecisionRecord.from_dict(d)
    assert again.prompt == "hello"
    assert again.trajectory_id == "seed-1"
    assert again.extra == {"custom": 1}


def test_decision_record_omits_none():
    rec = DecisionRecord(prompt="p", completion="1")
    assert "color" not in rec.as_dict()
    assert "chosen_index" not in rec.as_dict()


def test_prompt_artifact_defaults():
    a = PromptArtifact(text="t")
    assert a.moves == []
    assert a.version == "1.0"


def test_prompt_context_defaults():
    c = PromptContext(observation=object())
    assert c.history_window_size == 8
    assert c.include_footer is True
    assert c.current_strategy is None


def test_batch_request_strict():
    req = BatchRequest(
        custom_id="c", method="POST", url="/v1/responses", body={"model": "m"}, metadata={"k": "v"}
    )
    strict = req.as_dict(strict=True)
    assert "metadata" not in strict
    loose = req.as_dict(strict=False)
    assert loose["metadata"] == {"k": "v"}


def test_trajectory_round_trip():
    t = Trajectory(trajectory_id="x", game_seed=1, winner="RED", seat_order=["RED", "BLUE"])
    d = t.as_dict()
    assert d["game_seed"] == 1
    assert "source_hash" not in d


def test_teacher_response_and_validation_defaults():
    assert TeacherResponse().raw_text == ""
    v = ValidationResult(accepted=True)
    assert v.rejection_reason is None
    assert v.predicted_action is None

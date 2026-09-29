"""Teacher package: request building, parsing, gateway wiring."""

import json

import pytest

from catan_llm.llm.teacher import (
    DEFAULT_MODEL,
    TeacherGateway,
    build_batch_jsonl,
    build_batch_request,
    build_chat_messages,
    evaluate_response,
    extract_strategy,
    extract_think,
    iter_corpus,
    parse_predicted_index,
    parse_teacher_response,
    spec_for,
)
from catan_llm.llm.teacher.models import _extract_response_text, normalize_base_url

CORPUS = [
    {"prompt": "p1", "completion": "2", "game_id": 0, "decision_id": 1, "phase": "PLAY_TURN", "color": "RED", "num_moves": 5},
    {"prompt": "p2", "completion": "1", "game_id": 0, "decision_id": 2, "phase": "PLAY_TURN", "color": "RED", "num_moves": 5},
]


def _write_corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in CORPUS) + "\n")
    return p


def test_build_batch_jsonl_and_requests(tmp_path):
    src = _write_corpus(tmp_path)
    out = tmp_path / "batch.jsonl"
    info = build_batch_jsonl(src, out, model="gpt-4o-mini")
    assert info["count"] == 2
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines[0]["body"]["model"] == "gpt-4o-mini"
    assert lines[0]["custom_id"] == "game-0-decision-1--RED-PLAY_TURN"
    # strict strips metadata at the batch-builder level
    req = build_batch_request(CORPUS[0], custom_id="c1")
    assert "metadata" in req


def test_iter_corpus_and_validation(tmp_path):
    src = _write_corpus(tmp_path)
    recs = list(iter_corpus(src))
    assert len(recs) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": 1}\n')
    with pytest.raises(ValueError):
        build_batch_request(json.loads('{"prompt": 1}'), custom_id="x")


def test_spec_and_normalize_url():
    assert spec_for("gpt-5.6-luna").api == "responses"
    assert spec_for("glm-5.3").default_max_tokens == 4096
    assert normalize_base_url("https://opencode.ai/zen/go") == "https://opencode.ai/zen/go/v1"
    assert normalize_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"


def test_parse_predicted_and_evaluate():
    assert parse_predicted_index("<action>3</action> xyz") == 3
    assert parse_predicted_index("I pick 2", num_moves=4) == 2
    res = evaluate_response("<action>2</action>", CORPUS[0])
    assert res["correct"] is True
    assert res["expected_index"] == 2
    res2 = evaluate_response("5", CORPUS[0])
    assert res2["correct"] is False


def test_extract_response_text_shapes():
    class Msg:
        content = "c"

    class Choice:
        message = Msg()

    class Raw:
        choices = [Choice()]

    assert _extract_response_text(Raw()) == "c"
    assert _extract_response_text({"output_text": "hi"}) == "hi"
    assert _extract_response_text({"choices": [{"message": {"content": "z"}}]}) == "z"
    assert _extract_response_text(object()) == ""


def test_parse_teacher_response_blocks():
    r = parse_teacher_response("<think>t</think><strategy>s</strategy><action>1</action>")
    assert r.think_text == "t"
    assert r.strategy == "s"
    assert r.action == 1
    assert extract_think("<think> x </think>") == "x"
    assert extract_strategy("<strategy> s </strategy>") == "s"
    assert parse_teacher_response("").action is None


def test_gateway_messages():
    g = TeacherGateway(model=DEFAULT_MODEL)
    msgs = g.build_chat_messages(CORPUS[0])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "p1"
    no_sys = g.build_chat_messages(CORPUS[0], use_default_when_none=False, system_prompt=None)
    assert len(no_sys) == 1


class _Fake:
    def __init__(self, text="<action>2</action>"):
        self.output_text = text
        self.usage = None


class _FakeResponses:
    def __init__(self):
        self.client = self

    def create(self, **kwargs):
        return _Fake()


class _FakeSDK:
    def __init__(self):
        self.responses = _FakeResponses()
        self.batches = _FakeResponses()
        self.files = _FakeResponses()


def test_gateway_single_uses_client():
    g = TeacherGateway(client=_FakeSDK(), model="gpt-5.6-luna")
    res = g.evaluate_single(CORPUS[0])
    assert res["correct"] is True
    assert res["custom_id"] == "game-0-decision-1--RED-PLAY_TURN"

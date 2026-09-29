"""openai_batch facade CLI: batch (offline) and single dry-run."""

import json
import subprocess
import sys
from pathlib import Path

CORPUS = [
    {"prompt": "p1", "completion": "2", "game_id": 0, "decision_id": 1, "phase": "PLAY_TURN", "color": "RED", "num_moves": 5},
    {"prompt": "p2", "completion": "1", "game_id": 0, "decision_id": 2, "phase": "PLAY_TURN", "color": "RED", "num_moves": 5},
]


def _write(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in CORPUS) + "\n")
    return p


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "catan_llm.llm.openai_batch", *args],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        cwd="/Users/gregberkman/SOC/catan-llm-agent",
    )


def test_cli_batch_mode_offline(tmp_path):
    src = _write(tmp_path)
    out = tmp_path / "out.batch.jsonl"
    r = _run(["-i", str(src), "-o", str(out), "--model", "gpt-4o-mini"])
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info["count"] == 2
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert lines[0]["body"]["model"] == "gpt-4o-mini"


def test_cli_single_dry_run(tmp_path):
    src = _write(tmp_path)
    r = _run(["--single", "-i", str(src), "--index", "0", "--dry-run"])
    assert r.returncode == 0, r.stderr
    dry = json.loads(r.stdout)
    assert dry["custom_id"] == "game-0-decision-1--RED-PLAY_TURN"
    assert dry["expected_completion"] == "2"


def test_cli_batch_requires_output(tmp_path):
    src = _write(tmp_path)
    r = _run(["-i", str(src)])
    assert r.returncode != 0
    assert "--output is required" in r.stderr

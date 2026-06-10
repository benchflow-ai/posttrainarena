"""
Tests that score a single trial. Pytest passes → reward = 1.0;
any failure → reward = 0.0. Use one test class per check group so the
failure surface is readable when the agent gets it partially right.

The agent writes its output to BENCHFLOW_WORKSPACE (default `/root`).
This template assumes it writes to `/root/answer.json`; adjust to match
your task.
"""

import json
import os
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
ANSWER_FILE = WORKSPACE / "answer.json"


class TestAnswerFileExists:
    """Did the agent write a valid output?"""

    def test_file_exists(self):
        assert ANSWER_FILE.exists(), f"Answer file not found at {ANSWER_FILE}"

    def test_file_is_valid_json(self):
        with open(ANSWER_FILE) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Answer file is not valid JSON: {e}")


class TestAnswerContent:
    """Does the output match the expected schema and values?

    Replace these placeholders with real per-task checks.
    """

    def test_has_required_keys(self):
        with open(ANSWER_FILE) as f:
            data = json.load(f)
        # Example: assert "result" in data, "Missing required key: 'result'"
        assert isinstance(data, dict), "Answer must be a JSON object"

    def test_result_matches_expected(self):
        # Example check — delete or replace.
        # with open(ANSWER_FILE) as f:
        #     data = json.load(f)
        # assert data["result"] == 42, f"Expected 42, got {data['result']}"
        pass

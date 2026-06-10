"""Score the dogfood-hello-text trial.

Verifies the agent wrote a JSON file at /root/answer.json containing
the exact three required fields. Passing every test → reward 1.0.
"""

import json
import os
from pathlib import Path

import pytest

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
ANSWER_FILE = WORKSPACE / "answer.json"

EXPECTED = {
    "greeting": "hello",
    "subject": "post-training",
    "year": 2026,
}


class TestAnswerFileExists:
    def test_file_exists(self):
        assert ANSWER_FILE.exists(), f"Answer file not found at {ANSWER_FILE}"

    def test_file_is_valid_json(self):
        with open(ANSWER_FILE) as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Answer file is not valid JSON: {e}")


class TestAnswerContent:
    def _load(self) -> dict:
        with open(ANSWER_FILE) as f:
            data = json.load(f)
        assert isinstance(data, dict), "Answer must be a JSON object"
        return data

    def test_has_all_required_keys(self):
        data = self._load()
        missing = sorted(set(EXPECTED) - set(data))
        assert not missing, f"Missing required keys: {missing}"

    def test_greeting(self):
        assert self._load()["greeting"] == EXPECTED["greeting"]

    def test_subject(self):
        assert self._load()["subject"] == EXPECTED["subject"]

    def test_year(self):
        assert self._load()["year"] == EXPECTED["year"]

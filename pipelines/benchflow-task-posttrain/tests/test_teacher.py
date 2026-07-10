from __future__ import annotations

import json

from posttrainarena.benchflow_pipeline.teacher import TOOLS, _exchange


def test_exchange_matches_benchflow_sft_converter_shape() -> None:
    row = _exchange(
        model="teacher",
        request_messages=[{"role": "user", "content": "solve"}],
        response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "run_bash",
                                    "arguments": json.dumps({"command": "ls"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        duration_ms=3,
    )

    assert row["request"]["body"]["tools"] == TOOLS
    assert (
        row["response"]["body"]["choices"][0]["message"]["tool_calls"][0]["function"][
            "name"
        ]
        == "run_bash"
    )

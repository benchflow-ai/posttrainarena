from __future__ import annotations

from pathlib import Path

from posttrainarena.benchflow_pipeline.sft import render_rows


def test_render_rows_preserves_tool_definitions(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"solve"},{"role":"assistant","content":"done"}],'
        '"tool_defs":[{"type":"function","function":{"name":"submit","parameters":{"type":"object"}}}]}\n'
    )

    class Tokenizer:
        def apply_chat_template(
            self, messages, *, tools, tokenize, add_generation_prompt
        ):
            assert messages[-1]["content"] == "done"
            assert tools[0]["function"]["name"] == "submit"
            assert tokenize is False
            assert add_generation_prompt is False
            return "rendered"

    assert render_rows(source, Tokenizer()) == [{"text": "rendered"}]

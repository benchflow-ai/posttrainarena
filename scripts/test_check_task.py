"""Tests for scripts/check_task.py's warning about a verifier that needs the network the task turns off.

Runs standalone (`python3 scripts/test_check_task.py`) or under pytest. No third-party deps.
The template's verifier runs pytest from the image; the one before it downloaded pytest with apt-get, curl and uvx
when it ran, which fails or hangs in a sandbox without network (allow_internet: false): on Daytona the oracle's
verifier hung until BenchFlow's 180 s limit, three times out of three, so even the reference solution couldn't pass.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_task as C

KIT = Path(__file__).resolve().parent.parent / "starting-kit"
TEMPLATE = KIT / "template"

# The template's verifier before it ran pytest from the image (posttrainarena main at bcbaffb): the download ran
# unless BENCHFLOW_SKIP_VERIFIER_DEPS=1, which nothing sets.
OLD_TEST_SH = """#!/bin/bash
PYTEST_BIN="${BENCHFLOW_PYTEST_BIN:-pytest}"
if [ "${BENCHFLOW_SKIP_VERIFIER_DEPS:-0}" = "1" ]; then
  "$PYTEST_BIN" --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v
else
  apt-get update
  apt-get install -y curl

  curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh
  source $HOME/.local/bin/env

  uvx \\
    --with pytest==8.4.1 \\
    --with pytest-json-ctrf==0.3.5 \\
    pytest --ctrf /logs/verifier/ctrf.json /verifier/test_outputs.py -rA -v
fi
"""
OLD_DOCKERFILE = """FROM ubuntu:24.04
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /root
"""


def _task(parent, name="my-task", test_sh=None, dockerfile=None, internet=None, block="environment"):
    """A copy of the template, with its verifier, Dockerfile or network setting replaced."""
    task = Path(parent) / name
    shutil.copytree(TEMPLATE, task)
    if test_sh is not None:
        (task / "verifier" / "test.sh").write_text(test_sh)
    if dockerfile is not None:
        (task / "environment" / "Dockerfile").write_text(dockerfile)
    md = task / "task.md"
    text = md.read_text()
    if internet is not None:
        text = text.replace("  allow_internet: false", f"  {internet}")
    if block != "environment":
        text = text.replace("\nenvironment:\n", f"\n{block}:\n")
    md.write_text(text)
    return task


def _warnings(**kw):
    with tempfile.TemporaryDirectory() as tmp:
        task = _task(tmp, **kw)
        if kw.get("block", "environment") == "environment":   # this script's structure check wants `environment:`
            assert C.check_task(task) == [], C.check_task(task)   # a warning never makes the structure invalid
        return C.warnings_for(task)


def test_the_template_runs_its_verifier_offline():
    assert C.warnings_for(TEMPLATE) == []
    assert C.image_installs_pytest((TEMPLATE / "environment" / "Dockerfile").read_text())
    assert "allow_internet: false" in (TEMPLATE / "task.md").read_text()


def test_no_example_needs_the_network_it_turns_off():
    for task in sorted((KIT / "examples").iterdir()):
        if (task / "task.md").is_file():
            assert C.warnings_for(task) == [], task.name


def test_a_verifier_that_downloads_without_network_warns():
    found = _warnings(test_sh=OLD_TEST_SH, dockerfile=OLD_DOCKERFILE)
    assert len(found) == 1
    assert "downloads when it runs (apt-get install, curl, uvx)" in found[0]
    assert "allow_internet: false" in found[0] and "so even the reference solution can't pass" in found[0]


def test_the_old_verifier_warns_even_with_pytest_in_the_image():
    # its download isn't a fallback: it runs unless BENCHFLOW_SKIP_VERIFIER_DEPS=1
    assert _warnings(test_sh=OLD_TEST_SH)


def test_a_fallback_download_warns_only_when_the_image_lacks_pytest():
    fallback = (TEMPLATE / "verifier" / "test.sh").read_text()
    assert _warnings(test_sh=fallback) == []
    found = _warnings(test_sh=fallback, dockerfile=OLD_DOCKERFILE)
    assert len(found) == 1 and "environment/Dockerfile doesn't install pytest" in found[0]


def test_a_task_with_network_never_warns():
    assert _warnings(test_sh=OLD_TEST_SH, internet="allow_internet: true") == []
    assert _warnings(test_sh=OLD_TEST_SH, internet="cpus: 1") == []   # no allow_internet: the spec's default is true


def test_benchflow_native_blocks_count_too():
    assert _warnings(test_sh=OLD_TEST_SH, block="sandbox")
    assert _warnings(test_sh=OLD_TEST_SH, internet="network_mode: no-network")


def test_comments_and_other_downloads():
    commented = "#!/bin/bash\n# we used to run: uvx --with pytest pytest\npytest /verifier/test_outputs.py\n"
    assert _warnings(test_sh=commented) == []
    pip = "#!/bin/bash\npip install pytest==8.4.1\npytest /verifier/test_outputs.py\n"
    assert "(pip install)" in _warnings(test_sh=pip, dockerfile=OLD_DOCKERFILE)[0]


def test_main_prints_the_warning_and_still_exits_0():
    import contextlib
    import io
    out = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        _task(tmp, test_sh=OLD_TEST_SH, dockerfile=OLD_DOCKERFILE)
        with contextlib.redirect_stdout(out):
            code = C.main(["check_task.py", tmp])
    assert code == 0
    assert "✓ my-task — structure valid" in out.getvalue() and "⚠ warning: verifier/test.sh downloads" in out.getvalue()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"{len(tests)} passed")

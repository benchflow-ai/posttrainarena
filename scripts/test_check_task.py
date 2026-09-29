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


def _task(parent, name="my-task", test_sh=None, dockerfile=None, internet=None, block="environment", test_outputs=None):
    """A copy of the template, with its verifier, Dockerfile or network setting replaced."""
    task = Path(parent) / name
    shutil.copytree(TEMPLATE, task)
    if test_sh is not None:
        (task / "verifier" / "test.sh").write_text(test_sh)
    if test_outputs is not None:
        (task / "verifier" / "test_outputs.py").write_text(test_outputs)
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


def _template_with(line):
    """The template's test.sh (pytest from the image, the download only a fallback) with one line added before its
    checks run, where the round-11 funnel put its probes' downloads."""
    text = (TEMPLATE / "verifier" / "test.sh").read_text()
    anchor = 'mkdir -p "$(dirname "$REWARD_TEXT")"'
    assert anchor in text
    return text.replace(anchor, line + "\n" + anchor, 1)


# The round-11 funnel's probes (F11-03): the template's test.sh and Dockerfile, plus one line that fetches the file the
# checks compare with. Offline, the reference solution scored 0 (FileNotFoundError: /tmp/expected.json), and no check said so.
FUNNEL_PROBES = {
    "p7-wget-in-test": "wget -q -O /tmp/expected.json https://example.com/expected.json",
    "p8-curl-data": "curl -fsSL -o /tmp/expected.json https://example.com/expected.json",
    "p9-git-clone": "git clone -q https://github.com/example/answers /tmp/answers",
    "p10-hf-download": "hf download xdotli/funnel-r11-tasks expected.json --repo-type dataset --local-dir /tmp",
    "p11-curl-expected": "# keep the answers out of the image: fetch them when the verifier runs\n"
                         "curl -fsSL -o /tmp/expected.json https://huggingface.co/datasets/xdotli/funnel-r11-tasks/resolve/main/expected.json",
}
# p4-python-download: the template's test.sh, and a test_outputs.py that reads the expected rows from the Hub.
PYTHON_PROBE = '''"""Checks /root/answer.json for the 5xx-per-endpoint count."""
import json
import urllib.request
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))
ANSWER_FILE = WORKSPACE / "answer.json"
EXPECTED_URL = "https://huggingface.co/datasets/xdotli/funnel-r11-tasks/resolve/main/expected.json"
EXPECTED = json.load(urllib.request.urlopen(EXPECTED_URL))


def test_counts():
    data = json.load(open(ANSWER_FILE))
    assert {r["endpoint"]: r["count"] for r in data} == {r["endpoint"]: r["count"] for r in EXPECTED}
'''


def test_a_verifier_that_fetches_data_warns_whatever_the_pytest_fallback():
    for name, line in FUNNEL_PROBES.items():
        found = _warnings(test_sh=_template_with(line))
        assert len(found) == 1, (name, found)
        command = line.splitlines()[-1]
        assert found[0].startswith("verifier/test.sh fetches files from the network when it runs ("), (name, found)
        assert (command if len(command) <= 90 else command[:89] + "…") in found[0], (name, found)
        assert "allow_internet: false" in found[0] and "so even the reference solution can't pass" in found[0]
        assert "data under verifier/" in found[0] and "after the agent finishes" in found[0]


def test_verifier_python_that_fetches_data_warns():
    found = _warnings(test_outputs=PYTHON_PROBE)
    assert len(found) == 1, found
    assert found[0].startswith("verifier/test_outputs.py fetches files from the network when it runs "
                               "(urllib.request.urlopen from huggingface.co)"), found
    for source, what in [
        ('import requests\n\ndef test_x():\n    assert requests.get("https://example.com/expected.json").json() == 1\n',
         "requests.get from example.com"),
        ('from urllib.request import urlopen as fetch\nURL = "https://data.example.org/rows.csv"\nROWS = fetch(URL).read()\n',
         "urllib.request.urlopen from data.example.org"),
        ('import httpx\nBASE = "https://api.example.com"\n\ndef test_x():\n    assert httpx.get(BASE + "/truth").status_code == 200\n',
         "httpx.get from api.example.com"),
        ('import requests\nS = requests.Session()\n\ndef test_x():\n    assert S.get("https://example.com/a").ok\n',
         "requests.Session from example.com"),
        ('from huggingface_hub import hf_hub_download\nPATH = hf_hub_download("owner/answers", "expected.json", repo_type="dataset")\n',
         "huggingface_hub.hf_hub_download from the Hugging Face Hub"),
        ('import pandas as pd\nTRUTH = pd.read_csv("https://example.com/truth.csv")\n', "pandas.read_csv from example.com"),
    ]:
        found = _warnings(test_outputs=source)
        assert len(found) == 1 and f"({what})" in found[0], (what, found)


def test_other_ways_a_test_sh_fetches_data():
    for line in [
        'EXPECTED="$(curl -fsSL https://example.com/expected.json)"',
        "curl -fsSL https://example.com/data.tar.gz | tar -xz -C /tmp",
        "curl -s https://example.com/expected.json > /tmp/expected.json",
        "curl -sO https://example.com/expected.json",
        "wget https://example.com/expected.json",
        'DATA_URL="https://example.com/expected.json"\ncurl -L -o /tmp/expected.json "$DATA_URL"',
        "huggingface-cli download owner/answers expected.json --repo-type dataset",
        "python3 -c \"import urllib.request; urllib.request.urlretrieve('https://example.com/e.json', '/tmp/e.json')\"",
    ]:
        found = _warnings(test_sh=_template_with(line))
        assert len(found) == 1 and "fetches files from the network" in found[0], (line, found)


def test_what_is_not_a_data_fetch():
    # the network the sandbox has: its own services; a local repository; output thrown away; the template's installer
    for line in [
        "curl -sf --max-time 10 http://localhost:8000/health >/dev/null 2>&1",
        'BASE="${DRIVE_URL:-http://localhost:9003}"\ncurl -s "$BASE/_admin/state" > /tmp/final_state.json',
        "curl -s http://app:8080/api/items -o /tmp/items.json",
        "git clone /app/repo /tmp/check",
        'git clone "$WORKSPACE/repo" /tmp/check',
        "curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh",
        "curl -fL https://github.com/coursier/coursier/releases/download/v2.1.25-M23/cs-x86_64-pc-linux.gz | gzip -d > cs",
        "echo 'see https://example.com/docs for the format'",
    ]:
        assert _warnings(test_sh=_template_with(line)) == [], line
    for source in [
        'import requests\n\ndef test_api():\n    assert requests.get("http://localhost:8000/api").status_code == 200\n',
        '"""Expected values come from https://example.com/spec (copied into data/)."""\nimport json\n\n'
        'def test_x():\n    assert json.load(open("/root/answer.json")) == {"a": 1}\n',
        'FORBIDDEN = {"socket", "urllib", "requests"}\nURL = "https://example.com/"\n\n'
        'def test_x():\n    assert URL not in open("/root/answer.txt").read()\n',
        'import pandas as pd\n\ndef test_x():\n    assert len(pd.read_csv("/root/out.csv")) == 3\n',
    ]:
        assert _warnings(test_outputs=source) == [], source


def test_a_data_fetch_with_network_never_warns():
    for line in FUNNEL_PROBES.values():
        assert _warnings(test_sh=_template_with(line), internet="allow_internet: true") == []
    assert _warnings(test_outputs=PYTHON_PROBE, internet="allow_internet: true") == []


def test_tools_and_data_warn_separately():
    # the old template's downloads (no fallback), and a data fetch: the tools warning doesn't repeat the data fetch
    found = _warnings(test_sh=OLD_TEST_SH.replace("#!/bin/bash\n", "#!/bin/bash\nwget -q -O /tmp/e.json https://example.com/e.json\n"),
                      dockerfile=OLD_DOCKERFILE)
    assert len(found) == 2, found
    assert "downloads when it runs (apt-get install, curl, uvx)" in found[0]
    assert "(wget -q -O /tmp/e.json https://example.com/e.json)" in found[1]


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

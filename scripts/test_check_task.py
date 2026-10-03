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


def _task(parent, name="my-task", test_sh=None, dockerfile=None, internet=None, block="environment", test_outputs=None, extra=None):
    """A copy of the template, with its verifier, Dockerfile or network setting replaced (and `extra` files added)."""
    task = Path(parent) / name
    shutil.copytree(TEMPLATE, task)
    for rel, content in (extra or {}).items():
        (task / rel).parent.mkdir(parents=True, exist_ok=True)
        (task / rel).write_text(content)
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
        "echo 'see https://example.com/docs for the format'",
    ]:
        assert _warnings(test_sh=_template_with(line)) == [], line
    # an installer or a release binary isn't data; outside the pytest fallback it is a tool download (F12-02)
    for line in [
        "curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh",
        "curl -fL https://github.com/coursier/coursier/releases/download/v2.1.25-M23/cs-x86_64-pc-linux.gz | gzip -d > cs",
    ]:
        found = _warnings(test_sh=_template_with(line))
        assert len(found) == 1 and found[0].startswith("verifier/test.sh downloads when it runs (curl), but"), (line, found)
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


# The round-12 funnel's probes (F12-02): the template's test.sh and Dockerfile with one more network use each. The checks
# missed a verifier that installs a tool when it runs outside the pytest fallback, sources a helper that downloads, runs
# curl from its Python, or uses datasets, aws s3 or gsutil. Four of them scored 1.0 online and 0.0 with the network denied,
# while this script said "structure valid" (i1 subprocess curl, i2 os.system curl, i3 a sourced helper, i4 a pip install).
R12_DATA_LINES = {
    "p11-sh-aws-s3": "aws s3 cp s3://example-bucket/expected.json /tmp/expected.json",
    "p12-sh-gsutil": "gsutil cp gs://example-bucket/expected.json /tmp/expected.json",
    "gcloud-storage": "gcloud storage cp gs://example-bucket/expected.json /tmp/expected.json",
    "aws-with-options": "aws --region us-east-1 s3 sync s3://example-bucket/expected /tmp/expected",
}
R12_TOOL_LINES = {
    "p17-sh-pip-install-tool": ("pip3 install --break-system-packages requests==2.32.3", "pip3 install"),
    "p18-sh-apt-tool": ("apt-get update && apt-get install -y jq", "apt-get install"),
    "p19-sh-uvx-tool": ("uvx --from jq-cli jq --version", "uvx"),
    "p20-sh-npx-tool": ("npx -y json@11.0.0 --version", "npx"),
    "i4-sh-pip-install-dep": ('python3 -m pip install -q --target "${BENCHFLOW_WORKSPACE:-/root}/.vdeps" tabulate==0.9.0', "pip install"),
    "apk": ("apk add --no-cache jq", "apk add"),
    "uv-pip": ("uv pip install --system tabulate==0.9.0", "pip install"),
}
FETCH_SH = "#!/bin/bash\ncurl -fsSL -o /tmp/expected.json https://example.com/expected.json\n"


def test_cloud_storage_copies_are_data_fetches():
    for name, line in R12_DATA_LINES.items():
        found = _warnings(test_sh=_template_with(line))
        assert len(found) == 1 and found[0].startswith(f"verifier/test.sh fetches files from the network when it runs ({line})"), (name, found)


def test_a_tool_install_outside_the_fallback_warns():
    for name, (line, label) in R12_TOOL_LINES.items():
        found = _warnings(test_sh=_template_with(line))
        assert len(found) == 1, (name, found)
        assert found[0].startswith(f"verifier/test.sh downloads when it runs ({label}), but task.md sets allow_internet: false"), (name, found)
        assert "doesn't install pytest" not in found[0]   # the download isn't the fallback's, whose image has pytest


def _dockerfile_with(line):
    return (TEMPLATE / "environment" / "Dockerfile").read_text() + "\n" + line + "\n"


def test_an_install_of_what_the_image_has_passes():
    for line, image in [
        ("pip3 install --break-system-packages requests==2.32.3", "RUN pip3 install --break-system-packages requests==2.32.3"),
        ("pip3 install --break-system-packages requests", "RUN apt-get update && apt-get install -y python3-requests"),
        ("apt-get update && apt-get install -y jq", "RUN apt-get update && apt-get install -y --no-install-recommends jq curl"),
        ("uvx --from jq-cli jq --version", "RUN uv tool install jq-cli"),
        ("npx -y json@11.0.0 --version", "RUN npm install -g json@11.0.0"),
        ("npx jest --ci", "RUN npm ci --include=dev"),
        ("apk add --no-cache jq", "RUN apk add --no-cache jq"),
    ]:
        assert _warnings(test_sh=_template_with(line), dockerfile=_dockerfile_with(image)) == [], line
    # an option that installs whatever the image holds still downloads
    for line, image in [
        ("pip3 install --upgrade requests", "RUN pip3 install requests"),
        ('python3 -m pip install -q --target /tmp/deps tabulate==0.9.0', "RUN pip3 install tabulate==0.9.0"),
        ("pip3 install -r /verifier/requirements.txt", "RUN pip3 install requests"),
        ("uvx --isolated --from jq-cli jq --version", "RUN uv tool install jq-cli"),
    ]:
        assert len(_warnings(test_sh=_template_with(line), dockerfile=_dockerfile_with(image))) == 1, line


def test_what_is_not_a_tool_download():
    for line in [
        "pip install --no-deps --no-build-isolation /verifier/checker",   # a local package without its dependencies
        "python3 -m pip install --no-index --find-links /verifier/wheels checker",
        "command -v npx >/dev/null 2>&1 || echo 'no npx'",   # looks a tool up without running it
        "require_cmd() { command -v \"$1\" >/dev/null; }; require_cmd uvx",
        "curl --version",
        "curl -sf http://localhost:8080/health || true",
    ]:
        assert _warnings(test_sh=_template_with(line)) == [], line


def test_the_pytest_fallback_in_its_other_forms():
    run = 'pytest /verifier/test_outputs.py -rA -v'
    forms = [
        f"#!/bin/bash\nif ! command -v pytest >/dev/null 2>&1; then\n  pip install pytest==8.4.1\nfi\n{run}\n",
        f"#!/bin/bash\ncommand -v pytest >/dev/null 2>&1 || pip install pytest==8.4.1\n{run}\n",
        f"#!/bin/bash\ncommand -v pytest >/dev/null 2>&1 || {{\n  apt-get update\n  apt-get install -y python3-pytest\n}}\n{run}\n",
        f"#!/bin/bash\nif command -v pytest >/dev/null; then {run}; else uvx --with pytest==8.4.1 pytest /verifier/test_outputs.py; fi\n",
        f"#!/bin/bash\nif which pytest; then\n  {run}\nelif [ -x /usr/bin/python3 ]; then\n  pip install pytest\n  {run}\nelse\n  exit 1\nfi\n",
    ]
    for text in forms:
        assert _warnings(test_sh=text) == [], text
        found = _warnings(test_sh=text, dockerfile=OLD_DOCKERFILE)   # an image without pytest: the fallback runs
        assert len(found) == 1 and "environment/Dockerfile doesn't install pytest" in found[0], (text, found)


def test_a_heredoc_does_not_move_the_fallback():
    # Python's if/else in a heredoc aren't the shell's: the template's fallback stays excused, the install after it warns
    heredoc = "python3 - <<'PY'\nif True:\n    x = 1\nelse:\n    x = 2\nPY"
    text = _template_with(heredoc).replace("PYTEST_EXIT_CODE=$?", "PYTEST_EXIT_CODE=$?\npip3 install --break-system-packages requests", 1)
    found = _warnings(test_sh=text)
    assert len(found) == 1 and found[0].startswith("verifier/test.sh downloads when it runs (pip3 install), but"), found


def test_a_script_test_sh_sources_or_runs():
    for line in ['source "$VERIFIER_DIR/fetch.sh"', ". /verifier/fetch.sh", 'bash "$VERIFIER_DIR/fetch.sh"', '"$VERIFIER_DIR/fetch.sh"']:
        found = _warnings(test_sh=_template_with(line), extra={"verifier/fetch.sh": FETCH_SH})
        assert len(found) == 1, (line, found)
        assert found[0].startswith("verifier/fetch.sh, which verifier/test.sh runs, fetches files from the network when it runs "
                                   "(curl -fsSL -o /tmp/expected.json https://example.com/expected.json)"), (line, found)
    setup = "#!/bin/bash\npip3 install --break-system-packages tabulate==0.9.0\n"
    found = _warnings(test_sh=_template_with('bash "$VERIFIER_DIR/setup.sh"'), extra={"verifier/setup.sh": setup})
    assert len(found) == 1 and "(pip3 install in verifier/setup.sh)" in found[0], found
    # a helper test.sh never runs, or one only its pytest fallback runs (the image has pytest), isn't followed
    assert _warnings(extra={"verifier/fetch.sh": FETCH_SH}) == []
    fallback = (TEMPLATE / "verifier" / "test.sh").read_text().replace("  apt-get update\n", '  apt-get update\n  source "$VERIFIER_DIR/setup.sh"\n', 1)
    assert _warnings(test_sh=fallback, extra={"verifier/setup.sh": setup}) == []


R12_PYTHON = {
    "p37-py-datasets-load": ('from datasets import load_dataset\nEXPECTED = load_dataset("example/expected", split="train")[0]\n',
                             "datasets.load_dataset from the Hugging Face Hub"),
    "p38-py-subprocess-curl": ('import subprocess\nsubprocess.run(["curl", "-fsSL", "-o", "/tmp/expected.json", "https://example.com/expected.json"], check=True)\n',
                               "subprocess.run: curl -fsSL -o /tmp/expected.json https://example.com/expected.json"),
    "p39-py-os-system-wget": ('import os\nos.system("wget -q -O /tmp/expected.json https://example.com/expected.json")\n',
                              "os.system: wget -q -O /tmp/expected.json https://example.com/expected.json"),
    "p43-py-numpy-loadtxt-url": ('import numpy as np\nEXPECTED_TOTAL = float(np.loadtxt("https://example.com/total.txt"))\n',
                                 "numpy.loadtxt from example.com"),
    "p44-py-url-built-from-parts": ('import json\nimport urllib.request\nHOST = "example.com"\n'
                                    'EXPECTED = json.load(urllib.request.urlopen("https://" + HOST + "/expected.json"))\n',
                                    "urllib.request.urlopen from example.com"),
    "i1-py-subprocess-curl": ('import os\nimport subprocess\nfrom pathlib import Path\nWORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))\n'
                              'subprocess.run(["curl", "-fsSL", "-o", str(WORKSPACE / ".meta.json"), "https://huggingface.co/api/datasets/benchflow/tmax-dogfood-64"], check=True)\n',
                              "subprocess.run: curl -fsSL -o /root/.meta.json https://huggingface.co/api/datasets/benchflow/tmax-dogfood"),
    "i2-py-os-system-curl": ('import os\nfrom pathlib import Path\nWORKSPACE = Path(os.environ.get("BENCHFLOW_WORKSPACE", "/root"))\n'
                             'os.system("curl -fsSL -o " + str(WORKSPACE / ".meta.json") + " https://huggingface.co/api/datasets/benchflow/tmax-dogfood-64")\n',
                             "os.system: curl -fsSL -o /root/.meta.json https://huggingface.co/api/datasets/benchflow/tmax-dogfood"),
    "pandas-s3": ('import pandas as pd\nTRUTH = pd.read_parquet("s3://example-bucket/truth.parquet")\n', "pandas.read_parquet from s3://example-bucket"),
    "datasets-remote-files": ('import datasets\nROWS = datasets.load_dataset("json", data_files={"test": "https://example.com/rows.jsonl"})\n',
                              "datasets.load_dataset from the Hugging Face Hub"),
    "subprocess-pip": ('import subprocess, sys\nsubprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "tabulate"])\n',
                       "subprocess.check_call: pip install"),
}


def test_verifier_python_that_downloads_warns():
    for name, (source, what) in R12_PYTHON.items():
        found = _warnings(test_outputs=source)
        assert len(found) == 1 and found[0].startswith("verifier/test_outputs.py fetches files from the network when it runs ("), (name, found)
        assert what in found[0], (name, found)


def test_verifier_python_that_needs_no_network():
    for source in [
        'from pathlib import Path\nfrom datasets import load_dataset\nROWS = load_dataset("json", data_files=str(Path(__file__).parent / "rows.jsonl"))\n',
        'from pathlib import Path\nimport datasets\nROWS = datasets.load_dataset(str(Path(__file__).parent / "data"))\n',
        'import datasets\nROWS = datasets.load_from_disk("/verifier/data")\n',
        'import subprocess\nsubprocess.run(["pytest", "-q", "/verifier/checks.py"], check=True)\n',
        'import os\nos.system("sort /root/out.txt > /tmp/sorted.txt")\n',
        'import numpy as np\nTOTAL = np.loadtxt("/verifier/total.txt")\n',
        'import requests\nPORT = 8000\nOK = requests.get(f"http://localhost:{PORT}/health").ok\n',
    ]:
        assert _warnings(test_outputs=source) == [], source
    # a command that installs what the image has
    source = 'import subprocess\nsubprocess.run(["pip3", "install", "--break-system-packages", "requests==2.32.3"])\n'
    assert _warnings(test_outputs=source, dockerfile=_dockerfile_with("RUN pip3 install --break-system-packages requests==2.32.3")) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
    print(f"{len(tests)} passed")


def test_credit_needs_a_hub_handle_or_a_legacy_author_name():
    # Space PR #4 (hub identities): author_hub replaces author_name and author_email; a legacy author_name still counts.
    with tempfile.TemporaryDirectory() as tmp:
        task = _task(tmp)
        md = task / "task.md"
        template = md.read_text()
        assert "  author_hub: " in template and "author_email" not in template
        assert C.check_task(task) == []
        md.write_text(template.replace("  author_hub: your-hf-username", "  author_name: Your Name"))
        assert C.check_task(task) == []
        md.write_text(template.replace("  author_hub: your-hf-username", "  author_email: you@example.com"))
        assert C.check_task(task) == ["task.md metadata.author_hub required (legacy author_name is also accepted)"]

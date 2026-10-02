#!/usr/bin/env python3
"""
Minimal task.md structural check — runs in CI without depending on the
benchflow CLI (which is in flux: the multi-level dogfood that knows the
task.md format lives on an unreleased upstream branch).

Validates:
- task.md exists and has YAML frontmatter (--- ... ---) + a Markdown body
- Required frontmatter fields are present
- environment/Dockerfile exists and starts with `FROM `
- verifier/{test.sh, test_outputs.py, verifier.md} all exist
- verifier/rubrics/ contains at least one *.md file
- oracle/solve.sh exists

This is the same shape `bench tasks check --level publication-grade`
enforces locally; remove this script and swap CI to the benchflow CLI
once upstream lands the new format on main.

It checks structure only: that the files, keys and prompt section exist,
not their contents. The untouched template passes, placeholders and all.
scripts/run_local.sh replays the oracle and an empty trial, and the Arena's
validate checks the metadata values (category, license, origin).

One content check warns without failing: a verifier that needs the network
while task.md sets allow_internet: false. In a sandbox without network, which
is how scripts/run_local.sh and the Arena's oracle and no-op controls run a
task, a download fails or hangs, so even the reference solution can't pass.
It warns about two kinds:
- verifier/test.sh downloads tools when it runs (apt-get, apk, pip or uv
  installs, uvx, npx, an installer fetched with curl). An install of packages
  the Dockerfile installs too needs no network, and neither does a download in
  the template's fallback for an image without pytest (the else branch of its
  `command -v "$PYTEST_BIN"` check) when the Dockerfile installs pytest.
- The verifier fetches data when it runs: in test.sh, curl or wget into a file
  or a pipe, git clone, hf download, aws s3, gsutil or gcloud storage copies;
  in its Python (test_outputs.py and the other verifier/*.py), urllib, requests,
  httpx, huggingface_hub, datasets.load_dataset, and pandas or numpy reading a
  remote address, also as a URL built from parts, and commands it runs with
  subprocess or os.system. Nothing in the image stands in for that data, so
  this warns whatever the pytest fallback.
Both follow a script under verifier/ that test.sh sources or runs (one level).

Exit code: 0 if every task validates, 1 if any task has issues.
"""
from __future__ import annotations

import ast
import re
import shlex
import sys
import warnings
from pathlib import Path
from typing import Iterable

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
REQUIRED_FRONTMATTER = (
    "version",
    "metadata",
    "agent",
    "verifier",
    "environment",
)
REQUIRED_METADATA = (
    "category",
)

# What verifier/test.sh downloads with when it runs: package installs, curl or wget, uvx and the like.
DOWNLOADS = re.compile(
    r"\b(?P<install>(?:apt-get|apt|apk|yum|dnf|pip3?|pipx|uv|conda|mamba|npm|pnpm|yarn|gem|cargo|go)\s+(?:-\S+\s+)*"
    r"(?:install|add|get|sync|tool))\b"
    r"|\b(?P<fetch>curl|wget|uvx|pipx\s+run|npx|git\s+clone)\b"
)
# An install needs no network when the image has what it names (F12-02): the installer's family, the options that take
# a value, and the options that install whatever the image holds.
TOOL_FAMILY = {"pip": "pip", "pip3": "pip", "pipx": "pipx", "uvx": "uv-tool", "apt-get": "apt", "apt": "apt", "apk": "apk",
               "yum": "rpm", "dnf": "rpm", "conda": "conda", "mamba": "conda", "npm": "npm", "pnpm": "npm", "yarn": "npm",
               "npx": "npm", "gem": "gem", "cargo": "cargo", "go": "go"}
VALUE_OPTS = {
    "pip": {"-r", "--requirement", "-c", "--constraint", "-e", "--editable", "-t", "--target", "--prefix", "--root", "-i",
            "--index-url", "--extra-index-url", "-f", "--find-links", "--src", "--platform", "--python-version",
            "--implementation", "--abi", "--upgrade-strategy", "--progress-bar", "--log", "--proxy", "--retries", "--timeout",
            "--exists-action", "--trusted-host", "--cert", "--client-cert", "--cache-dir", "--global-option", "-C",
            "--config-settings", "--no-binary", "--only-binary", "--report", "--python", "--group"},
    "apt": {"-o", "--option", "-c", "--config-file", "-t", "--target-release", "--default-release"},
    "apk": {"-X", "--repository", "-p", "--root", "--arch", "-t", "--virtual", "--repositories-file", "--cache-dir"},
    "npm": {"--prefix", "--registry", "--cache", "-w", "--workspace", "--userconfig", "-p", "--package", "-c", "--call"},
    "uv-tool": {"--from", "--with", "--with-editable", "--with-requirements", "-p", "--python", "--index", "--index-url",
                "--default-index", "--extra-index-url", "-f", "--find-links", "--cache-dir", "--config-file", "--directory",
                "--project", "-w"},
    "pipx": {"--spec", "--python", "--index-url", "-i", "--pip-args"},
    "conda": {"-c", "--channel", "-n", "--name", "-p", "--prefix", "--file"},
}
ANYWAY = {"-r", "--requirement", "-e", "--editable", "-t", "--target", "--prefix", "--root", "-U", "--upgrade",
          "--force-reinstall", "-I", "--ignore-installed", "--reinstall", "--refresh", "--isolated", "--update-cache",
          "--with-requirements", "--with-editable", "--file"}
RUNNER_SOURCE = {"--from", "--spec", "-p", "--package"}   # where uvx, pipx run and npx take the tool from
# The statements of test.sh that run only when the image has no pytest: the branch after its `command -v pytest` check.
KEYWORD = re.compile(r"(if|elif|then|else|fi|do|done|\{|\})(?=\s|$)\s*")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
# A copy to or from cloud storage, which needs the network whatever it copies.
CLOUD_COPY = re.compile(
    r"(?<![\w./$-])(?:aws(?:\s+--?[\w-]+(?:[= ][^\s-]\S*)?)*\s+s3(?:\s+(?:cp|sync|mv|ls)|api\s+[\w-]+)"
    r"|gsutil(?:\s+-\S+)*\s+(?:cp|rsync|mv|cat|ls)"
    r"|gcloud(?:\s+--?[\w-]+(?:=\S+)?)*\s+storage\s+(?:cp|rsync|mv|cat|ls|objects))(?=\s|$)"
)
# A script under verifier/ that test.sh sources or runs: `source X`, `. X`, `bash X`.
SHELL_RUNNERS = {"source", ".", "bash", "sh", "zsh", "dash", "ksh"}
# Words after which a tool's name is the command that runs (sudo pip, python3 -m pip, uv pip), not an argument
# (`command -v npx` or `require_cmd uvx` look a tool up without running it).
RUN_PREFIXES = {"sudo", "env", "exec", "time", "nohup", "then", "do", "else", "timeout", "xargs", "nice", "stdbuf", "uv", "-m", "!"}
# A test.sh that runs pytest from the image when the image has it (the template's `command -v "$PYTEST_BIN"`).
PYTEST_IN_IMAGE = re.compile(r"\b(?:command\s+-v|which|type\s+-[pP]|hash)\s+[\"']?(?:\$\{?PYTEST_BIN\}?|pytest)(?![\w-])")
OFFLINE = {"false", "no", "off", "0"}

# What a verifier fetches as data when it runs (F11-03): the file it grades with, which no image line stands in for.
FETCHER = re.compile(r"(?<![\w./$-])(curl|wget)(?=\s)")
HF_DOWNLOAD = re.compile(r"(?<![\w./$-])(hf|huggingface-cli)\s+download\b")
GIT_CLONE = re.compile(r"(?<![\w./$-])git\s+(?:-[cC]\s+\S+\s+)*clone(?=\s)")
URL_HOST = re.compile(r"(?:https?|ftp)://([^/\s:'\"`$?#]+)[^\s'\"`<>|;)]*")   # an address; group 1 its host
STDOUT_TO = re.compile(r"(?:^|\s)[1&]?>>?\|?\s*([^\s;&|<>]+)")
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
# Where installers and package indexes live: a fetch from one of these is a tool download (the first warning), not data.
TOOL_HOSTS = {
    "astral.sh", "pypi.org", "files.pythonhosted.org", "bootstrap.pypa.io", "download.pytorch.org", "deb.debian.org",
    "security.debian.org", "archive.ubuntu.com", "security.ubuntu.com", "dl-cdn.alpinelinux.org", "registry.npmjs.org",
    "nodejs.org", "deb.nodesource.com", "crates.io", "static.rust-lang.org", "sh.rustup.rs", "go.dev", "dl.google.com",
    "proxy.golang.org", "repo.anaconda.com", "conda.anaconda.org", "get.docker.com", "download.docker.com",
    "cloud.r-project.org", "cran.r-project.org", "opam.ocaml.org", "coq.inria.fr",
}
# curl's and wget's short options that take a value (in a cluster like -fsSLo FILE, the value follows the letter).
CURL_VALUE_OPTS, WGET_VALUE_OPTS = set("AbcCdDeEFHKmoPQrtTuUwxXyYz"), set("aABDeiIloOPQRtTUwX")
GIT_CLONE_VALUE_OPTS = {"-b", "--branch", "-o", "--origin", "-u", "--upload-pack", "-c", "--config", "--depth", "-j",
                        "--jobs", "--reference", "--reference-if-able", "--separate-git-dir", "--template", "--filter",
                        "--shallow-since", "--shallow-exclude", "--server-option", "--bundle-uri", "--ref-format"}
# Python that fetches over HTTP when it runs: module -> functions and clients. huggingface_hub's downloads need no URL.
PY_FETCH = {
    "urllib.request": {"urlopen", "urlretrieve"},
    "requests": {"get", "post", "put", "patch", "delete", "head", "options", "request", "Session", "session"},
    "httpx": {"get", "post", "put", "patch", "delete", "head", "options", "request", "stream", "Client", "AsyncClient"},
    "urllib3": {"request", "PoolManager"},
    "aiohttp": {"request", "ClientSession"},
    "http.client": {"HTTPConnection", "HTTPSConnection"},
}
PY_HUB = {"huggingface_hub": {"hf_hub_download", "snapshot_download"}}
# Readers that take a remote address as a path: pandas' read_*, numpy's text loaders (F12-02).
PY_READERS = {"numpy": {"loadtxt", "genfromtxt"}}
# Commands the verifier's Python runs: what they fetch is checked as test.sh's is.
PY_SHELL_OUTS = {"subprocess": {"run", "call", "check_call", "check_output", "Popen", "getoutput", "getstatusoutput"},
                 "os": {"system", "popen"}}
# datasets.load_dataset's builders that read the files they're given (local unless an address says otherwise).
DATASET_BUILDERS = {"json", "csv", "parquet", "text", "arrow", "pandas", "imagefolder", "audiofolder", "videofolder",
                    "webdataset", "xml", "sql", "generator"}
REMOTE_FS = re.compile(r"\b((?:s3a?|gs|gcs|hf|az|abfss?)://[^/\s'\"`]+)")   # object stores and the Hub, as readers take them
PY_FETCH_TEXT = re.compile(
    r"\b(urlopen|urlretrieve|hf_hub_download|snapshot_download|requests\.(?:get|post|request|Session)|"
    r"httpx\.(?:get|post|request|stream|Client|AsyncClient)|HTTPS?Connection|loadtxt|genfromtxt|"
    r"read_(?:csv|json|parquet|table|excel|feather|xml|html|fwf|orc))\s*\("
)
LOAD_DATASET_TEXT = re.compile(r"\bload_dataset\(\s*(['\"])([^'\"]+)\1")


def parse_yaml_keys(block: str) -> set[str]:
    """Cheap top-level key extractor — avoids importing PyYAML in CI."""
    keys: set[str] = set()
    for line in block.splitlines():
        if not line or line.startswith(("#", " ", "\t", "-")):
            continue
        if ":" not in line:
            continue
        key = line.split(":", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def parse_metadata_keys(block: str) -> set[str]:
    """Extract keys nested under `metadata:` (2-space indent expected)."""
    keys: set[str] = set()
    in_metadata = False
    for line in block.splitlines():
        stripped = line.lstrip()
        if line.startswith("metadata:"):
            in_metadata = True
            continue
        if in_metadata:
            if line and not line.startswith((" ", "\t")):
                in_metadata = False
                continue
            if stripped.startswith("#") or not stripped:
                continue
            if ":" in stripped:
                keys.add(stripped.split(":", 1)[0].strip())
    return keys


def block_values(frontmatter: str, block: str) -> dict[str, str]:
    """Scalar keys directly under a top-level block (`environment:`, `sandbox:`), comments dropped."""
    values: dict[str, str] = {}
    inside, indent = False, None
    for line in frontmatter.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            key, _, rest = line.partition(":")
            inside, indent = key.strip() == block and not rest.strip(), None
            continue
        if not inside or ":" not in line:
            continue
        depth = len(line) - len(line.lstrip())
        indent = depth if indent is None else indent
        if depth == indent:
            key, _, raw = line.strip().partition(":")
            values[key.strip()] = re.split(r"\s+#", raw, maxsplit=1)[0].strip().strip("\"'")
    return values


def offline(frontmatter: str) -> bool:
    """True when task.md turns the sandbox's network off: allow_internet: false under environment: (or sandbox:, in
    BenchFlow's own format), or BenchFlow's network_mode: no-network."""
    for block in ("environment", "sandbox"):
        values = block_values(frontmatter, block)
        if values.get("allow_internet", "").lower() in OFFLINE or values.get("network_mode", "").lower() == "no-network":
            return True
    return False


def image_installs_pytest(dockerfile: str) -> bool:
    """A RUN instruction of the Dockerfile installs pytest (pip, uv, apt's python3-pytest ...)."""
    joined = re.sub(r"\\\n", " ", dockerfile)
    return any(
        re.match(r"\s*RUN\b", line, re.I) and re.search(r"\binstall\b", line) and re.search(r"(?<![\w.])pytest(?!\w)", line)
        for line in joined.splitlines()
    )


def remote_host(host: str) -> bool:
    """A host a sandbox without network can't reach: not this machine, a bare service name or a placeholder."""
    host = host.lower().strip("[]")
    return ("." in host and not re.match(r"(localhost|127\.|0\.0\.0\.0)", host) and not host.endswith(".localhost")
            and not re.search(r"[{}<>%]", host))


def tool_address(url) -> bool:
    """An address that serves tools, not data: an installer, a package index, a release binary."""
    return url.group(1).lower() in TOOL_HOSTS or "/releases/download/" in url.group(0)


def shell_lines(text: str) -> list[str]:
    """A shell script's commands: continuation lines joined, comments dropped."""
    lines = []
    for line in re.sub(r"\\\n", " ", text).splitlines():
        code = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].strip()
        if code:
            lines.append(code)
    return lines


def expand(text: str, names: dict[str, str]) -> str:
    """$NAME, ${NAME} and ${NAME:-default} replaced by what the script assigns (or the default), where it can tell."""
    def value(m):
        name = m.group(1) or m.group(3)
        return names.get(name, m.group(2) if m.group(2) is not None else m.group(0))
    for _ in range(3):
        new = re.sub(r"\$\{(\w+)(?::?[-=]([^}]*))?\}|\$(\w+)", value, text)
        if new == text:
            break
        text = new
    return text


def command_at(line: str, start: int) -> tuple[str, str | None]:
    """The command starting at `start`, up to an unquoted ; & | ) or backtick, and the command it pipes into."""
    quote, depth, i = None, 0, start
    while i < len(line):
        ch = line[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "\\":
            i += 1
        elif line.startswith("$(", i):
            depth, i = depth + 1, i + 1
        elif ch == ")" and depth:
            depth -= 1
        elif ch == "&" and (line[i - 1:i] == ">" or line[i + 1:i + 2] == ">"):
            pass   # 2>&1, &>
        elif ch in ";&|)`":
            break
        i += 1
    rest, target = line[i:], None
    if rest.startswith("|") and not rest.startswith("||"):
        words = [w for w in rest[1:].split() if not re.match(r"[\w.-]+=|-", w) and w not in ("sudo", "env", "exec", "command")]
        target = words[0].rsplit("/", 1)[-1] if words else None
    return line[start:i].strip(), target


def words_of(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return command.split()


def fetch_output(tool: str, command: str) -> str | None:
    """Where curl or wget writes what it fetches: a path, "-" for stdout, or None for stdout (curl's default)."""
    out = None if tool == "curl" else "(its file name)"   # wget saves to a file unless told otherwise
    value_opts, output_letter = (CURL_VALUE_OPTS, "o") if tool == "curl" else (WGET_VALUE_OPTS, "O")
    long_output = "--output" if tool == "curl" else "--output-document"
    words, i = words_of(command), 1
    while i < len(words):
        word, following = words[i], words[i + 1] if i + 1 < len(words) else ""
        if word == long_output:
            out, i = following, i + 1
        elif word.startswith(long_output + "="):
            out = word.split("=", 1)[1]
        elif tool == "curl" and word in ("--remote-name", "--remote-name-all"):
            out = "(its file name)"
        elif tool == "wget" and word == "--spider":
            return "/dev/null"
        elif re.match(r"-[A-Za-z]", word):
            for k, letter in enumerate(word[1:]):
                if tool == "curl" and letter == "O":
                    out = "(its file name)"
                elif letter in value_opts:
                    if letter == output_letter:
                        out = word[k + 2:] or following
                    i += 0 if word[k + 2:] else 1
                    break
        i += 1
    redirect = STDOUT_TO.search(command)   # > FILE, >> FILE, 1> FILE (2> is stderr)
    return redirect.group(1) if redirect and out in (None, "-", "/dev/stdout") else out


def assignments(lines: list[str]) -> dict[str, str]:
    """What a shell script assigns to its variables (NAME=value, export NAME=value), the last value of each."""
    names = {}
    for line in lines:
        m = re.match(r"(?:export\s+|local\s+|readonly\s+)?([A-Za-z_]\w*)=(.*)$", line)
        if m:
            names[m.group(1)] = (words_of(m.group(2)) or [""])[0]
    return names


def shell_fetches(text: str) -> tuple[list[str], list[str]]:
    """The data a shell script fetches when it runs (each command, shortened) and its lines without those commands.

    Data: curl or wget into a file, a variable or a pipe (not into a shell: that runs an installer), unless every address
    is in the sandbox or an installer's, a package index's or a release binary; git clone of a repository that isn't a
    local path; hf download; a copy to or from cloud storage (aws s3, gsutil, gcloud storage)."""
    lines = shell_lines(text)
    names = assignments(lines)
    fetched, rest = [], []
    for line in lines:
        spans = []
        for m in FETCHER.finditer(line):
            command, target = command_at(line, m.start())
            out = fetch_output(m.group(1), command)
            substituted = bool(re.search(r"(\$\(|`)\s*$", line[:m.start()]))
            addresses = list(URL_HOST.finditer(expand(command, names)))
            remote = [u for u in addresses if remote_host(u.group(1))]
            if addresses and not remote:
                continue   # a service in the sandbox (localhost, a bare service name)
            if remote and all(tool_address(u) for u in remote):
                continue   # an installer, a package or a release binary: the tools warning's
            to_file = out not in (None, "-", "/dev/stdout")
            piped = not to_file and target and target not in SHELLS
            if (to_file and out != "/dev/null") or (not to_file and substituted) or piped:
                spans.append((m.start(), m.start() + len(command), command + (f" | {target}" if piped else "")))
        for m in GIT_CLONE.finditer(line):
            command, _ = command_at(line, m.start())
            words, i, source = words_of(expand(command, names)), 0, None
            words = words[next((k for k, w in enumerate(words) if w == "clone"), 0) + 1:]
            while i < len(words) and source is None:
                if words[i] in GIT_CLONE_VALUE_OPTS:
                    i += 1
                elif not words[i].startswith("-"):
                    source = words[i]
                i += 1
            if source and (("://" in source and not source.startswith("file:"))
                           or re.match(r"([\w.+-]+@)?[\w-]+(\.[\w-]+)+:", source)):   # URL or host:path; else a local path
                spans.append((m.start(), m.start() + len(command), command))
        for m in HF_DOWNLOAD.finditer(line):
            command, _ = command_at(line, m.start())
            spans.append((m.start(), m.start() + len(command), command))
        for m in CLOUD_COPY.finditer(line):
            command, _ = command_at(line, m.start())
            spans.append((m.start(), m.start() + len(command), command))
        for start, end, command in sorted(spans):
            short = " ".join(command.split())
            fetched.append(short if len(short) <= 90 else short[:89] + "…")
        for start, end, _ in sorted(spans, reverse=True):
            line = line[:start] + " " + line[end:]
        rest.append(line)
    return list(dict.fromkeys(fetched)), rest


def split_statements(line: str) -> list[str]:
    """A shell line's statements: split at each ; outside quotes and $( ) (a case's ;; is not a split)."""
    out, quote, depth, start, i = [], None, 0, 0, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == quote:
                quote = None
            elif ch == "\\" and quote == '"':
                i += 1
        elif ch in "'\"":
            quote = ch
        elif ch == "\\":
            i += 1
        elif line.startswith("$(", i):
            depth, i = depth + 1, i + 1
        elif ch == ")" and depth:
            depth -= 1
        elif ch == ";" and not depth:
            if line.startswith(";;", i):
                i += 1
            else:
                out.append(line[start:i])
                start = i + 1
        i += 1
    out.append(line[start:])
    return [s.strip() for s in out if s.strip()]


def unquoted(text: str, token: str, start: int = 0) -> int:
    """Where `token` first appears in `text` from `start`, outside quotes, or -1."""
    quote, i = None, 0
    while i < len(text):
        ch = text[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif i >= start and text.startswith(token, i):
            return i
        i += 1
    return -1


def statements_of(lines: list[str]) -> list[tuple[str, bool]]:
    """A shell script's statements, each with whether it runs only when the image has no pytest (F12-02): the else and
    elif branches of an `if` whose condition checks that pytest is there (the template's `command -v "$PYTEST_BIN"`), the
    then branch of one that checks it isn't (`if ! command -v pytest`), and what follows `||` after such a check, a
    `{ ... }` group included. A heredoc's lines stay as they are, in the context around them."""
    out, stack, heredoc = [], [], None

    def fallback() -> bool:
        return any(frame["fb"] for frame in stack if not frame.get("cond"))

    for line in lines:
        if heredoc:
            if line.strip() == heredoc:
                heredoc = None
            else:
                out.append((line, fallback()))
            continue
        m = HEREDOC.search(line)
        heredoc = m.group(2) if m else None
        for stmt in split_statements(line):
            while stmt:
                k = KEYWORD.match(stmt)
                word = k.group(1) if k else None
                if word == "if":
                    stmt = stmt[k.end():]
                    out.append((stmt, fallback()))   # the condition runs where the if is
                    stack.append({"kind": "if", "cond": True, "text": stmt, "fb": False, "after": False})
                    stmt = ""
                elif word == "elif" and stack and stack[-1]["kind"] == "if":
                    frame, stmt = stack[-1], stmt[k.end():]
                    out.append((stmt, fallback() or frame["after"]))
                    frame.update(cond=True, text=stmt)
                    stmt = ""
                elif word == "then" and stack and stack[-1]["kind"] == "if":
                    frame, stmt = stack[-1], stmt[k.end():]
                    check = PYTEST_IN_IMAGE.search(frame["text"])
                    negated = bool(check) and (frame["text"].lstrip().startswith("!")
                                               or frame["text"][:check.start()].rstrip().endswith("!"))
                    if frame["after"]:
                        frame["fb"] = True   # an elif after the check: it runs only when pytest isn't there
                    else:
                        frame["fb"], frame["after"] = negated, bool(check) and not negated
                    frame["cond"] = False
                elif word == "else" and stack and stack[-1]["kind"] == "if":
                    frame, stmt = stack[-1], stmt[k.end():]
                    frame["fb"], frame["cond"] = frame["after"], False
                elif word == "fi" and stack and stack[-1]["kind"] == "if":
                    stack.pop()
                    stmt = stmt[k.end():]
                elif word == "}" and stack and stack[-1]["kind"] == "group":
                    stack.pop()
                    stmt = stmt[k.end():]
                elif word == "{":
                    stack.append({"kind": "group", "fb": False})
                    stmt = stmt[k.end():]
                elif word in ("do", "done"):
                    stmt = stmt[k.end():]
                else:
                    check = PYTEST_IN_IMAGE.search(stmt)
                    cut = unquoted(stmt, "||", check.end()) if check else -1
                    if cut < 0:
                        out.append((stmt, fallback()))
                        break
                    out.append((stmt[:cut].strip(), fallback()))
                    tail = stmt[cut + 2:].strip()
                    if tail.startswith("{") and not tail.endswith("}"):
                        stack.append({"kind": "group", "fb": True})   # command -v pytest || { ... }
                        tail = tail[1:].strip()
                    if tail:
                        out.append((tail, True))
                    break
    return [(s, f) for s, f in out if s]


def package_name(family: str, word: str) -> str | None:
    """A package's name as an install names it (pytest==8.4.1, jq=1.7*, json@11.0.0, @scope/x@2, requests[socks]), or
    None for a URL, a local path, a file or a variable, which the image can't be shown to hold."""
    if "://" in word or word.startswith((".", "/", "~", "$", "git+")) or re.search(r"\.(whl|tar\.gz|tgz|zip|deb|rpm|apk|txt)$", word):
        return None
    if family == "npm":
        name = "@" + word[1:].split("@", 1)[0] if word.startswith("@") else word.split("@", 1)[0]
    else:
        name = re.split(r"[\[<>=!~;@:*\s]", word, maxsplit=1)[0]
    if family in ("pip", "uv-tool", "pipx"):
        name = re.sub(r"[-_.]+", "-", name)
    return name.lower() or None


def tool_use(command: str) -> tuple[str | None, list[str] | None, bool]:
    """What a tool command installs or runs: its installer family, the packages it names (None when it names none the
    image could hold: a requirements file, a URL, `npm install` or `uv sync` alone, curl or wget), and whether an option
    makes it install whatever the image holds (--target, --upgrade, --force-reinstall, --isolated ...)."""
    words = words_of(command)
    cut = next((i for i, w in enumerate(words) if w[:1] in "<>|&;" or re.fullmatch(r"\d*>+&?\d*", w)), len(words))
    words = words[:cut]
    if not words:
        return None, None, False
    head, args, runner = words[0].rsplit("/", 1)[-1], words[1:], False
    subs = [w for w in args if not w.startswith("-")]
    if head == "uv":
        if subs[:1] != ["tool"] or subs[1:2] not in (["install"], ["run"]):
            return "uv", None, False   # uv add, uv sync: a project's dependencies, from its lock file
        family, runner, args = "uv-tool", subs[1] == "run", args[args.index(subs[1]) + 1:]
    elif head in ("uvx", "npx"):
        family, runner = TOOL_FAMILY[head], True
    elif head == "pipx":   # pipx run fetches its app into its own cache whatever pipx installed: nothing stands in for it
        runner = subs[:1] == ["run"]
        family, args = ("pipx-run" if runner else "pipx"), (args[args.index(subs[0]) + 1:] if subs else [])
    elif head in TOOL_FAMILY:
        family = TOOL_FAMILY[head]
        at = next((i for i, w in enumerate(args) if w in ("install", "add", "get", "sync", "tool")), None)
        if at is None:
            return family, None, False
        args = args[at + 1:]
    else:
        return None, None, False   # curl, wget, git clone: what they fetch can't be in the image
    opts, named, sources, anyway, i = VALUE_OPTS.get(family, set()), [], [], False, 0
    while i < len(args):
        word = args[i]
        if word.startswith("-") and word != "-":
            key, _, value = word.partition("=")
            anyway = anyway or key in ANYWAY
            if key in opts and not value:
                value, i = (args[i + 1] if i + 1 < len(args) else ""), i + 1
            if value and key in RUNNER_SOURCE | {"--with"} and runner:
                (sources if key in RUNNER_SOURCE else named).append(value)
        else:
            named.append(word)
            if runner:
                break   # what follows the tool are its own arguments
        i += 1
    if runner and sources:   # uvx --from jq-cli jq: the package is jq-cli, not the command jq
        named = sources + [w for w in named[:-1]]
    flags = {w.split("=", 1)[0] for w in args if w.startswith("-")}
    local = named and all(re.match(r"[./~$]", w) and "://" not in w or w.endswith(".whl") for w in named)
    if family == "pip" and ("--no-index" in flags or local and "--no-deps" in flags
                            and ("--no-build-isolation" in flags or all(w.endswith(".whl") for w in named))):
        return family, [], False   # from local files only: nothing to download
    names = [package_name(family, w) for w in named]
    return family, (names if names and None not in names else None), anyway


def at_command(stmt: str, start: int) -> bool:
    """The word at `start` is the command that runs there (at the start, after && | ; ( ` or a quote, or after sudo, env,
    timeout N, python3 -m ...), not an argument of another command."""
    before = stmt[:start].rstrip()
    if not before or before[-1] in "&|;(`{'\"":
        return True
    words = before.split()
    return (words[-1] in RUN_PREFIXES or bool(re.fullmatch(r"[A-Za-z_]\w*=\S*", words[-1]))
            or len(words) > 1 and words[-2] in ("sudo", "env", "timeout", "nice") and (words[-1].startswith("-") or words[-1][:1].isdigit()))


def image_packages(dockerfile: str) -> dict[str, set[str]]:
    """What environment/Dockerfile's RUN instructions install, by installer family: apt's python3-X counts as pip's X, a
    `uv tool install` is what uvx runs (uvx uses an installed tool unless --isolated), and an npm, pnpm or yarn install of
    a project's dependencies (no package named) lets npx run them ("*")."""
    have: dict[str, set[str]] = {}
    for line in re.sub(r"\\\n", " ", dockerfile).splitlines():
        run = re.match(r"\s*RUN\s+(.*)", line, re.I)
        if run and re.search(r"\b(?:npm|pnpm)\s+(?:ci|clean-install)\b", run.group(1)):
            have.setdefault("npm", set()).add("*")
        for hit in DOWNLOADS.finditer(run.group(1)) if run else ():
            if not hit.group("install") or not at_command(run.group(1), hit.start()):
                continue
            family, names, _ = tool_use(command_at(run.group(1), hit.start())[0])
            if family == "npm" and names is None:
                names = ["*"]
            for name in names or []:
                have.setdefault(family, set()).add(name)
                if family == "apt" and name.startswith("python3-"):
                    have.setdefault("pip", set()).add(name[len("python3-"):])
    return have


def tool_downloads(statements: list[tuple[str, bool]], names: dict[str, str], have: dict[str, set[str]],
                   has_pytest: bool) -> tuple[list[str], bool]:
    """The tools a script's statements download when they run (installer and subcommand, or the fetcher), leaving out an
    install of packages the image installs too, curl or wget of an address in the sandbox, and a download in the pytest
    fallback when the image has pytest; and whether one it names is in that fallback (the image lacks pytest)."""
    found, in_fallback = [], False
    for stmt, fallback in statements:
        if fallback and has_pytest:
            continue   # it runs only when the image has no pytest, and this one has it
        for hit in DOWNLOADS.finditer(stmt):
            if not at_command(stmt, hit.start()):
                continue   # an argument: `command -v uvx`, `require_cmd npx`
            command, _ = command_at(stmt, hit.start())
            words = (hit.group("install") or hit.group("fetch")).split()
            if words[0] == "git":
                continue   # a remote clone is a data fetch (shell_fetches), a local one needs no network
            if words[0] in ("curl", "wget") and not any(remote_host(u.group(1)) for u in URL_HOST.finditer(expand(command, names))):
                continue   # a service in the sandbox, or no address at all (curl --version)
            family, packages, anyway = tool_use(command)
            held = have.get(family, set())
            if packages == [] or packages and not anyway and (words[0] == "npx" and "*" in held or all(p in held for p in packages)):
                continue   # local files only, or the image installs them: the install (or the runner) finds them there
            found.append(f"{words[0]} {words[-1]}" if hit.group("install") else " ".join(words))
            in_fallback = in_fallback or fallback
    return list(dict.fromkeys(found)), in_fallback


def verifier_file(task_dir: Path, path: str, folder: str = "verifier") -> str | None:
    """The file under the verifier's folder that a path in test.sh names ($VERIFIER_DIR/x.sh, /verifier/x.sh, /tests/x.sh,
    ./x.sh, x.sh), or None."""
    for marker in ("/verifier/", "/tests/"):
        if marker in path:
            path = path.rsplit(marker, 1)[1]
            break
    else:
        if path.startswith((f"{folder}/", "verifier/", "tests/")):
            path = path.split("/", 1)[1]
        elif path.startswith("./"):
            path = path[2:]
        elif path.startswith("$") and "/" in path:
            path = path.split("/", 1)[1]   # $(dirname "$0")/x.sh, an unset $DIR/x.sh
        elif "/" in path:
            return None
    if not path or ".." in path.split("/") or path == "test.sh":
        return None
    return f"{folder}/{path}" if (task_dir / folder / path).is_file() else None


def helper_scripts(task_dir: Path, text: str, folder: str = "verifier") -> dict[str, bool]:
    """The scripts under verifier/ that test.sh sources or runs (`source X`, `. X`, `bash X`, or `$VERIFIER_DIR/X.sh`),
    one level down: each with whether test.sh runs it only in its pytest fallback."""
    lines = shell_lines(text)
    names, found = assignments(lines), {}
    for stmt, fallback in statements_of(lines):
        words, i = words_of(expand(stmt, names)), 0
        while i < len(words) and (re.match(r"[A-Za-z_]\w*=", words[i]) or words[i] in ("sudo", "env", "exec", "command", "time")):
            i += 1
        runner = i < len(words) and words[i] in SHELL_RUNNERS
        if runner:
            i += 1
            while i < len(words) and words[i].startswith("-"):
                i = len(words) if words[i] == "-c" else i + 1   # sh -c '...' runs a string, not a script
        if i >= len(words) or not (runner or "/" in words[i]):
            continue
        rel = verifier_file(task_dir, words[i], folder)
        if rel:
            found[rel] = found.get(rel, True) and fallback
    return found


def dotted_name(node, aliases: dict[str, str]) -> str | None:
    """requests.get, urllib.request.urlopen ...: a call's function by the module it was imported from, if it was."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id in aliases:
        return ".".join([aliases[node.id]] + parts[::-1])
    return None


def hosts_in(node, values: dict, depth: int = 0) -> list[str]:
    """The hosts of the addresses an expression is built from: its strings, and what the names in it are assigned."""
    hosts = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            hosts += URL_HOST.findall(sub.value)
        elif isinstance(sub, ast.Name) and sub.id in values and depth < 3:
            hosts += hosts_in(values[sub.id], values, depth + 1)
    return hosts


def text_of(node, values: dict, depth: int = 0) -> str:
    """The text an expression makes, as far as the code shows it: strings, the names assigned them, + and f-strings, str()
    and Path(), os.path.join and pathlib's /, os.environ.get's default; "…" for a part it can't tell."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else "…"
    if isinstance(node, ast.Name):
        return text_of(values[node.id], values, depth + 1) if node.id in values and depth < 3 else "…"
    if isinstance(node, ast.JoinedStr):
        return "".join(text_of(v.value if isinstance(v, ast.FormattedValue) else v, values, depth) for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        return text_of(node.left, values, depth) + ("/" if isinstance(node.op, ast.Div) else "") + text_of(node.right, values, depth)
    if isinstance(node, ast.Call):
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
        if called in ("str", "Path", "PurePath", "PosixPath", "fspath", "abspath", "expanduser") and node.args:
            return text_of(node.args[0], values, depth)
        if called == "join" and isinstance(func, ast.Attribute) and not isinstance(func.value, ast.Constant):
            return "/".join(text_of(a, values, depth) for a in node.args)   # os.path.join
        if called in ("get", "getenv") and len(node.args) > 1:
            return text_of(node.args[1], values, depth)   # os.environ.get(NAME, DEFAULT)
    return "…"


def shell_command(call, values: dict) -> str:
    """The command line a subprocess or os call runs: its list's words, joined, or its string; "" when the code doesn't
    show it."""
    arg = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg in ("args", "cmd", "command")), None)
    if isinstance(arg, ast.Name) and isinstance(values.get(arg.id), (ast.List, ast.Tuple)):
        arg = values[arg.id]
    if isinstance(arg, (ast.List, ast.Tuple)):
        return " ".join(shlex.quote(text_of(e, values)) for e in arg.elts)
    return text_of(arg, values) if arg is not None else ""


def hub_dataset_id(path: str) -> bool:
    """A load_dataset path that names a dataset on the Hub (OWNER/NAME, or a NAME that isn't one of the builders that read
    the files they're given), not a local folder or file."""
    return (bool(re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", path)) and path not in DATASET_BUILDERS and not path.startswith(".")
            and not re.search(r"\.(json|jsonl|csv|tsv|parquet|arrow|txt|py)$", path, re.I))


def strings_in(node, values: dict, depth: int = 0) -> list[str]:
    """The strings an expression is built from: its constants, and what the names in it are assigned."""
    found = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            found.append(sub.value)
        elif isinstance(sub, ast.Name) and sub.id in values and depth < 3:
            found += strings_in(values[sub.id], values, depth + 1)
    return found


def loads_from_hub(call, values: dict) -> bool:
    """datasets.load_dataset reads from the Hub: its path is a dataset's id, or a builder (json, csv, ...) is given remote
    files; a local folder or file, or a path the code doesn't show, isn't judged a download."""
    path = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "path"), None)
    text = text_of(path, values) if path is not None else "…"
    if "…" in text:
        return False
    if text in DATASET_BUILDERS:
        files = [s for k in call.keywords if k.arg in ("data_files", "data_dir") for s in strings_in(k.value, values)]
        return any(REMOTE_FS.search(s) or any(remote_host(h) for h in URL_HOST.findall(s)) for s in files)
    return bool(REMOTE_FS.search(text) or any(remote_host(h) for h in URL_HOST.findall(text))) or hub_dataset_id(text)


def python_fetches(source: str, have: dict[str, set[str]] | None = None) -> list[str]:
    """What Python fetches from a remote address when it runs: urllib, requests, httpx and the like with an address
    outside the sandbox (as far as the code shows it, built from parts too), huggingface_hub's downloads,
    datasets.load_dataset from the Hub, pandas or numpy reading a remote address; and what the commands it runs with
    subprocess or os.system fetch or download (`have`: what the image installs, as image_packages gives it)."""
    try:
        with warnings.catch_warnings():   # the task's own invalid escapes ("\d" in a plain string) are not this check's to report
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return python_text_fetches(source)
    aliases, values, docstrings = {}, {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = node.value
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)):
            docstrings.add(id(node.body[0].value))
    everywhere = [h for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
                  and id(n) not in docstrings for h in URL_HOST.findall(n.value)]
    found = []
    for node in ast.walk(tree):
        name = dotted_name(node.func, aliases) if isinstance(node, ast.Call) else None
        if not name:
            continue
        module, _, function = name.rpartition(".")
        if function in PY_HUB.get(module, ()):
            found.append(f"{name} from the Hugging Face Hub")
            continue
        if function in PY_SHELL_OUTS.get(module, ()):   # a command it runs: what that fetches or downloads
            fetched, rest = shell_fetches(shell_command(node, values))
            tools, _ = tool_downloads([(line, False) for line in rest], {}, have or {}, False)
            found += [f"{name}: {what}" for what in fetched + tools]
            continue
        if name == "datasets.load_dataset":
            if loads_from_hub(node, values):
                found.append(f"{name} from the Hugging Face Hub")
            continue
        reader = (module == "pandas" and function.startswith("read_")) or function in PY_READERS.get(module, ())
        if function not in PY_FETCH.get(module, ()) and not reader:
            continue
        args = node.args[:1] + [k.value for k in node.keywords
                                if k.arg in ("url", "filepath_or_buffer", "path_or_buf", "io", "fname", "path")]
        built = [text_of(arg, values) for arg in args]   # a URL built from parts: "https://" + HOST + "/expected.json"
        hosts = [h for arg in args for h in hosts_in(arg, values)] + [h for text in built for h in URL_HOST.findall(text)]
        if not hosts and not reader and not any(not remote_host(h) for h in everywhere):
            hosts = everywhere   # a client, or an address built elsewhere: the addresses the file names
        remote = sorted({h.lower() for h in hosts if remote_host(h)} | {s for text in built if reader for s in REMOTE_FS.findall(text)})
        if remote:
            found.append(f"{name} from {', '.join(remote)}")
    return list(dict.fromkeys(found))


def python_text_fetches(text: str) -> list[str]:
    """python_fetches by text, for Python that doesn't parse or sits inside a shell script (python3 -c, a heredoc)."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    calls = list(dict.fromkeys(m.group(1) for m in PY_FETCH_TEXT.finditer(code)))
    remote = sorted({h.lower() for h in URL_HOST.findall(code) if remote_host(h) and h.lower() not in TOOL_HOSTS})
    found = [f"{c} from {'the Hugging Face Hub' if c in ('hf_hub_download', 'snapshot_download') else ', '.join(remote)}"
             for c in calls if remote or c in ("hf_hub_download", "snapshot_download")]
    if any(hub_dataset_id(m.group(2)) for m in LOAD_DATASET_TEXT.finditer(code)):
        found.append("load_dataset from the Hugging Face Hub")
    return found


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def verifier_scripts(task_dir: Path) -> list[tuple[str, str, bool]]:
    """verifier/test.sh and the scripts under verifier/ it sources or runs, one level down: (path, text, whether test.sh
    runs it only in its pytest fallback)."""
    text = read(task_dir / "verifier" / "test.sh")
    if not text:
        return []
    return [("verifier/test.sh", text, False)] + [(rel, read(task_dir / rel), fb) for rel, fb in helper_scripts(task_dir, text).items()]


def data_fetches(task_dir: Path, have: dict[str, set[str]] | None = None) -> dict[str, list[str]]:
    """What the verifier fetches as data when it runs, by file: verifier/test.sh and the scripts it sources or runs, then
    its Python files (with what the commands they run download; `have`: what the image installs)."""
    found = {}
    for rel, text, _ in verifier_scripts(task_dir):
        fetched, rest = shell_fetches(text)
        fetched += python_text_fetches("\n".join(rest))
        if fetched:
            found[rel if rel == "verifier/test.sh" else f"{rel}, which verifier/test.sh runs,"] = fetched
    for path in sorted((task_dir / "verifier").glob("*.py")):
        fetched = python_fetches(read(path), have)
        if fetched:
            found[f"verifier/{path.name}"] = fetched
    return found


def warnings_for(task_dir: Path) -> list[str]:
    """Content problems that don't fail the check: a verifier that needs the network the task turns off, to download
    its tools (unless the image has them) or to fetch data (whatever the image has)."""
    task_md = task_dir / "task.md"
    if not task_md.is_file():
        return []
    m = FRONTMATTER_RE.match(read(task_md))
    if not m or not offline(m.group(1)):
        return []
    found = []
    dockerfile = read(task_dir / "environment" / "Dockerfile")
    have, has_pytest = image_packages(dockerfile), image_installs_pytest(dockerfile)
    tools, from_fallback = [], False
    for rel, text, sourced_in_fallback in verifier_scripts(task_dir):
        code = shell_fetches(text)[1]   # the tools part leaves out what the data part reports
        statements = [(s, fb or sourced_in_fallback) for s, fb in statements_of(code)]
        got, fb = tool_downloads(statements, assignments(code), have, has_pytest)
        tools += [t if rel == "verifier/test.sh" else f"{t} in {rel}" for t in got]
        from_fallback = from_fallback or fb
    tools = list(dict.fromkeys(tools))
    if tools:
        why = (" (it runs pytest from the image when the image has it, but environment/Dockerfile doesn't install pytest)"
               if from_fallback else "")
        found.append(
            f"verifier/test.sh downloads when it runs ({', '.join(tools)}){why}, but task.md sets allow_internet: false. "
            "In a sandbox without network (scripts/run_local.sh, the Arena's oracle and no-op controls) the download fails "
            "or hangs, so even the reference solution can't pass. Install what the verifier needs in environment/Dockerfile and run it "
            "from the image, as starting-kit/template does (pytest==8.4.1 and pytest-json-ctrf==0.3.5); an install of "
            "packages the Dockerfile installs too passes."
        )
    for rel, fetched in data_fetches(task_dir, have).items():
        shown = "; ".join(fetched[:3]) + (f"; and {len(fetched) - 3} more" if len(fetched) > 3 else "")
        found.append(
            f"{rel} fetches files from the network when it runs ({shown}), but task.md sets allow_internet: false. In a "
            "sandbox without network (scripts/run_local.sh, the Arena's oracle and no-op controls) the fetch fails or hangs, "
            "so even the reference solution can't pass, whatever the image installs. Ship what the verifier reads with the "
            "task instead: data under verifier/, which reaches the sandbox with the verifier, after the agent finishes, so "
            "the agent never reads it (examples/seclog-bruteforce-triage keeps verifier/data/auth.log and reads it from "
            "Path(__file__).parent); tools in environment/Dockerfile."
        )
    return found


def check_task(task_dir: Path) -> list[str]:
    issues: list[str] = []

    # task.md ---------------------------------------------------------------
    task_md = task_dir / "task.md"
    if not task_md.exists():
        return [f"Missing required file: task.md"]

    text = task_md.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        issues.append("task.md must start with YAML frontmatter (--- ... ---)")
    else:
        frontmatter = m.group(1)
        body = text[m.end():]
        top_keys = parse_yaml_keys(frontmatter)
        for required in REQUIRED_FRONTMATTER:
            if required not in top_keys:
                issues.append(f"task.md frontmatter missing: {required}")
        metadata_keys = parse_metadata_keys(frontmatter)
        if not {"author_hub", "author_name"} & metadata_keys:
            issues.append("task.md metadata.author_hub required (legacy author_name is accepted)")
        for required in REQUIRED_METADATA:
            if required not in metadata_keys:
                issues.append(f"task.md metadata.{required} required")
        if "## prompt" not in body:
            issues.append("task.md body must contain a '## prompt' section")

    # environment/ ----------------------------------------------------------
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        issues.append("Missing required file: environment/Dockerfile")
    else:
        # Skip comment lines and blanks; the first executable instruction
        # must be FROM.
        first_instr = next(
            (
                line
                for line in dockerfile.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )
        if not first_instr.upper().startswith("FROM "):
            issues.append("environment/Dockerfile first instruction must be FROM")

    # verifier/ -------------------------------------------------------------
    for f in ("verifier/test.sh", "verifier/test_outputs.py", "verifier/verifier.md"):
        if not (task_dir / f).exists():
            issues.append(f"Missing required file: {f}")
    rubrics = task_dir / "verifier" / "rubrics"
    if not rubrics.is_dir():
        issues.append("Missing required directory: verifier/rubrics/")
    else:
        if not any(p.suffix == ".md" for p in rubrics.iterdir()):
            issues.append("verifier/rubrics/ must contain at least one *.md rubric")

    # oracle/ ---------------------------------------------------------------
    if not (task_dir / "oracle" / "solve.sh").exists():
        issues.append("Missing required file: oracle/solve.sh")

    return issues


def iter_task_dirs(roots: Iterable[str]) -> Iterable[Path]:
    for root in roots:
        p = Path(root)
        if p.is_dir() and (p / "task.md").exists():
            yield p
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "task.md").exists():
                    yield child


def main(argv: list[str]) -> int:
    targets = argv[1:] or ["starting-kit/examples", "starting-kit/template"]
    overall_ok = True
    any_seen = False
    for task_dir in iter_task_dirs(targets):
        any_seen = True
        issues = check_task(task_dir)
        if issues:
            overall_ok = False
            print(f"✗ {task_dir.name} — {len(issues)} issue(s):")
            for i in issues:
                print(f"  → {i}")
        else:
            print(f"✓ {task_dir.name} — structure valid")
        for w in warnings_for(task_dir):
            print(f"  ⚠ warning: {w}")
    if not any_seen:
        print("no task directories found")
        return 1
    print(
        "This checks structure only: that the required files, frontmatter keys and "
        "'## prompt' section exist, not their contents (the template's placeholders pass) "
        "or whether the verifier and oracle work. Its one content check, a verifier that "
        "downloads tools or data while the task turns the network off, is a warning."
    )
    if overall_ok:
        print(
            "Next: scripts/run_local.sh <task> (the oracle must score 1) and "
            "scripts/run_local.sh <task> --skip-oracle (an empty trial must not)."
        )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

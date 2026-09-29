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
- verifier/test.sh downloads tools when it runs (apt-get, curl, pip or uv
  installs, uvx). A test.sh that runs pytest from the image when the image has
  it (command -v pytest) and a Dockerfile that installs pytest pass, as the
  template does.
- The verifier fetches data when it runs: in test.sh, curl or wget into a file
  or a pipe, git clone, hf download; in its Python (test_outputs.py), urllib,
  requests, httpx or huggingface_hub reading a remote address. Nothing in the
  image stands in for that data, so this warns whatever the pytest fallback.

Exit code: 0 if every task validates, 1 if any task has issues.
"""
from __future__ import annotations

import ast
import re
import shlex
import sys
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
    "author_name",
    "author_email",
    "category",
)

# What verifier/test.sh downloads with when it runs: package installs, curl or wget, uvx and the like.
DOWNLOADS = re.compile(
    r"\b(?P<install>(?:apt-get|apt|apk|yum|dnf|pip3?|uv|conda|mamba|npm|pnpm|yarn|gem|cargo|go)\s+(?:-\S+\s+)*"
    r"(?:install|add|get|sync|tool))\b"
    r"|\b(?P<fetch>curl|wget|uvx|pipx\s+run|npx|git\s+clone)\b"
)
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
PY_FETCH_TEXT = re.compile(
    r"\b(urlopen|urlretrieve|hf_hub_download|snapshot_download|requests\.(?:get|post|request|Session)|"
    r"httpx\.(?:get|post|request|stream|Client|AsyncClient)|HTTPS?Connection)\s*\("
)


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


def shell_fetches(text: str) -> tuple[list[str], list[str]]:
    """The data a shell script fetches when it runs (each command, shortened) and its lines without those commands.

    Data: curl or wget into a file, a variable or a pipe (not into a shell: that runs an installer), unless every address
    is in the sandbox or an installer's, a package index's or a release binary; git clone of a repository that isn't a
    local path; hf download."""
    lines = shell_lines(text)
    names = {}
    for line in lines:
        m = re.match(r"(?:export\s+|local\s+|readonly\s+)?([A-Za-z_]\w*)=(.*)$", line)
        if m:
            names[m.group(1)] = (words_of(m.group(2)) or [""])[0]
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
        for start, end, command in sorted(spans):
            short = " ".join(command.split())
            fetched.append(short if len(short) <= 90 else short[:89] + "…")
        for start, end, _ in sorted(spans, reverse=True):
            line = line[:start] + " " + line[end:]
        rest.append(line)
    return list(dict.fromkeys(fetched)), rest


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


def python_fetches(source: str) -> list[str]:
    """What Python fetches from a remote address when it runs: urllib, requests, httpx and the like with an address
    outside the sandbox (as far as the code shows it), huggingface_hub's downloads, pandas reading a remote address."""
    try:
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
        reader = module == "pandas" and function.startswith("read_")
        if function not in PY_FETCH.get(module, ()) and not reader:
            continue
        args = node.args[:1] + [k.value for k in node.keywords if k.arg in ("url", "filepath_or_buffer", "path_or_buf", "io")]
        hosts = [h for arg in args for h in hosts_in(arg, values)]
        if not hosts and not reader and not any(not remote_host(h) for h in everywhere):
            hosts = everywhere   # a client, or an address built elsewhere: the addresses the file names
        remote = sorted({h.lower() for h in hosts if remote_host(h)})
        if remote:
            found.append(f"{name} from {', '.join(remote)}")
    return list(dict.fromkeys(found))


def python_text_fetches(text: str) -> list[str]:
    """python_fetches by text, for Python that doesn't parse or sits inside a shell script (python3 -c, a heredoc)."""
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    calls = list(dict.fromkeys(m.group(1) for m in PY_FETCH_TEXT.finditer(code)))
    remote = sorted({h.lower() for h in URL_HOST.findall(code) if remote_host(h) and h.lower() not in TOOL_HOSTS})
    return [f"{c} from {'the Hugging Face Hub' if c in ('hf_hub_download', 'snapshot_download') else ', '.join(remote)}"
            for c in calls if remote or c in ("hf_hub_download", "snapshot_download")]


def data_fetches(task_dir: Path) -> dict[str, list[str]]:
    """What the verifier fetches as data when it runs, by file: verifier/test.sh, then its Python files."""
    found = {}
    test_sh = task_dir / "verifier" / "test.sh"
    if test_sh.is_file():
        text = test_sh.read_text(encoding="utf-8", errors="replace")
        fetched, rest = shell_fetches(text)
        fetched += python_text_fetches("\n".join(rest))
        if fetched:
            found["verifier/test.sh"] = fetched
    for path in sorted((task_dir / "verifier").glob("*.py")):
        fetched = python_fetches(path.read_text(encoding="utf-8", errors="replace"))
        if fetched:
            found[f"verifier/{path.name}"] = fetched
    return found


def warnings_for(task_dir: Path) -> list[str]:
    """Content problems that don't fail the check: a verifier that needs the network the task turns off, to download
    its tools (unless the image has them) or to fetch data (whatever the image has)."""
    task_md, test_sh = task_dir / "task.md", task_dir / "verifier" / "test.sh"
    if not task_md.is_file():
        return []
    m = FRONTMATTER_RE.match(task_md.read_text(encoding="utf-8", errors="replace"))
    if not m or not offline(m.group(1)):
        return []
    found = []
    text = test_sh.read_text(encoding="utf-8", errors="replace") if test_sh.is_file() else ""
    _, code = shell_fetches(text)   # the tools part leaves out what the data part reports
    tools = []
    for line in code:
        for hit in DOWNLOADS.finditer(line):
            words = (hit.group("install") or hit.group("fetch")).split()
            tools.append(f"{words[0]} {words[-1]}" if hit.group("install") else " ".join(words))
    tools = list(dict.fromkeys(tools))
    dockerfile = task_dir / "environment" / "Dockerfile"
    has_pytest = dockerfile.is_file() and image_installs_pytest(dockerfile.read_text(encoding="utf-8", errors="replace"))
    fallback = any(PYTEST_IN_IMAGE.search(line) for line in code)
    if tools and not (fallback and has_pytest):   # with both, the download is a fallback for an image without pytest
        why = (" (it runs pytest from the image when the image has it, but environment/Dockerfile doesn't install pytest)"
               if fallback else "")
        found.append(
            f"verifier/test.sh downloads when it runs ({', '.join(tools)}){why}, but task.md sets allow_internet: false. "
            "In a sandbox without network (scripts/run_local.sh, the Arena's oracle and no-op controls) the download fails "
            "or hangs, so even the reference solution can't pass. Install what the verifier needs in environment/Dockerfile and run it "
            "from the image, as starting-kit/template does (pytest==8.4.1 and pytest-json-ctrf==0.3.5)."
        )
    for rel, fetched in data_fetches(task_dir).items():
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

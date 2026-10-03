"""Tests for scripts/check_submission.py's arguments.

Runs standalone (`python scripts/test_check_submission.py`) or under pytest. No
third-party deps. A collection is a folder with submission.yaml and envs/; the
script takes one (the Starter kit's my-collection/), a folder of collections
(submissions/, the default), or --help.
"""

import contextlib
import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_submission as C

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starting-kit", "template")


def _collection(parent, name="my-collection"):
    """The Starter kit's layout: my-collection/submission.yaml and envs/my-task/ copied from the template."""
    root = os.path.join(parent, name)
    shutil.copytree(TEMPLATE, os.path.join(root, "envs", "my-task"))
    with open(os.path.join(root, "submission.yaml"), "w", encoding="utf-8") as fh:
        fh.write("team_name: Test Team\ncontact_email: test@example.org\ntrack: environments\n")
    return root


def _run(*args, cwd=None):
    out, code, here = io.StringIO(), None, os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                code = C.main(["check_submission.py", *args])
            except SystemExit as exc:   # argparse: --help and usage errors
                code = exc.code
    finally:
        os.chdir(here)
    return code, out.getvalue()


def test_a_collection_folder_is_checked_as_one_collection():
    # Round 8 (F8-13): this answered "✗ envs — Missing required file: submission.yaml", though the file exists.
    with tempfile.TemporaryDirectory() as tmp:
        folder = _collection(tmp)
        for arg in (folder, folder + os.sep):
            code, out = _run(arg)
            assert code == 0, out
            assert out.strip() == "✓ my-collection — structure valid", out


def test_a_folder_of_collections_checks_each_one():
    with tempfile.TemporaryDirectory() as tmp:
        _collection(tmp, "team-a")
        _collection(tmp, "team-b")
        os.makedirs(os.path.join(tmp, "_scratch"))   # skipped, as in submissions/
        code, out = _run(tmp)
        assert code == 0, out
        assert out.splitlines() == ["✓ team-a — structure valid", "✓ team-b — structure valid"], out


def test_a_folder_of_collections_still_reports_one_without_submission_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        _collection(tmp, "team-a")
        os.makedirs(os.path.join(tmp, "team-b", "envs"))
        code, out = _run(tmp)
        assert code == 1, out
        assert "✗ team-b — 1 issue(s):" in out and "Missing required file: submission.yaml" in out, out


def test_help_prints_usage_and_checks_nothing():
    # Round 8 (F8-13): "--help" answered "no --help/ directory — nothing to check".
    code, out = _run("--help")
    assert code == 0, out
    assert out.startswith("usage: check_submission.py [-h] [FOLDER ...]"), out
    assert "a collection (the folder that holds submission.yaml and envs/)" in out, out


def test_an_envs_folder_or_a_task_package_is_explained():
    with tempfile.TemporaryDirectory() as tmp:
        folder = _collection(tmp)
        code, out = _run(os.path.join(folder, "envs"))
        assert code == 1 and "holds task packages, not collections" in out and "scripts/check_task.py" in out, out
        code, out = _run(os.path.join(folder, "envs", "my-task"))
        assert code == 1 and "is a task package, not a collection" in out, out


def test_a_missing_folder_is_a_usage_error():
    with tempfile.TemporaryDirectory() as tmp:
        code, out = _run(os.path.join(tmp, "nope"))
        assert code == 2 and "no such folder" in out, out


def test_the_default_is_submissions_and_its_absence_is_not_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        code, out = _run(cwd=tmp)
        assert code == 0 and out.strip() == "no submissions/ directory — nothing to check", out
        _collection(os.path.join(tmp, "submissions"), "team-a")
        code, out = _run(cwd=tmp)
        assert code == 0 and out.strip() == "✓ team-a — structure valid", out


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"{len(tests)} tests passed")


def test_the_submitting_hub_account_is_the_contact():
    # Space PR #4 (hub identities): submission.yaml needs only team_name and track; an older contact_email still passes.
    with tempfile.TemporaryDirectory() as tmp:
        folder = _collection(tmp)
        manifest = os.path.join(folder, "submission.yaml")
        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write("team_name: Test Team\ntrack: environments\n")
        code, out = _run(folder)
        assert code == 0, out
        with open(manifest, "w", encoding="utf-8") as fh:
            fh.write("team_name: Test Team\n")
        code, out = _run(folder)
        assert code != 0 and "submission.yaml missing: track" in out, out

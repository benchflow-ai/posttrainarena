from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers.utils import get_json_schema

from posttrainarena.benchflow_pipeline.openenv.server import LocalOpenEnvServer
from posttrainarena.benchflow_pipeline.openenv.tool_env import (
    OpenEnvToolEnvironment,
    build_openenv_integration,
    openenv_environment_reward,
)


ROOT = Path(__file__).resolve().parents[4]


class FakeBenchFlowEnvironment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.task_id: str | None = None
        self.reward = 0.0
        self.rollout_dir: Path | None = None
        self.last_returncode: int | None = None
        self.finalize_count = 0
        self.closed = False
        self.last_reset_kwargs: dict[str, Any] = {}

    def reset(self, **kwargs: Any) -> str:
        self.last_reset_kwargs = dict(kwargs)
        self.task_id = str(kwargs["benchflow_task_id"])
        self.rollout_dir = self.root / self.task_id / str(id(self))
        self.rollout_dir.mkdir(parents=True)
        self.reward = 0.0
        self.last_returncode = None
        self.closed = False
        return f"ready:{self.task_id}"

    def run_bash(self, command: str) -> str:
        self.last_returncode = 0
        return f"ran:{command}"

    def submit(self, answer: str) -> str:
        self.reward = 1.0 if answer == "correct" else 0.0
        self._finalize()
        return f"submission recorded; reward={self.reward:g}"

    def _finalize(self) -> None:
        self.finalize_count += 1
        self.closed = True

    def _close(self) -> None:
        self.closed = True


def _row(task_id: str) -> dict[str, str]:
    return {"benchflow_task_id": task_id, "benchflow_task_dir": f"/{task_id}"}


def test_public_tools_have_transformers_json_schemas(tmp_path: Path) -> None:
    server = LocalOpenEnvServer(lambda: FakeBenchFlowEnvironment(tmp_path))
    environment = OpenEnvToolEnvironment(server.base_url)
    try:
        assert get_json_schema(environment.run_bash)["function"]["name"] == "run_bash"
        assert get_json_schema(environment.submit)["function"]["name"] == "submit"
    finally:
        environment._close()
        server.close()


def test_real_benchflow_spec_builds_openenv_tool_factory(tmp_path: Path) -> None:
    from benchflow.integrations.trl import BashHarnessConfig, BenchFlowSpec

    spec = BenchFlowSpec(
        tasks_dir=ROOT / "starting-kit/examples",
        include_tasks=["seclog-bruteforce-triage"],
        bash_harness=BashHarnessConfig(
            environment="docker",
            jobs_dir=tmp_path,
            reset_message="solve",
        ),
    )
    integration = build_openenv_integration(spec, None)
    environment = integration.environment_factory()
    try:
        assert integration.task_rows[0]["benchflow_task_id"] == (
            "seclog-bruteforce-triage"
        )
        assert get_json_schema(environment.run_bash)["function"]["name"] == "run_bash"
        assert get_json_schema(environment.submit)["function"]["name"] == "submit"
    finally:
        integration.close()


def test_real_openenv_protocol_preserves_tools_reward_and_artifacts(
    tmp_path: Path,
) -> None:
    environments: list[FakeBenchFlowEnvironment] = []

    def factory() -> FakeBenchFlowEnvironment:
        environment = FakeBenchFlowEnvironment(tmp_path)
        environments.append(environment)
        return environment

    server = LocalOpenEnvServer(factory)
    environment = OpenEnvToolEnvironment(server.base_url)
    try:
        assert environment.reset(**_row("task-a")) == "ready:task-a"
        assert environment.run_bash("pwd") == "ran:pwd"
        assert environment.last_returncode == 0
        state = environment._state()
        assert state.task_id == "task-a"
        assert state.step_count == 1
        assert state.done is False
        assert environment.submit("correct") == "submission recorded; reward=1"
        assert environment.reward == 1.0
        assert environment.rollout_dir is not None
        assert environment.rollout_dir.is_dir()
        assert environments[-1].finalize_count == 1
    finally:
        environment._close()
        server.close()


def test_reward_function_finalizes_unsubmitted_episode(tmp_path: Path) -> None:
    environments: list[FakeBenchFlowEnvironment] = []

    def factory() -> FakeBenchFlowEnvironment:
        environment = FakeBenchFlowEnvironment(tmp_path)
        environments.append(environment)
        return environment

    server = LocalOpenEnvServer(factory)
    environment = OpenEnvToolEnvironment(server.base_url)
    try:
        environment.reset(**_row("task-a"))
        assert openenv_environment_reward([None], environments=[environment]) == [0.0]
        assert environments[-1].finalize_count == 1
        assert environments[-1].closed is True
        assert environment._closed is False
        assert environment.reset(**_row("task-b")) == "ready:task-b"
        assert environment.run_bash("pwd") == "ran:pwd"
    finally:
        environment._close()
        server.close()


def test_sessions_do_not_share_benchflow_state(tmp_path: Path) -> None:
    server = LocalOpenEnvServer(lambda: FakeBenchFlowEnvironment(tmp_path))
    first = OpenEnvToolEnvironment(server.base_url)
    second = OpenEnvToolEnvironment(server.base_url)
    try:
        first.reset(**_row("task-a"))
        second.reset(**_row("task-b"))
        first.run_bash("one")

        assert first.task_id == "task-a"
        assert second.task_id == "task-b"
        assert first.rollout_dir != second.rollout_dir
        assert second.last_returncode is None
    finally:
        first._close()
        second._close()
        server.close()


def test_reset_replaces_prior_benchflow_runtime(tmp_path: Path) -> None:
    environments: list[FakeBenchFlowEnvironment] = []

    def factory() -> FakeBenchFlowEnvironment:
        environment = FakeBenchFlowEnvironment(tmp_path)
        environments.append(environment)
        return environment

    server = LocalOpenEnvServer(factory)
    environment = OpenEnvToolEnvironment(server.base_url)
    try:
        environment.reset(**_row("task-a"))
        first_rollout = environment.rollout_dir
        environment.reset(**_row("task-b"))

        assert environment.task_id == "task-b"
        assert environment.rollout_dir != first_rollout
        assert environments[-1].task_id == "task-b"
    finally:
        environment._close()
        server.close()


def test_server_resolves_task_id_to_server_side_task_path(tmp_path: Path) -> None:
    environments: list[FakeBenchFlowEnvironment] = []

    def factory() -> FakeBenchFlowEnvironment:
        environment = FakeBenchFlowEnvironment(tmp_path)
        environments.append(environment)
        return environment

    server = LocalOpenEnvServer(factory)
    server.close()
    from posttrainarena.benchflow_pipeline.openenv.server import BenchFlowOpenEnv
    from posttrainarena.benchflow_pipeline.openenv.models import PostTrainAction

    environment = BenchFlowOpenEnv(
        factory,
        {
            "task-a": {
                "benchflow_task_id": "task-a",
                "benchflow_task_dir": "/server/tasks/task-a",
            }
        },
    )
    try:
        environment.reset(
            benchflow_task_id="task-a",
            benchflow_task_dir="/client/path/task-a",
        )
        assert environments[-1].last_reset_kwargs["benchflow_task_dir"] == (
            "/server/tasks/task-a"
        )
        environment.step(PostTrainAction(type="finalize"))
    finally:
        environment.close()

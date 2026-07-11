"""OpenEnv server that delegates every episode to BenchFlow."""

from __future__ import annotations

import atexit
import socket
import threading
import time
from collections.abc import Callable
from typing import Any
from urllib.request import urlopen

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment

from .models import PostTrainAction, PostTrainObservation, PostTrainState


class BenchFlowOpenEnv(Environment):
    """Protocol surface over one owned BenchFlow runtime environment."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        environment_factory: Callable[[], Any],
        task_rows: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._environment = environment_factory()
        self._task_rows = task_rows or {}
        self._episode_id: str | None = None
        self._step_count = 0
        self._done = False

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> PostTrainObservation:
        del seed
        task_id = kwargs.get("benchflow_task_id")
        resolved = self._task_rows.get(str(task_id)) if task_id is not None else None
        reset_kwargs = {**kwargs, **resolved} if resolved else kwargs
        output = self._environment.reset(**reset_kwargs) or ""
        self._episode_id = episode_id
        self._step_count = 0
        self._done = False
        return self._observation(str(output))

    def step(
        self,
        action: PostTrainAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> PostTrainObservation:
        del timeout_s, kwargs
        if self._done:
            raise RuntimeError("episode is already complete")
        self._step_count += 1
        if action.type == "run_bash":
            if action.command is None:
                raise ValueError("run_bash requires command")
            return self._observation(self._environment.run_bash(action.command))
        if action.type == "submit":
            if action.answer is None:
                raise ValueError("submit requires answer")
            output = self._environment.submit(action.answer)
        else:
            self._environment._finalize()
            output = "episode finalized"
        self._done = True
        return self._observation(output)

    @property
    def state(self) -> PostTrainState:
        return PostTrainState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._environment.task_id,
            done=self._done,
            reward=float(self._environment.reward),
            rollout_dir=_path_string(self._environment.rollout_dir),
        )

    def close(self) -> None:
        self._environment._close()

    def _observation(self, output: str) -> PostTrainObservation:
        return PostTrainObservation(
            output=output,
            reward=float(self._environment.reward),
            done=self._done,
            task_id=self._environment.task_id,
            rollout_dir=_path_string(self._environment.rollout_dir),
            last_returncode=self._environment.last_returncode,
        )


def create_openenv_app(
    environment_factory: Callable[[], Any],
    task_rows: dict[str, dict[str, Any]] | None = None,
):
    return create_app(
        lambda: BenchFlowOpenEnv(environment_factory, task_rows),
        PostTrainAction,
        PostTrainObservation,
        env_name="posttrain-benchflow",
        max_concurrent_envs=64,
    )


class LocalOpenEnvServer:
    """Own a real local HTTP/WebSocket OpenEnv server for one integration."""

    def __init__(self, environment_factory: Callable[[], Any]) -> None:
        self._environment_factory = environment_factory
        self._server: Any | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._base_url: str | None = None
        atexit.register(self.close)

    @property
    def base_url(self) -> str:
        with self._lock:
            if self._base_url is None:
                self._start()
            assert self._base_url is not None
            return self._base_url

    def close(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
            self._base_url = None
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=5)

    def _start(self) -> None:
        import uvicorn

        port = _unused_port()
        app = create_openenv_app(self._environment_factory)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        self._server = server
        self._thread = thread
        self._base_url = f"http://127.0.0.1:{port}"
        thread.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with urlopen(f"{self._base_url}/health", timeout=0.25):
                    return
            except OSError:
                time.sleep(0.05)
        self.close()
        raise RuntimeError("OpenEnv server did not become healthy")


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _path_string(value: Any) -> str | None:
    return None if value is None else str(value)

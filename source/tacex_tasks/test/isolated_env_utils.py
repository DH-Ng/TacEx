"""Helpers to run env checks in isolated Isaac Sim processes (no Isaac imports here).

Why: Isaac Sim keeps ~0.65 GiB GPU memory per closed env, so running all envs in one process runs out of GPU memory,
and a crash in one env would take down all following tests. Each task gets its own process (``run_env_check.py``);
both ``num_envs`` variants of a task share that process.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile

import pytest

RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_env_check.py")
NUM_ENVS = (32, 1)
LIST_TIMEOUT_S = 600
TASK_TIMEOUT_S = 900
LOG_TAIL_LINES = 40


def _run(args: list[str], timeout: float) -> tuple[dict | list | None, str]:
    """Runs the runner script; returns (parsed JSON output or None, combined stdout/stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.json")
        try:
            proc = subprocess.run(
                [sys.executable, RUNNER, *args, "--out", out],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            log = proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as e:
            log = f"TIMEOUT after {timeout} s\n" + (e.stdout or b"").decode(errors="replace")
        data = None
        if os.path.exists(out):
            with open(out) as f:
                data = json.load(f)
        return data, log


def _tail(log: str) -> str:
    return "\n".join(log.splitlines()[-LOG_TAIL_LINES:])


def list_tasks(factory: bool) -> list[str]:
    """Task IDs to test (collected in a separate Isaac process, since the registry needs Isaac imports)."""
    tasks, log = _run(["--list"] + (["--factory"] if factory else []), LIST_TIMEOUT_S)
    if tasks is None:
        raise RuntimeError(f"Could not collect task list:\n{_tail(log)}")
    return tasks


@functools.lru_cache(maxsize=None)
def run_task(task_name: str) -> tuple[dict | None, str]:
    """Runs all NUM_ENVS variants of a task in one fresh process (cached, so each task starts Isaac only once)."""
    return _run(["--task", task_name, "--num_envs", *map(str, NUM_ENVS)], TASK_TIMEOUT_S)


def check_task(task_name: str, num_envs: int):
    """Pytest body: fails with the env's error (or the process log tail if it crashed/hung)."""
    results, log = run_task(task_name)
    result = (results or {}).get(str(num_envs))
    if result is None:
        pytest.fail(f"{task_name} (num_envs={num_envs}) produced no result (crash or timeout). Log tail:\n{_tail(log)}")
    if not result["passed"]:
        pytest.fail(f"{task_name} (num_envs={num_envs}): {result['message']}")

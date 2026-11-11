# Adjusted version from https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/test/test_factory_environments.py

"""Runs every (non-UIPC) TacEx factory env with random actions and checks for valid signals.

Each task runs in its own Isaac Sim process (see ``isolated_env_utils.py``), so this file does not launch Isaac.
"""

import pytest
from isolated_env_utils import NUM_ENVS, check_task, list_tasks


@pytest.mark.parametrize("num_envs", NUM_ENVS)
@pytest.mark.parametrize("task_name", list_tasks(factory=True))
@pytest.mark.isaacsim_ci
def test_factory_environments(task_name, num_envs):
    check_task(task_name, num_envs)

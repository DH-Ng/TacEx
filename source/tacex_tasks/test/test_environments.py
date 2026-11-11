# Adjusted version from https://github.com/isaac-sim/IsaacLab/tree/9d6321463067c541ce1a24531ff87f99a18fd8f7/source/isaaclab_tasks/test

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runs every (non-factory, non-UIPC) TacEx env with random actions and checks for valid signals.

Each task runs in its own Isaac Sim process (see ``isolated_env_utils.py``), so this file does not launch Isaac.
Set ``TACEX_TEST_HEADLESS=0`` to watch the envs in the GUI.
"""

import pytest
from isolated_env_utils import NUM_ENVS, check_task, list_tasks


@pytest.mark.parametrize("num_envs", NUM_ENVS)
@pytest.mark.parametrize("task_name", list_tasks(factory=False))
def test_environments(task_name, num_envs):
    check_task(task_name, num_envs)

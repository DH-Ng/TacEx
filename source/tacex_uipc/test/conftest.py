"""Shared setup for the tacex_uipc tests.

The tests are grouped in sub directories. pytest only puts a test file's own directory on ``sys.path``, so this adds
the test root as well, for the shared helpers (e.g. ``debug_vis_helper``).

Only standard library code here: conftest.py is imported before the test modules launch the Isaac Sim app.
"""

import pathlib
import sys

_TEST_ROOT = str(pathlib.Path(__file__).resolve().parent)
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

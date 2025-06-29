"""
Python module serving as a project/extension template.
"""

from .uipc_sim import *
from .uipc_object.uipc_object import *
from .uipc_attachments import *

from .uipc_interactive_scene import UipcInteractiveScene
from .direct_uipc_rl_env import *

# Register UI extensions.
from .ui_extension_example import *

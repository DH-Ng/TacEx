"""Ready to use configurations for the GelSight Mini.

The configurations differ mainly in the employed simulation approach and their combination.
"""

from .generic_gsmini_cfg import GeneralGelSightMiniCfg

from .gsmini_taxim import GELSIGHT_MINI_TAXIM_CFG
from .gsmini_taxim_fots import GELSIGHT_MINI_TAXIM_FOTS_CFG
from .gsmini_pix2pix import GELSIGHT_MINI_PIX2PIX_CFG

try:
    from .gsmini_taxim_fem import GELSIGHT_MINI_TAXIM_FEM_CFG
except ImportError:
    pass

try:  # needs isaaclab_contrib
    from .gsmini_tacsl import GELSIGHT_MINI_TACSL_CFG, GELSIGHT_MINI_TACSL_SIM_CFG
except ImportError:
    pass

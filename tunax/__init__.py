"""
Importation of Tunax classes and functions for shortcuts.

"""


import jax
jax.config.update('jax_enable_x64', True)

from .space import Grid, State, Trajectory
from .case import Case, CaseTracable
from .closure import ClosureParametersAbstract, ClosureStateAbstract, Closure
from .database import Data, Obs, Weights
from .functions import tridiag_solve, add_boundaries
from .closures_registry import CLOSURES_REGISTRY
from .model import (
    SingleColumnModel, lmd_swfrac, advance_tra_ed, advance_dyn_cor_ed, diffusion_solver
)
from .fitter import FittableParameter, FittableParametersSet

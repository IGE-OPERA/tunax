"""
Global variable and functions used by several tests.

The .jld2 and .nc files that are used are extracted from a Large Eddy Simulation (LES) database [1].

References
----------
[1] Wagner, G. L. (2026). Reference Large-Eddy Simulations of the upper-ocean boundary layer from
    the CATKE calibration suite (Wagner et al., 2025) (Version 1.0.0) [Dataset]. Zenodo.
    https://doi.org/10.5281/zenodo.20057136

"""


from pathlib import Path
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import equinox as eqx

from tunax import Data


LES_PATH = Path('tests') / 'data' / 'test_database' / 'les_6h_4m_free_convection.nc'
"""Path pointing to the 6h, 4m, free convection LES from the database."""
LES_MAPPING: dict[str, dict[str, str]] = {
    'variables': {
        'time': 'time',
        'zr': 'zr',
        'zw': 'zw',
        'u': 'u',
        'v': 'v',
        'b': 'b',
        'pt': 'pt'
    },
    'parameters': {
        'grav': 'grav',
        'fcor': 'fcor',
        'ustr_sfc': 'ustr_sfc'
    },
    'metadatas': {},
    'adjust_params': {
        'b_sfc': 'b_sfc',
        'omega_p_inv': 'pt_timescale',
        'lambda_c': 'pt_width',
        'zc_m': 'pt_depth',
        'lz': 'pt_lz',
        'jb': 'b_sunny_flux',
        'eps1': 'b_eps1',
        'lambda1': 'b_lambda1',
        'lambda2': 'b_lambda2'
    }
}
"""
Dict[str, Dict[str, str]] : Access links to data in .nc files.
"""

def forcing_passive_tracer(
        z: float,
        omega_p: float,
        lambda_c: float,
        zc: float,
        lz: float
    ) -> float:
    """
    Forcing function of the passive tracer as a gaussian function centered at the bottom of the
    mixing depth.
    """
    omega_m = omega_p*lambda_c*jnp.sqrt(2*jnp.pi)/lz
    fz = omega_p*jnp.exp(-(z-zc)**2/(2*lambda_c**2)) + omega_m
    return float(fz)

def adjust_fun(data: Data, adjust_pars: dict[str, Any]) -> Data:
    """
    Adjusting function for free_convection LES. 
    """
    case = data.case
    # passive tracer forcing
    omega_p = 1/adjust_pars['omega_p_inv']
    lambda_c = adjust_pars['lambda_c']
    zc_m = adjust_pars['zc_m']
    lz = adjust_pars['lz']
    def wrapped_forcing_pt(z: float) -> float:
        return forcing_passive_tracer(z, omega_p, lambda_c, -zc_m, lz)
    case = replace(case, pt_forcing=wrapped_forcing_pt)
    case = replace(case, b_forcing=(0., adjust_pars['b_sfc']))
    return eqx.tree_at(lambda t: t.case, data, case)

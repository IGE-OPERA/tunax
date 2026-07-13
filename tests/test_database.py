"""
Unit tests for the module tunax.database

"""

import pytest
import jax.numpy as jnp
import numpy as np
from h5py import File as H5pyFile
from h5py import Dataset as H5pyDataset
from pathlib import Path
from tunax.database import Data, DimsType
from typing import cast, Dict, Any
from dataclasses import replace
import equinox as eqx


nc_path = Path('tests') / 'data' / 'test_database' / 'test.nc'
jld2_path = Path('tests') / 'data' / 'test_database' / 'test.jld2'

jld2_names_mapping: Dict[str, Dict[str, str]] = {
    'variables': {
        'time': 'timeseries/t',
        'zr': 'grid/zᵃᵃᶜ',
        'zw': 'grid/zᵃᵃᶠ',
        'u': 'timeseries/u',
        'v': 'timeseries/v',
        'b': 'timeseries/b',
        'pt': 'timeseries/c',
        'nz': 'grid/Nz'
    },
    'parameters': {
        'ustr_sfc': 'parameters/momentum_flux',
        'fcor': 'parameters/coriolis_parameter'
    },
    'metadatas': {},
    'adjust_params': {
        'b_sfc': 'parameters/buoyancy_flux',
        'zc_m': 'parameters/tracer_forcing_depth',
        'omega_p_inv': 'parameters/tracer_forcing_timescale',
        'lambda_c': 'parameters/tracer_forcing_width',
        'jb': 'parameters/penetrating_buoyancy_flux'
    }
}
jld2_dims_mapping = cast(Dict[str, DimsType], {
    'zr': (None,),
    'zw': (None,),
    'u': (None, 0, 0),
    'v': (None, 0, 0),
    'b': (None, 0, 0),
    'pt': (None, 0, 0)
})
nc_names_mapping: Dict[str, Dict[str, str]] = {
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

def test__proj_dims():
    # no cut with 2 dimensions
    arr = np.arange(20).reshape(4, 5)
    result = Data._proj_dims(arr, (None, 2), 4)
    np.testing.assert_array_equal(np.asarray(result), np.array([2, 7, 12, 17]))
    # even number cut
    arr = np.arange(10)
    result = Data._proj_dims(arr, (None,), 6)
    np.testing.assert_array_equal(np.asarray(result), np.array([2, 3, 4, 5, 6, 7]))
    # odd number cut
    arr = np.arange(11)
    with pytest.warns(UserWarning):
        result = Data._proj_dims(arr, (None,), 6)
        np.testing.assert_array_equal(np.asarray(result), np.array([2, 3, 4, 5, 6, 7]))
    # wrong len of dims
    arr = np.zeros((4, 5))
    with pytest.raises(ValueError):
        Data._proj_dims(arr, (None,), 4)
    # wrong number of None in dims
    arr = np.zeros((4, 5))
    with pytest.raises(ValueError):
        Data._proj_dims(arr, (None, None), 4)
    # empty dimensions from jld2 file (test on shapes)
    jl = H5pyFile(jld2_path, 'r')
    arr = cast(H5pyDataset, jl['timeseries/b/0'])
    assert Data._proj_dims(arr, (None, 0, 0), 64).shape == (64,)

def test__load_jld2():
    # error test on importation (case nz=None)
    _, _ = Data._load_jld2(str(jld2_path), jld2_names_mapping, None, jld2_dims_mapping)
    # error test on importation (case nz=64)
    _, _ = Data._load_jld2(str(jld2_path), jld2_names_mapping, 64, jld2_dims_mapping)

def test__load_nc():
    # error test on importation
    _, _ = Data._load_nc(str(nc_path), nc_names_mapping)

def forcing_passive_tracer(z: float, omega_p: float, lambda_c: float, zc: float, lz: float):
    omega_m = omega_p*lambda_c*jnp.sqrt(2*jnp.pi)/lz
    fz = omega_p*jnp.exp(-(z-zc)**2/(2*lambda_c**2)) + omega_m
    return  fz

def forcing_buoyancy_sunny(
        z: float,
        jb: float,
        eps1: float = 0.6,
        lambda1: float = 1.,
        lambda2: float = 16.
    ):
    fz = -jb*(eps1/lambda1*jnp.exp(z/lambda1) + (1-eps1)/lambda2*jnp.exp(z/lambda2))
    return  fz

def adjust_fun(data: Data, adjust_pars: Dict[str, Any]) -> Data:
    case = data.case
    # passive tracer forcing
    omega_p = 1/adjust_pars['omega_p_inv']
    lambda_c = adjust_pars['lambda_c']
    zc_m = adjust_pars['zc_m']
    lz = adjust_pars['lz']
    def wrapped_forcing_pt(z: float):
        return forcing_passive_tracer(z, omega_p, lambda_c, -zc_m, lz)
    case = replace(case, pt_forcing=wrapped_forcing_pt)
    # sunny forcing
    if adjust_pars['forcing'] == 'strong_wind_and_sunny':
        jb = adjust_pars['jb']
        eps1 = adjust_pars['eps1']
        lambda1 = adjust_pars['lambda1']
        lambda2 = adjust_pars['lambda2']
        def wrapped_forcing_b_sunny(z: float):
            return forcing_buoyancy_sunny(z, jb, eps1, lambda1, lambda2)
        case = replace(case, b_forcing=wrapped_forcing_b_sunny)
    # classical buoyancy forcing
    else:
        case = replace(case, b_forcing=(0., adjust_pars['b_sfc']))
    return eqx.tree_at(lambda t: t.case, data, case)

def test_load():
    # importation without adjusting function
    data = Data.load(str(nc_path), nc_names_mapping)
    assert data.case.do_pt == True
    assert data.case.eos_tracers == 'b'
    # error check on importation with adjusting function
    _ = Data.load(str(nc_path), nc_names_mapping, adjust_fun, {'forcing': 'free_convection'})
    # error check on jld2 file
    _ = Data.load(str(jld2_path), jld2_names_mapping, nz=64, dims_mapping=jld2_dims_mapping)
    # other file type check
    with pytest.raises(ValueError):
        _ = Data.load('test.wrong_ext', {})
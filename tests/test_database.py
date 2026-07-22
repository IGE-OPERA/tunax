"""
Tests for the module tunax.database.
"""

from pathlib import Path

import pytest
import jax.numpy as jnp
import numpy as np
from jax import Array
from h5py import File as H5pyFile

from tunax.database import Data, TransAxisParType

from .conf_test import LES_PATH, LES_MAPPING, adjust_fun


LES_JLD2_PATH = Path('tests') / 'data' / 'test_database' / 'les_6h_4m_free_convection.jld2'
"""Path pointing to the 6h, 4m, free convection LES from the database in the .jld2 versions"""

JLD2_LESFC_NAMES_MAPPING: dict[str, dict[str, str]] = {
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
        'omega_p_inv': 'parameters/tracer_forcing_timescale',
        'lambda_c': 'parameters/tracer_forcing_width',
        'zc_m': 'parameters/tracer_forcing_depth',
        'jb': 'parameters/penetrating_buoyancy_flux'
    }
}
"""Mapping of variables for .jld2 file."""

JLD2_TRANS_AXIS_MAPPING_STR_INT: dict[str, TransAxisParType]= {
    'zr': ((True, 0, (64,)),),
    'zw': ((True, 0, ('nz+1',)),),
    'u': ((True, 0, ('nz',)), (False, 0, ()), (False, 0, ()),),
    'v': ((True, 0, ('nz',)), (False, 0, ()), (False, 0, ()),),
    'b': ((True, 0, (64,)), (False, 0, ()), (False, 0, ()),),
    'pt': ((True, 0, (64,)), (False, 0, ()), (False, 0, ()),)
}
"""Correct definition of projection and slicing for the .jld2 file."""

ADJUST_FUN_PARS_JLD2 = {
    'forcing': 'free_convection',
    'lz': -256,
    'eps1': .6,
    'lambda1': 1.,
    'lambda2': 16.
}
"""Parameters to complete the adjusting function (they are not written in the .jld2 file)."""


@pytest.mark.parametrize(
    'arr, trans_axis, expected',
    [
        # no cut, 2D->1D
        (jnp.arange(20).reshape(4, 5), ((True, 0, ()), (False, 2, ())), jnp.array([2, 7, 12, 17])),
        # even number cut, 1D -> 1D
        (jnp.arange(10), ((True, 0, (6,)),), jnp.array([2, 3, 4, 5, 6, 7])),
        # 2 slice cut
        (jnp.arange(10), ((True, 0, (2, 5)),), jnp.array([2, 3, 4])),
        # 3 slice cut
        (jnp.arange(10), ((True, 0, (1, 6, 2)),), jnp.array([1, 3, 5])),
        # 2 dimensions, no cut, transposition
        (
            jnp.arange(20).reshape(4, 5),
            ((True, 1, ()), (True, 0, ())),
            jnp.arange(20).reshape(4, 5).transpose()
        ),
        # transposition + projection
        (
            jnp.arange(60).reshape(3, 4, 5),
            ((True, 1, (),), (False, 2, (),), (True, 0, ())),
            jnp.array([[10, 30, 50], [11, 31, 51], [12, 32, 52], [13, 33, 53], [14, 34, 54]])
        ),
        # transposition + projection + slices
        (
            jnp.arange(60).reshape(3, 4, 5),
            ((True, 1, (1,),), (False, 2, (),), (True, 0, (1, 4))),
            jnp.array([[31], [32], [33]])
        )
    ]
)
def test_datatransform_arr_valid(
        arr: Array,
        trans_axis: TransAxisParType,
        expected: Array
    ) -> None:
    """
    Tests of results for the method Data.transform_arr.
    """
    result = Data.transform_arr(arr, trans_axis)
    np.testing.assert_array_equal(result, expected)

@pytest.mark.parametrize(
    'arr, trans_axis',
    [
        # wrong len of axis
        (jnp.zeros((4, 5)), ((True, 0, ()),)),
        # bounds too large
        (jnp.arange(11), ((True, 0, (12,)),)),
        # wrong input tuple
        (jnp.arange(11), ((True, 0, (12,), 0),))
    ]
)
def test_transform_arr_invalid(
        arr: Array,
        trans_axis: TransAxisParType
    ) -> None:
    """
    Tests of errors for the method Data.transform_arr.
    """
    with pytest.raises(ValueError):
        Data.transform_arr(arr, trans_axis)

def test_datatransform_arr_odd() -> None:
    """
    Test of result and warning for the method Data.transform_arr for an off number slice.
    """
    arr = jnp.arange(11)
    with pytest.warns(UserWarning):
        result = Data.transform_arr(arr, ((True, 0, (6,)),))
        np.testing.assert_array_equal(result, jnp.array([2, 3, 4, 5, 6, 7]))

def test_datatransform_arr_jld2():
    """
    Smoke test and result test for a .jld2 file array with empty dimensions for Data.transform_arr.
    """
    jl = H5pyFile(LES_JLD2_PATH, 'r')
    arr = jnp.array(jl['timeseries/b/0'])
    result = Data.transform_arr(arr, ((True, 0, (64,)), (False, 0, ()), (False, 0, ())))
    assert result.shape == (64,)

def test_datatransform_arr_():
    """
    Smoke test and result test for a scalar from a .jld2 file for Data.transform_arr.
    """
    jl = H5pyFile(LES_JLD2_PATH, 'r')
    arr = jnp.array(jl['parameters/coriolis_parameter'])
    result = Data.transform_arr(arr, ())
    assert result.shape == ()
    assert result == 1e-4

def test_load():
    """
    Test of the automatic setting of passive tracer and eos_tracers.
    """
    data = Data.load(LES_PATH, LES_MAPPING)
    assert data.case.do_pt is True
    assert data.case.eos_tracers == 'b'

def test_load_str_path():
    """
    Smoke test with a path defined by a string.
    """
    _ = Data.load(str(LES_PATH), LES_MAPPING)

def test_load_adjusting_function():
    """
    Smoke test of the usage of an appropriate adjusting function.
    """
    _ = Data.load(LES_PATH, LES_MAPPING, adjust_fun=adjust_fun)

def test_load_jld2_no_slice():
    """
    Smoke test of importating .jld2 without slicing.
    """
    jld2_trans_mapping_no_slice: dict[str, TransAxisParType] = {
        'u': ((True, 0, ()), (False, 0, ()), (False, 0, ()),),
        'v': ((True, 0, ()), (False, 0, ()), (False, 0, ()),),
        'b': ((True, 0, ()), (False, 0, ()), (False, 0, ()),),
        'pt': ((True, 0, ()), (False, 0, ()), (False, 0, ()),)
    }
    _ = Data.load(
        LES_JLD2_PATH, JLD2_LESFC_NAMES_MAPPING, trans_axis_mapping=jld2_trans_mapping_no_slice,
        adjust_fun=adjust_fun, adjust_fun_pars_out=ADJUST_FUN_PARS_JLD2, time_sep=True
    )

def test_load_jld2_slice():
    """
    Smoke and shape test of loading a .jld2 file with slicing.
    """
    data = Data.load(
        LES_JLD2_PATH, JLD2_LESFC_NAMES_MAPPING, trans_axis_mapping=JLD2_TRANS_AXIS_MAPPING_STR_INT,
        adjust_fun=adjust_fun, adjust_fun_pars_out=ADJUST_FUN_PARS_JLD2, time_sep=True
    )
    pt_shape = data.trajectory.pt.shape # pyright: ignore[reportOptionalMemberAccess]
    assert data.trajectory.u.shape == pt_shape
    assert data.trajectory.u.shape[1] == 64
    assert data.trajectory.u.shape[0] == data.trajectory.time.shape[0]
    assert data.trajectory.grid.zr.shape[0] == 64

def test_load_time_sep():
    """
    Error check of importation without time separation a file that separates the time steps.
    """
    with pytest.raises(TypeError):
        _ = Data.load(
            LES_JLD2_PATH, JLD2_LESFC_NAMES_MAPPING,
            trans_axis_mapping=JLD2_TRANS_AXIS_MAPPING_STR_INT, adjust_fun=adjust_fun,
            adjust_fun_pars_out=ADJUST_FUN_PARS_JLD2, time_sep=False
        )

def test_load_other_files():
    """
    Error check on the importation of another file extension.
    """
    with pytest.raises(ValueError):
        _ = Data.load('test.wrong_ext', {})

"""
Geometry and variables of the model.

This module contains the objects that are used in Tunax to describe the geometry of the water column
in :class:`Grid`, the variables of the water column at one time-step in :class:`State` and the time-
series of the model computation in :class:`Trajectory`. These classes can be obtained by the
prefix :code:`tunax.space.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
from collections.abc import Callable
from typing import cast
import warnings

import equinox as eqx
import xarray as xr
import jax.numpy as jnp
from jax import Array, vmap
from jaxtyping import Float

from tunax.functions import _format_to_single_line

type ArrNz = Float[Array, 'nz']
"""Type that describes a float :class:`~jax.Array` of shape (nz)."""
type ArrNzp1 = Float[Array, 'nz+1']
"""Type that describes a float :class:`~jax.Array` of shape (nz+1)."""
type ArrNt = Float[Array, 'nt']
"""Type that describes a float :class:`~jax.Array` of shape (nt)."""
type ArrNtNz = Float[Array, 'nz nt']
"""Type that describes a float :class:`~jax.Array` of shape (nt, nz)."""

TRACERS_NAMES: list[str] = ['t', 's', 'b', 'pt']
"""Names of the tracers, in the order of temperature, salinity, buoyancy and passive tracer."""
VARIABLE_NAMES: list[str] = ['u', 'v'] + TRACERS_NAMES
"""Names of all the variables, zonal and meridionnal velocities in addition of the tracers."""
VARIABLE_SHAPES: dict[str, str] = {
    'u': 'zr',
    'v': 'zr',
    't': 'zr',
    's': 'zr',
    'b': 'zr',
    'pt': 'zr'
}
"""Shapes of all the variables on the water column."""


def _piecewise_linear_ramp(z: float, z0: float, f0: float)-> float:
    r"""
    Mathemacial function used for the state initialisation.

    Apply to z a function linear by part and continuous :
    f(z) = 0 if z < zm and f(z) = f0 (1-z/zm) else.

    Parameters
    ----------
    z : float
        The value where to apply the function.
    zm : float
        The point of connexion of the two linear parts of the function.
    f0 : float
        The value of the function in 0.

    Returns
    -------
    fz : float
        The value of the function in z.
    """
    return f0*(z/-z0+1) * (z>z0)

def _piecewise_linear_flat(z: float, zm: float, f0: float, sl: float) -> float:
    r"""
    Mathemacial function used for the state initialisation.

    Apply to z a function linear by part and continuous :
    f(z) = f0 + s_l (z-zm) if z < z_m and f(z) = f0 else.

    Parameters
    ----------
    z : float
        The value where to apply the function.
    zm : float
        The point of connexion of the two linear parts of the function.
    f0 : float
        The value of the function in 0 and in the right part of the
        funcion.
    sl : float
        The slope of the left part of the function.

    Returns
    -------
    fz : float
        The value of the function in z.
    """
    return f0 + sl*(z-zm) * (z<zm)


class Grid(eqx.Module):
    r"""
    One dimensional spatial geometry of a water column.

    This mesh is made up of a number of :attr:`nz` of cells (:attr:`zr`) of potentially varying
    thickness (:attr:`hz`), separated by interface points (:attr:`zw`) and extending from the ocean
    surface at a depth of :math:`0` to the ocean floor at a depth of :attr:`hbot`.

    Parameters
    ----------
    zr : float :class:`~jax.Array` of shape (nz)
        cf. :attr:`zr`.
    zw : float :class:`~jax.Array` of shape (nz+1)
        cf. :attr:`zw`.

    Attributes
    ----------
    nz : int
        Number of cells.
    hbot : float
        Depth of the water column :math:`[\text m]`.
    zr : float :class:`~jax.Array` of shape (nz)
        Depths of cell centers from deepest to shallowest :math:`[\text m]`
    zw : float :class:`~jax.Array` of shape (nz+1)
        Depths of cell interfaces from deepest to shallowest :math:`[\text m]`.
    hz : float :class:`~jax.Array` of shape (nz)
        Thickness of cells from deepest to shallowest :math:`[\text m]`.

    Note
    ----
    The constructor :code:`__init__` takes only :attr:`zr` and :attr:`zw` as as arguments and
    construct the other attributes from them. The centers of the cells :attr:`zr` are not necessarly
    the middle between the interfaces :attr:`zw` but should be between.

    """

    nz: int = eqx.field(static=True)
    hbot: float
    zr: ArrNz
    zw: ArrNzp1
    hz: ArrNz

    def __init__(self, zr: ArrNz, zw: ArrNzp1) -> None:
        self.nz = zr.shape[0]
        self.hbot = float(zw[0])
        self.zw = zw
        self.zr = zr
        self.hz = zw[1:] - zw[:-1]

    def find_index(self, h: float) -> int:
        r"""
        Find the index of a depth.

        Find the index :code:`i` so that the depth :code:`h` is in cell :code:`i`, which means
        :math:`z^w_i \leqslant -h \leqslant z^w_{i+1}` if :math:`h \leqslant 0` and :math:`i=-1` if
        :math:`h>0`.

        Parameters
        ----------
        h : float, positive
            The depth to search the index :math:`[\text m]`.

        Returns
        -------
        i : int
            The index corresponding to the depth :code:`h`.
        """
        return int(jnp.searchsorted(self.zw, -h, side='right')) - 1

    @classmethod
    def linear(cls, nz: int, hbot: int) -> Grid:
        r"""
        Creates a grid with equal thickness cells.

        The grid instance will have :attr:`nz` cells of equal thickness for a depth of :attr:`hbot`.

        Parameters
        ----------
        nz : int
            Number of cells.
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.

        Returns
        -------
        grid : Grid
            The linear grid.
        """
        zw = jnp.linspace(-hbot, 0, nz+1)
        zr = 0.5*(zw[:-1]+zw[1:])
        return cls(zr, zw)

    @classmethod
    def analytic(cls, nz: int, hbot: float, hc: float, theta: float=6.5) -> Grid:
        r"""
        Creates a grid of type analytic.

        The grid instance will have a depth of :attr:`hbot` and :attr:`nz` cells of thickness almost
        equals above :code:`hc` and wider under, the strecht parameter being defined by
        :code:`theta`.

        Parameters
        ----------
        nz : int
            Number of cells.
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.
        hc : float, positive
            Reference depth :math:`[\text m]`.
        theta : float, default=6.5
            Stretching parameter toward the surface :math:`[\text{dimensionless}]`.

        Returns
        -------
        grid : Grid
            The analytic grid.
        """
        sc_w = jnp.linspace(-1, 0, nz+1)
        sc_r = (sc_w[:-1] + sc_w[1:])/2
        cs_r = (1-jnp.cosh(theta*sc_r))/(jnp.cosh(theta)-1)
        cs_w = (1-jnp.cosh(theta*sc_w))/(jnp.cosh(theta)-1)
        zw = (hc*sc_w + hbot*cs_w) * hbot/(hbot+hc)
        zr = (hc*sc_r + hbot*cs_r) * hbot/(hbot+hc)
        return cls(zr, zw)

    @classmethod
    def orca75(cls, hbot: float) -> Grid:
        r"""
        Creates the ORCA75 grid from NEMO.

        The whole grid is created then levels between the depth :attr:`hbot` and :math:`0` are
        extracted.

        Parameters
        ----------
        hbot : float, positive
            Depth of the water column :math:`[\text m]`.

        Returns
        -------
        grid : Grid
            The ORCA75 grid.
        """
        nz_orca = 75
        zsur = -3958.95137127683
        za2 = 100.7609285
        za0 = 103.9530096
        za1 = 2.415951269
        zkth = 15.3510137
        zkth2 = 48.02989372
        zacr = 7.
        zacr2 = 13.
        sc_r = jnp.arange(nz_orca-0.5, 0.5, -1)
        sc_w = jnp.arange(nz_orca, 0, -1)
        zw_orca = -(zsur + za0*sc_w + za1*zacr*jnp.log(jnp.cosh((sc_w-zkth)/zacr)) +\
                    za2*zacr2*jnp.log(jnp.cosh((sc_w-zkth2)/zacr2)))
        zr_orca = -(zsur + za0*sc_r + za1*zacr*jnp.log(jnp.cosh((sc_r-zkth)/zacr)) +\
                    za2*zacr2*jnp.log(jnp.cosh((sc_r-zkth2)/zacr2)))
        ibot = jnp.argmin(zw_orca <= -hbot)
        if ibot == 0:
            ibot = 1
        zw_orca = zw_orca.at[-1].set(0.)
        zw = zw_orca[ibot-1:]
        zr = zr_orca[ibot-1:]
        return cls(zr, zw)

    @classmethod
    def load(cls, ds: xr.Dataset) -> Grid:
        """
        Creates the grid defined by a dataset :code:`ds` of an observation.

        The dataset must be formated to have the variables corresponding to :attr:`zr` and
        :attr:`zw`.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset from which to extract the grid.

        Returns
        -------
        grid : Grid
            The loaded grid.
        """
        zw = jnp.array(ds['zw'], dtype=jnp.float32)
        zr = jnp.array(ds['zr'], dtype=jnp.float32)
        return cls(zr, zw)


class State(eqx.Module):
    r"""
    Water column state at one time-step.

    This state is defined on a :attr:`grid` describing the geometry, and is composed of the
    variables of the water column : the values of the momentum variables :attr:`u` and :attr:`v`
    (which are mandatory) and the tracers variables :attr:`t`, :attr:`s`, :attr:`b` and :attr:`pt`
    (which are optionals). The constructor takes all the attributes as parameters.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column.
    u : float :class:`~jax.Array` of shape (nz)
        Zonal velocity on the center of the cells :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v : float :class:`~jax.Array` of shape (nz)
        Meridional velocity on the center of the cells :math:`\left[\text m \cdot
        \text s^{-1}\right]`.
    t : float :class:`~jax.Array` of shape (nz), optionnal
        Temperature on the center of the cells :math:`[° \text C]`.
    s : float :class:`~jax.Array` of shape (nz), optionnal
        Salinity on the center of the cells :math:`[\text{psu}]`.
    b : float :class:`~jax.Array` of shape (nz), optionnal
        Buoyancy on the center of the cells :math:`[\text{dimensionless}]`.
    pt : float :class:`~jax.Array` of shape (nz), optionnal
        A passive tracer on the center of the cells :math:`[\text{dimensionless}]`.

    """

    grid: Grid
    u: ArrNz
    v: ArrNz
    t: ArrNz | None = None
    s: ArrNz | None = None
    b: ArrNz | None = None
    pt: ArrNz | None = None

    @classmethod
    def zeros(cls, grid: Grid, tracers: list[str]) -> State:
        """
        Initialize an instance with all variables equals to zero from a grid.

        Parameters
        ----------
        grid : Grid
            Geometry of the water column.

        Returns
        -------
        state : State
            An instance defined on the grid with all variables set to 0.        
        """
        zero_array = jnp.zeros(grid.nz)
        tracers_dict = {}
        for tracer_name in tracers:
            tracers_dict[tracer_name] = zero_array
        return State(grid, u=zero_array, v=zero_array, **tracers_dict)

    def init_u(self, hmxl: float=20., u_sfc: float=0.) -> State:
        r"""
        Initialize zonal velocity with a classical wind stratification.

        Return a State object where :attr:`u` is continuous and linear by part
        :math:`u(z) = \begin{cases}
        0 & \text{if } z < h_{\text{mxl}}\\
        u_{\text{sfc}} \left( 1 - \dfrac z {h_{\text{mxl}}}\right) &
        \text{else} \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        u_sfc : float, default=0.
            Surface zonal velocity :math:`\left[\text m \cdot \text s^{-1}\right]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of zonal velocity.
        """
        maped_fun = vmap(_piecewise_linear_ramp, in_axes=(0, None, None))
        maped_fun = cast(Callable[[ArrNz, float, float], ArrNz], maped_fun)
        u_new = maped_fun(self.grid.zr, -hmxl, u_sfc)
        return eqx.tree_at(lambda t: t.u, self, u_new)

    def init_v(self, hmxl: float=20., v_sfc: float=0.) -> State:
        r"""
        Initialize meridional velocity with a classical wind stratification.

        Return a State object where :attr:`v` is continuous and linear by part
        :math:`v(z) = \begin{cases}
        0 & \text{if } z < h_{\text{mxl}}\\
        v_{\text{sfc}} \left( 1 - \dfrac z {h_{\text{mxl}}}\right) &
        \text{else} \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        u_sfc : float, default=0.
            Surface meridional velocity :math:`\left[\text m \cdot \text s^{-1}\right]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of meridional velocity.
        """
        maped_fun = vmap(_piecewise_linear_ramp, in_axes=(0, None, None))
        maped_fun = cast(Callable[[ArrNz, float, float], ArrNz], maped_fun)
        v_new = maped_fun(self.grid.zr, -hmxl, v_sfc)
        return eqx.tree_at(lambda t: t.v, self, v_new)

    def init_t(self, hmxl: float=20., t_sfc: float=21., strat_t: float=5.1e-2) -> State:
        r"""
        Initialize temperature with a classical tracer stratification.

        Return a State object where :attr:`t` is linear by part and continous
        :math:`T(z) = \begin{cases}
        t_{\text{sfc}} + S_T(z-h_{\text{mxl}}) & \text{if } z <
        h_{\text{mxl}}\\
        t_{\text{sfc}} & \text{else}
        \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        t_sfc : float, default=21.
            Surface temperature :math:`[° \text C]`.
        strat_t : float, default=5.1e-2
            Thermal stratification above the mixed layer noted by :math:`S_T`
            :math:`[\text K \cdot \text m ^{-1}]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of temperature.
        """
        maped_fun = vmap(_piecewise_linear_flat, in_axes=(0, None, None, None))
        maped_fun = cast(Callable[[ArrNz, float, float, float], ArrNz], maped_fun)
        t_new = maped_fun(self.grid.zr, -hmxl, t_sfc, strat_t)
        return eqx.tree_at(lambda tree: tree.t, self, t_new)

    def init_s(self, hmxl: float=20., s_sfc: float=35., strat_s: float=1.3e-2) -> State:
        r"""
        Initialize salinity with a classical tracer stratification.

        Return a State object where :attr:`s` is linear by part and continous
        :math:`S(z) = \begin{cases}
        s_{\text{sfc}} + S_S(z-h_{\text{mxl}}) & \text{if } z <
        h_{\text{mxl}}\\
        s_{\text{sfc}} & \text{else}
        \end{cases}`

        Parameters
        ----------
        hmxl : float, default=20.
            Mixed layer depth :math:`[\text m]`.
        s_sfc : float, default=21.
            Surface salinity :math:`[\text{psu}]`.
        strat_s : float, default=5.1e-2
            Salinity stratification above the mixed layer noted by :math:`S_T`
            :math:`[\text{psu} \cdot \text m ^{-1}]`.

        Returns
        -------
        state : State
            The :code:`self` object with the the new value of temperature.
        """
        maped_fun = vmap(_piecewise_linear_flat, in_axes=(0, None, None, None))
        maped_fun = cast(Callable[[ArrNz, float, float, float], ArrNz], maped_fun)
        s_new = maped_fun(self.grid.zr, -hmxl, s_sfc, strat_s)
        return eqx.tree_at(lambda t: t.s, self, s_new)


class Trajectory(eqx.Module):
    r"""
    Define the history of a simulation or an observation.

    Contains the timeseries of the momentum (mandatory) variables and the tracers variables
    (optionals) throught the space of the :attr:`grid` and the :attr:`time`. The constructor takes
    all the attributes as parameters.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column.
    time : float :class:`~jax.Array` of shape (nt)
        Time at each steps of observation from the begining of the simulation :math:`[\text s]`.
    u : float :class:`~jax.Array` of shape (nt, nz)
        Time-serie of zonal velocity :math:`\left[\text m \cdot \text s^{-1}\right]`.
    v : float :class:`~jax.Array` of shape (nt, nz)
        Time-serie of meridional velocity :math:`\left[\text m \cdot \text s^{-1}\right]`.
    t : float :class:`~jax.Array` of shape (nt, nz), optionnal
        Time-serie of temperature :math:`[\text C°]`.
    s : float :class:`~jax.Array` of shape (nt, nz), optionnal
        Time-serie of salinity :math:`[\text{psu}]`.
    b : float :class:`~jax.Array` of shape (nt, nz), optionnal
        Time-serie of buoyancy :math:`[\text{dimensionless}]`.
    pt : float :class:`~jax.Array` of shape (nt, nz), optionnal
        Time-serie a passive tracer :math:`[\text{dimensionless}]`.
        
    """

    grid: Grid
    time: Float[Array, 'nt']
    u: ArrNtNz
    v: ArrNtNz
    t: ArrNtNz | None = None
    s: ArrNtNz | None = None
    b: ArrNtNz | None = None
    pt: ArrNtNz | None = None

    def to_ds(self) -> xr.Dataset:
        """
        Exports the trajectory in an xarray.Dataset.

        The dimensions of the dataset are :attr:`time`, :code:`grid.zr` and :code:`grid.zw`, the
        variables are :attr:`u`, :attr:`v` and the tracers that are not set to :code:`None`, all
        defined on the dimensions (:attr:`time`, :code:`zr`) or :code:`zw` depending on
        :data:`VARIABLE_NAMES`.

        Returns
        -------
        ds : xarray.Dataset
            Dataset of the trajectory.
        """
        variables = {}
        for var_name in VARIABLE_NAMES:
            var = getattr(self, var_name)
            if var is not None:
                variables[var_name] = (('time', VARIABLE_SHAPES[var_name]), var)
        coords = {
            'time': self.time,
            'zr': self.grid.zr,
            'zw': self.grid.zw
        }
        return xr.Dataset(variables, coords)

    def to_nc(self, nc_path: str) -> None:
        r"""
        Write on a NetCDF file.

        The dimensions are :attr:`time`, :code:`grid.zr` and :code:`grid.zw`, the variables are
        :attr:`u`, :attr:`v` and the tracers that are not set to :code:`None`, all defined on the
        dimensions (:attr:`time`, :code:`zr`) or :code:`zw` depending on :data:`VARIABLE_NAMES`.
        
        Parameters
        ----------
        nc_path : str
            Path of the file on which write the trajectory.
        """
        ds = self.to_ds()
        ds.to_netcdf(nc_path)

    def extract_state(self, i_time: int) -> State:
        """
        Extracts the water column state at one time index.

        Parameters
        ----------
        i_time : int
            The time index of the moment to extract the state.

        Returns
        -------
        state : State
            The state of the trajectory at the time of index :code:`i_time`.
        """
        variables = {}
        for var_name in VARIABLE_NAMES:
            var = getattr(self, var_name)
            if var is not None:
                variables[var_name] = var[i_time, :]
        return State(self.grid, **variables)

    def cut(self, out_nt_cut: int) -> list[Trajectory]:
        """
        Cut the trajectory in sub-trajectories of :code:`out_nt_cut` output steps.

        The first and last state of two consecutive trajectories are the same. :code:`out_nt_cut`
        is the number of output steps, it means that the time dimension of the sub-trajectories
        have :code:`out_nt_cut+1` elements.

        Parameters
        ----------
        out_nt_cut : int
            Number of output steps of the sub-trajectories.
        
        Returns
        -------
        traj_list : list[Trajectory]
            List of the sub-trajectories in the chronological order.

        Warns
        -----
        Lost last trajectory
            If :code:`out_nt_cut` does not divide the number of output step of the initial
            trajectory. In this case the last part of the trajectory (which is too short) is
            abandonned.
        """
        out_nt = self.time.shape[0] - 1
        if out_nt%out_nt_cut != 0:
            warnings.warn(_format_to_single_line("""
                If out_nt_cut does not divide the number of output step of the initial trajectory.
                In this case the last part of the trajectory (which is too short) is abandonned.
            """))
        traj_list = []
        n_cuts = out_nt//out_nt_cut
        for i_cut in range(n_cuts):
            i_start, i_end = out_nt_cut*i_cut, out_nt_cut*(i_cut+1) + 1
            cut_time = self.time[i_start:i_end]
            var_dict_cut = {}
            for var in VARIABLE_NAMES:
                if getattr(self, var) is not None:
                    var_dict_cut[var] = getattr(self, var)[i_start:i_end, :]
            cut_traj = Trajectory(self.grid, cut_time, **var_dict_cut)
            traj_list.append(cut_traj)
        return traj_list

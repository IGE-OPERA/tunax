"""
Physical parameters and forcings.

This module contains the :class:`Case` class which is the one used to describe the parameters and
the forcings of a model. It also contains the :class:`CaseTracable` which is a variation of the
first classe which works better with JAX specificities. This classes can be obtained by the prefix
:code:`tunax.case.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
from typing import Union, Tuple, Callable, Optional, TypeAlias
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
from jax import device_get

from tunax.functions import add_boundaries
from tunax.space import Grid, ArrNz, ArrNtNz

ForcingType: TypeAlias = Union[
    Tuple[float, float],
    Callable[[float], float],
    Callable[[float, float], float]
]
"""Type that represent the different possible types of the forcings in :class:`Case`."""
ForcingArrayType: TypeAlias = Union[Tuple[float, float], ArrNz, ArrNtNz]
"""Type that represent the different possible types of the forcings in :class:`CaseTracable`."""

_OMEGA = 7.292116e-05
"""Rotation rate of the Earth [rad.s-1]."""
_RAD_DEG = jnp.pi/180.
"""Measure of one degree in radiant [rad.°-1]."""


class Case(eqx.Module):
    r"""
    Physical parameters and forcings.

    This class contains all the physical constants, and the constant forcings that definine an
    experience for the model. The forcings can be described as functions. The transformations from
    :class:`Case` to :class:`CaseTracable` is done when a model instance is created. The constructor
    takes all the attributes as parameters.

    Attributes
    ----------
    rho0 : float, default=1024.
        Default density of saltwater :math:`[\text{kg} \cdot \text{m}^{-3}]`.
    grav : float, default=9.81
        Gravity acceleration :math:`[\text{m} \cdot \text{s}^{-2}]`.
    cp : float, default=3985.
        Specific heat capacity of saltwater
        :math:`[\text{J} \cdot \text{kg}^{-1} \cdot \text{K}^{-1}]`.
    eos_tracers : str, default='t'
        Tracers used for the equation of state (the computation of the density). One of
        {:code:`'t'`, :code:`'s'`, :code:`'ts`', :code:`'b'`}.
        - 't': temperature
        - 's': salinity
        - 'ts': temperature and salinity
        - 'b': buoyancy
    alpha : float, default=2e-4
        Thermal expansion coefficient :math:`[\text{K}^{-1}]`.
    beta : float, default=8e-4
        Salinity expansion coefficient :math:`[\text{psu}^{-1}]`.
    t_rho_ref : float, default=0.
        Reference temperature for the equation of state (the computation of the density)
        :math:`[° \text C]`.
    s_rho_ref : float, default=35.
        Reference salinity for the equation of state (the computation of the density)
        :math:`[\text{psu}]`.
    do_pt : bool, default=False
        Compute or not a passive tracer.
    vkarmn : float, default=0.384
        Von Kármán constant :math:`[\text{dimensionless}]`.
    fcor : float, default=0.
        Coriolis frequency at the water column :math:`[\text{rad} \cdot \text{s}^{-1}]`.
    ustr_sfc : float, default=0.
        Zonal wind stress :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    ustr_btm : float, default=0.
        Zonal current stress at the bottom :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    vstr_sfc : float, default=0.
        Meridional wind stress :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    vstr_btm : float, default=0.
        Meridional current stress at the bottom :math:`[\text{m}^{2} \cdot \text{s}^{-2}]`.
    t_forcing : tuple of 2 floats or a function, optionnal, default=None
        Description of the forcing of temperature (potentially no forcing if the variable is not
        activated i.e. if :code:`'t'` is not in :code:`eos_tracers`). There are 3 cases :

        - **Border forcing** : tuple of 2 floats, the first one is the forcing at the bottom and the
          second ont is the forcing at the top of the water column, the unit is in
          :math:`[\text{K} \cdot \text{m} \cdot \text{s}^{-1}]`.
        
        - **Deep constant forcing** : function of signature float->float, the parameter is the depth
          and the ouput is the value of the forcing at this depth in
          :math:`[\text{K} \cdot \text{s}^{-1}]`. The values of the functions represent the flux of
          the forcing (the derivative along the depth).)

        - **Deep variable forcing** : function of signature (float, float)->float, the parameters
          are the depth and the time and the ouput is the value of the forcing at this depth and
          this time in :math:`[\text{K} \cdot \text{s}^{-1}]`. The values of the functions represent
          the flux of the forcing (the derivative along the depth).

    s_forcing : tuple of 2 floats or a function, optionnal, default=None
        Description of the forcing of salinity (potentially no forcing if the variable is not
        activated i.e. if :code:`'s'` is not in :code:`eos_tracers`). There are the 3 same cases as
        for the :code:`t_forcing`, and the units are
        :math:`[\text{psu} \cdot \text{m} \cdot \text{s}^{-1}]` for the border forcing and
        :math:`[\text{psu} \cdot \text{s}^{-1}]` for the other ones.
    b_forcing : tuple of 2 floats or a function, optionnal, default=None
        Description of the forcing of buoyancy (potentially no forcing if the variable is not
        activated i.e. if :code:`'b'` is not in :code:`eos_tracers`). There are the 3 same cases as
        for the :code:`t_forcing`, and the units are
        :math:`[\text{m} \cdot \text{s}^{-1}]` for the border forcing and
        :math:`[\text{s}^{-1}]` for the other ones.
    pt_forcing : tuple of 2 floats or a function, optionnal, default=None
        Description of the forcing of passive tracer (potentially no forcing if the variable is not
        activated i.e. if :code:`do_pt` is not set). There are the 3 same cases as for the
        :code:`t_forcing`, and the units are :math:`[\text{m} \cdot \text{s}^{-1}]`
        for the border forcing and :math:`[\text{s}^{-1}]` for the other ones.

    """

    # physcal constants
    rho0: float = 1024.
    grav: float = 9.81
    cp: float = 3985.
    eos_tracers: str = eqx.field(default='t', static=True)
    alpha: float = 2e-4
    beta: float = 8e-4
    t_rho_ref: float = 0.
    s_rho_ref: float = 35.
    do_pt: bool = eqx.field(default=False, static=True)
    vkarmn: float = 0.384
    # dynamic forcings
    fcor: float = 0.
    ustr_sfc: float = 0.
    ustr_btm: float = 0.
    vstr_sfc: float = 0.
    vstr_btm: float = 0.
    # tracers forcings
    t_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    s_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    b_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)
    pt_forcing: Optional[ForcingType] = eqx.field(default=None, static=True)

    def set_lat(self, lat: float) -> Case:
        """
        Set the Coriolis frequency from the latitude.

        Parameters
        ----------
        lat : float
            Latitude of the water column :math:`[°]`.
        
        Returns
        -------
        case : Case
            The :code:`self` object with the the new value of :attr:`fcor`.
        """
        fcor = float(device_get(2.*_OMEGA*jnp.sin(_RAD_DEG*lat)))
        case = eqx.tree_at(lambda t: t.fcor, self, fcor)
        return case


class CaseTracable(eqx.Module):
    r"""
    Physical parameters and forcings tracable by JAX.

    This class is similar to the :class:`Case` one, but the function forcings are transformed in
    arrays so that the class is tracable by JAX which means that we can use :func:`~jax.jit` and
    :func:`~jax.grad` more freely. The constructor takes all the attributes as parameters.

    Attributes
    ----------
    rho0 : float, default=1024.
        cf. :attr:`Case.rho0`
    grav : float, default=9.81
        cf. :attr:`Case.grav`
    cp : float, default=3985.
        cf. :attr:`Case.cp`
    eos_tracers : str, default='t'
        cf. :attr:`Case.eos_tracers`
    alpha : float, default=2e-4
        cf. :attr:`Case.alpha`
    beta : float, default=8e-4
        cf. :attr:`Case.beta`
    t_rho_ref : float, default=0.
        cf. :attr:`Case.t_rho_ref`
    s_rho_ref : float, default=35.
        cf. :attr:`Case.s_rho_ref`
    do_pt : bool, default=False
        cf. :attr:`Case.do_pt`
    vkarmn : float, default=0.384
        cf. :attr:`Case.vkarmn`
    fcor : float, default=0.
        cf. :attr:`Case.fcor`
    ustr_sfc : float, default=0.
        cf. :attr:`Case.ustr_sfc`
    ustr_btm : float, default=0.
        cf. :attr:`Case.ustr_btm`
    vstr_sfc : float, default=0.
        cf. :attr:`Case.vstr_sfc`
    vstr_btm : float, default=0.
        cf. :attr:`Case.vstr_btm`
    t_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal, default=None
        Description of the temperature forcing cf. :attr:`Case.t_forcing`, the type depends on the
        forcing type :
        
        - **Border forcing** : tuple of 2 floats, the first one is the forcing at the bottom and the
          second one is the forcing at the top of the water column, the unit is in
          :math:`[\text{K} \cdot \text{m} \cdot \text{s}^{-1}]`.
        
        - **Deep constant forcing** : array of shape (nz) : the value of the forcing function on the
          geometrical :class:`Grid` of the model. The values represent the forcing flux, which is
          for each cell the difference between the forcing at the top of the cell and the forcing at
          bottom.
        
        - **Deep variable forcing** : array of shape (nt, nz) : the value of the forcing function on
          the geometrical :class:`Grid` and the different iteration times of the model. As for deep
          constant forcing, the values represent the flux of the forcing at every time.

    s_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal, default=None
        Same as :attr:`t_forcing` for Salinity.
    b_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal, default=None
        Same as :attr:`t_forcing` for buoyancy.
    pt_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal, default=None
        Same as :attr:`t_forcing` for passive tracer.
    t_forcing_type : str, optionnal, default=None
        Description of the type of temperature forcing : :code:`'borders'` for **border forcing**,
        :code:`'constant'` for **deep constant forcing** and :code:`'variable'` for **deep variable
        forcing**
    s_forcing_type : str, optionnal, default=None
        Same as :attr:`t_forcing_type` for salinity.
    b_forcing_type : str, optionnal, default=None
        Same as :attr:`t_forcing_type` for buoyancy.
    pt_forcing_type : str, optionnal, default=None
        Same as :attr:`t_forcing_type` for passive tracer.

    """

    # physcal constants
    rho0: float = 1024.
    grav: float = 9.81
    cp: float = 3985.
    eos_tracers: str = eqx.field(default='t', static=True)
    alpha: float = 2e-4
    beta: float = 8e-4
    t_rho_ref: float = 0.
    s_rho_ref: float = 35.
    do_pt: bool = eqx.field(default=False, static=True)
    vkarmn: float = 0.384
    # dynamic forcings
    fcor: float = 0.
    ustr_sfc: float = 0.
    ustr_btm: float = 0.
    vstr_sfc: float = 0.
    vstr_btm: float = 0.
    # tracers forcings
    t_forcing: Optional[ForcingArrayType] = None
    s_forcing: Optional[ForcingArrayType] = None
    b_forcing: Optional[ForcingArrayType] = None
    pt_forcing: Optional[ForcingArrayType] = None
    t_forcing_type: Optional[str] = eqx.field(default=None, static=True)
    s_forcing_type: Optional[str] = eqx.field(default=None, static=True)
    b_forcing_type: Optional[str] = eqx.field(default=None, static=True)
    pt_forcing_type: Optional[str] = eqx.field(default=None, static=True)

    def tra_promote_borders_constant(self, tra: str, grid: Grid) -> CaseTracable:
        """
        Promote the dimension of a forcing from borders to constant.

        It can be use to apply :func:`~jax.vmap` on :class:`SingleColumnModel` for batch computing.
        The input :class:`CaseTracable` instance should have a borders forcing type and the ouput
        instance will have deep constant forcing type.
        
        Parameters
        ----------
        tra : str
            Name of the tracer variable of the concerned forcing. One of {:code:`'t'`, :code:`'s'`,
            :code:`'b`', :code:`'pt`'}.
        grid : Grid
            Spatial grid to compute the constant forcing on.
        
        Returns
        -------
        case_tracable : CaseTracable
            The :code:`self` object with the promoted forcing.
        """
        border_forcing = getattr(self, f'{tra}_forcing')
        constant_forcing = add_boundaries(
            -border_forcing[0], jnp.zeros(grid.nz-2), border_forcing[1]
        )
        case_tracable = eqx.tree_at(lambda t: getattr(t, f'{tra}_forcing'), self, constant_forcing)
        dico = {f'{tra}_forcing_type': 'constant'}
        case_tracable = replace(case_tracable, **dico)
        return case_tracable

    def tra_promote_borders_variable(self, tra: str, grid: Grid, nt: int) -> CaseTracable:
        """
        Promote the dimension of a forcing from borders to variable.

        It can be use to apply :func:`~jax.vmap` on :class:`SingleColumnModel` for batch computing.
        The input :class:`CaseTracable` instance should have a borders forcing type and the ouput
        instance will have deep variable forcing type.
        
        Parameters
        ----------
        tra : str
            Name of the tracer variable of the concerned forcing. One of {:code:`'t'`, :code:`'s'`,
            :code:`'b`', :code:`'pt`'}.
        grid : Grid
            Spatial grid to compute the constant forcing on.
        nt : int
            Number of integration iterations of the model.
        
        Returns
        -------
        case_tracable : CaseTracable
            The :code:`self` object with the promoted forcing.
        """
        border_forcing = getattr(self, f'{tra}_forcing')
        constant_forcing = add_boundaries(
            -border_forcing[0], jnp.zeros(grid.nz-2), border_forcing[1]
        )
        variable_forcing = jnp.tile(constant_forcing, (nt, 1))
        case_tracable = eqx.tree_at(lambda t: getattr(t, f'{tra}_forcing'), self, variable_forcing)
        dico = {f'{tra}_forcing_type': 'variable'}
        case_tracable = replace(case_tracable, **dico)
        return self

    def tra_promote_constant_variable(self, tra: str, nt: int) -> CaseTracable:
        """
        Promote the dimension of a forcing from borders to constant.

        It can be use to apply :func:`~jax.vmap` on :class:`~model.SingleColumnModel` for batch
        computing. The input :class:`CaseTracable` instance should have a constant forcing type and
        the ouput instance will have deep variable forcing type.
        
        Parameters
        ----------
        tra : str
            Name of the tracer variable of the concerned forcing. One of {:code:`'t'`, :code:`'s'`,
            :code:`'b'`, :code:`'pt`'}.
        grid : Grid
            Spatial grid to compute the constant forcing on.
        nt : int
            Number of integration iterations of the model.
        
        Returns
        -------
        case_tracable : CaseTracable
            The :code:`self` object with the promoted forcing.
        """
        constant_forcing = getattr(self, f'{tra}_forcing')
        variable_forcing = jnp.tile(constant_forcing, (nt, 1))
        case_tracable = eqx.tree_at(lambda t: getattr(t, f'{tra}_forcing'), self, variable_forcing)
        dico = {f'{tra}_forcing_type': 'variable'}
        case_tracable = replace(case_tracable, **dico)
        return self

"""
Physical parameters and forcings.

This module contains the :class:`Case` class which is the one used to describe the parameters and
the forcings of a model. It also contains the :class:`CaseTracable` which is a variation of the
first classe which works better with JAX specificities. This classes can be obtained by the prefix
:code:`tunax.case.` or directly by :code:`tunax.`.

"""

from __future__ import annotations
from collections.abc import Callable
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
from jax import device_get

from tunax.functions import FloatJax, add_boundaries
from tunax.space import Grid, ArrNz, ArrNtNz

type ForcingType = (
    float |
    tuple[float, float] |
    Callable[[float], float] |
    Callable[[float, float], float]
)
"""Type that represent the different possible types of the forcings in :class:`Case`."""

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
    t_forcing : float, tuple 2 floats or a function, optionnal
        Description of the forcing of temperature (potentially no forcing if the variable is not
        activated i.e. if :code:`'t'` is not in :code:`eos_tracers`). The unit is in
        :math:`[\text{K} \cdot \text{s}^{-1}]`. There are 4 cases depending on the
        object type :
        - **Surface forcing** : a float, forcing flux at the top of the water column.
        - **Border forcing** : tuple of 2 floats, the first one is the forcing at the bottom and the
          second ont is the forcing at the top of the water column.
        - **Deep constant forcing** : function of signature float->float, the parameter is the depth
          and the ouput is the value of the forcing flux at this depth.
        - **Deep variable forcing** : function of signature (float, float)->float, the parameters
          are the depth and the time and the ouput is the value of the forcing at this depth and
          this time.
    s_forcing : float, tuple 2 floats or a function, optionnal
        Salinity forcing, same description than :code:`t_forcing`. The unit is in
        :math:`[\text{psu} \cdot \text{s}^{-1}]`
    b_forcing : float, tuple 2 floats or a function, optionnal
        Buoyancy forcing, same description than :code:`t_forcing`. The unit is in
        :math:`[\text{s}^{-1}]`.
    pt_forcing : float, tuple 2 floats or a function, optionnal
        Passive tracer forcing, same description than :code:`t_forcing`. The unit is in
        :math:`[\text{s}^{-1}]`.

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
    t_forcing: ForcingType | None = None
    s_forcing: ForcingType | None = None
    b_forcing: ForcingType | None = None
    pt_forcing: ForcingType | None = None

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
    t_forcing : :class:`~jax.Array` of shape (nz) or (nt, nz), optionnal
        Description of the temperature forcing cf. :attr:`Case.t_forcing` as a forcing flux. The
        shape of the array depends on the forcing type :
        - **constant forcing** : array of shape (nz) : the value of the forcing function on the
          geometrical :class:`Grid` of the model.
        - **Variable forcing** : array of shape (nt, nz) : the value of the forcing function on
          the geometrical :class:`Grid` and the different iteration times of the model.
        The surface, borders and "deep constant forcing" from the class :class:`Case` are
        transformed in "constant forcing" array. And the space-time function forcing for the class
        :class:`Case` are transformed in "variable forcing".
    s_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal
        Same as :attr:`t_forcing` for Salinity.
    b_forcing : tuple of 2 floats or :class:`~jax.Array` (nz) or (nt, nz), optionnal
        Same as :attr:`t_forcing` for buoyancy.
    pt_forcing : :class:`~jax.Array` of shape (nz) or (nt, nz), optionnal
        Same as :attr:`t_forcing` for passive tracer.
    t_for_var : bool, default = False
        Is the temperature forcing variable with time (is the shape of :attr:`t_forcing` (nt, nz) or
        (nz,)).
    s_for_var : bool, default = False
        Same as :attr:`t_for_var` for salinity.
    b_for_var : bool, default = False
        Same as :attr:`t_for_var` for buoyancy.
    pt_for_var : bool, default = False
        Same as :attr:`t_for_var` for passive tracer.

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
    t_forcing: ArrNz | ArrNtNz | None = None
    s_forcing: ArrNz | ArrNtNz | None = None
    b_forcing: ArrNz | ArrNtNz | None = None
    pt_forcing: ArrNz | ArrNtNz | None = None
    t_for_var: bool = eqx.field(default=False, static=True)
    s_for_var: bool = eqx.field(default=False, static=True)
    b_for_var: bool = eqx.field(default=False, static=True)
    pt_for_var: bool = eqx.field(default=False, static=True)

    def tra_promote_constant_variable(self, tra: str, nt: int) -> CaseTracable:
        """
        Promote the dimension of a forcing from constant to variable.
        
        Parameters
        ----------
        tra : str, {:code:`'t'`, :code:`'s'`, :code:`'b'`, :code:`'pt`'}
            Name of the tracer variable of the concerned forcing.
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
        case_tracable = replace(case_tracable, {f'{tra}_for_var': True})
        return case_tracable

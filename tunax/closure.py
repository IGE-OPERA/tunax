"""
Abstractions for defining closures.

A closure is a a set of physical equations which determines the sub-mesh turbulence of the water
column at a specific time. The purpose of Tunax is to calibrate the parameters of these equations.
This module contains the abstract classes required to define a closure, which is done in the folder
:code:`closures/`. These classes can be obtained by the prefix :code:`tunax.closure.` or directly by
:code:`tunax.`.
"""

from abc import ABC
from collections.abc import Callable

import equinox as eqx

from tunax.case import CaseTracable
from tunax.space import Grid, State, ArrNzp1


class ClosureParametersAbstract(eqx.Module, ABC):
    """
    Abstraction for the parameters of a closure.
    
    To define a closure, a class that inherits from this one must be created. This parent class does
    not impose anything on the child one. The child class should be fill of the parameters involved
    in the closure which may be calibrated. The child class must be the only place where these
    parameters are defined for running a forward model : the instance of this class is given to the
    closure functions to recover the parameters and do the computation. For a run of a calibration,
    the description of the parameters to calibration is done from the child class. It can also
    includes parameters that are not dedicated to be calibrated such as physical or mathematical
    constants.
    
    """


class ClosureStateAbstract(eqx.Module, ABC):
    r"""
    Abstraction for the water column state linked to the closure.

    To define a closure, a class that inherits rom this one must be created. This parent class
    imposes a grid and three variables as attributes : these variables are used at each step of the
    forward model to compute the next step of tracers and momentum. The child class should be fill
    with other arrays and scalars that may evolve throught time and necessary for the computation of
    the closure at the next step. The child class is similary to the :class:`~space.State` class but
    for the diffusivity part. The :code:`__init__` method can be overwrited to fill the variables
    with initial values. The constructor takes a grid a set of the closure parameters to initialize
    the state.

    Parameters
    ----------
    grid : Grid
        cf. :attr:`grid`.
    closure_parameters : ClosureParametersAbstract
        A set of parameters of the closure used to initialize the state.

    Attributes
    ----------
    grid : Grid
        Geometry of the water column, should be the same than for the :class:`~space.State` instance
        used in the model.
    akt : float :class:`~jax.Array` of shape (nz+1)
        Eddy-diffusivity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    akv : float :class:`~jax.Array` of shape (nz+1)
        Eddy-viscosity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.

    """

    grid: Grid
    akt: ArrNzp1
    akv: ArrNzp1

    def __init__(self, grid: Grid, closure_parameters: ClosureParametersAbstract):
        pass


class Closure[CloStateT: ClosureStateAbstract, CloParT: ClosureParametersAbstract](eqx.Module):
    r"""
    Implementation of a physical closure for computing eddy-diffusivity.

    This class contains the three parts that defines a closure of a vertical physics. These parts
    will be used in the model to compute the eddy-diffusivity and eddy-viscosity at each time-step
    from the current water column :class:`~space.State`, the current water column state of the
    closure (child class of :class:`~ClosureStateAbstract`), the physical
    :class:`~case.CaseTracable` and the closure parameters (child class of
    :class:`~ClosureParametersAbstract`). The constructor takes all the attributes as parameters.

    Attributes
    ----------
    name : str
        The name of the closure.
    parameters_class : type[ClosureParametersAbstract]
        A child class of :class:`~ClosureParametersAbstract` that defines the constant parameters
        used in the computation done by the closure, it includes the parameters that may be
        calibrated.
    state_class : type[ClosureStateAbstract]
        A child class of :class:`~ClosureStateAbstract` that defines the state of the water column
        for the variables used by the closure computation.
    step_fun : Callable[[State, CloStateT, float, CloParT, CaseTracable], CloStateT]
        Ths function is called at every step of the forward model to compute the eddy-diffusivity
        before resolving the equation of the tracers and of the momentum.
        
        Parameters
        ----------
        state : State
            Current state of the water column.
        closure_state : CloStateT
            Current state of the water column for the variables used by the closure.
        dt : float
            Time-step of the forward model :math:`[\text s]`.
        closure_params : CloParT
            Values of the parameters used by the closure (time-independant).
        case_tracable : CaseTracable
            Physical parameters and forcings of the model run.
        
        Returns
        -------
        closure_state : CloStateT
            State of the water column for the variables used by the closure at the next time-step.
        
        Notes
        -----
        :code:`CloStateT` is the type reprensenting the instances of the child classes of
        :class:`~ClosureStateAbstract` and :code:`CloParT` is the same for
        :class:`~ClosureParametersAbstract`.

    """

    name : str
    parameters_class: type[ClosureParametersAbstract]
    state_class: type[ClosureStateAbstract]
    step_fun: Callable[[State, CloStateT, float, CloParT, CaseTracable], CloStateT]

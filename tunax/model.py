"""
Single column forward model core.

This module contains the main class :class:`SingleColumnModel` which is the core of Tunax for
implementing computing the evolution of a water column of the ocean. It also contains functions
that are used in this computation. The model was traduced from Frotran to JAX with the work of
Florian Lemarié and Manolis Perrot [1]_, the translation was done in part using the work of Anthony
Zhou, Linnia Hawkins and Pierre Gentine [2]_. this class and these functions can be obtained by the
prefix :code:`tunax.model.` or directly by :code:`tunax.`.

References
----------
.. [1] M. Perrot and F. Lemarié. Energetically consistent Eddy-Diffusivity Mass-Flux convective
    schemes. Part I: Theory and Models (2024). url :
    `hal.science/hal-04439113 <https://hal.science/hal-04439113>`_.
.. [2] A. Zhou, L. Hawkins and P. Gentine. Proof-of-concept: Using ChatGPT to Translate and
    Modernize an Earth System Model from Fortran to Python/JAX (2024). url :
    `arxiv.org/abs/2405.00018 <https://arxiv.org/abs/2405.00018>`_.

"""

from __future__ import annotations
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, jit, vmap

from tunax.case import Case, CaseTracable
from tunax.space import Grid, State, Trajectory, ArrNz, ArrNzp1, ArrNt, VARIABLE_NAMES
from tunax.functions import FloatJax, tridiag_solve, add_boundaries
from tunax.closure import ClosureParametersAbstract, ClosureStateAbstract, Closure
from tunax.closures_registry import CLOSURES_REGISTRY

type StatesTime = tuple[State, ClosureStateAbstract, float]
"""Type that represent the values that are transformed in an integration step of the model."""


class SingleColumnModel(eqx.Module):
    r"""
    Single column forward model core.

    This forward model of tunax is the combination of 4 things : the physical case
    :attr:`case_tracable`, an initial state of the water column :attr:`init_state`, the time
    information with :attr:`nt`, :attr:`dt` and :attr:`p_out` and the abstraction of the chosen
    closure for eddy-diffusivity :attr:`closure`. Adding a set of parameters for the closure one can
    run the model with the method :meth:`run`. The builder of this class takes the case as
    arguments all the attributes, except for the case : the parameter is an instance of
    :class:`~case.Case` which is transformed in a :class:`~case.CaseTracable` instance for JAX
    purpose and for the closure : the parameter is only the name of the closure.

    Parameters
    ----------
    nt : int
        cf. :attr:`nt`.
    dt : float
        cf. :attr:`dt`.
    p_out : int
        cf. :attr:`p_out`.
    init_state : State
        cf. :attr:`init_state`.
    case : Case
        Physical case and forcings of the experiment.
    closure_name : str
        Name of the chosen closure, must be a key of
        :attr:`~closures_registry.CLOSURES_REGISTRY`, see its documentation
        for the available closures.
    start_time : float, default=0.
        cf. :attr:`start_time`.

    Attributes
    ----------
    nt : int
        Number of integration interations.
    dt : float
        Time-step of integration for every iteration :math:`[\text s]`.
    p_out : int
        Number of of time-steps between every output step.
    init_state : State
        Initial physical state of the water column.
    case_tracable : CaseTracable
        Physical case and forcings of the experiment, made tracable for JAX purposes.
    closure : Closure
        Abstraction representing the chosen closure.
    start_time : float, default=0.
        Value of the starting time in :math:`[\text s]`.
    checkpoint : bool, default=False
        Use the :func:`~jax.checkpoint` on the partial run method. Used for economize the memory
        when computing the gradient, especially on GPUs.

    Note
    ----
    The closure parameters are given only at the end when we compute the model so that the
    calibrations of the parameters are easier.
        
    """

    nt: int = eqx.field(static=True)
    dt: float
    p_out: int = eqx.field(static=True)
    init_state: State
    case_tracable: CaseTracable
    closure: Closure = eqx.field(static=True)
    start_time: float
    checkpoint: bool = eqx.field(static=True)

    def __init__(
            self,
            nt: int,
            dt: float,
            p_out: int,
            init_state: State,
            case: Case,
            closure_name: str,
            start_time: float=0.,
            checkpoint: bool=False
        ) -> None:
        self.nt = nt
        self.dt = dt
        self.p_out = p_out
        self.init_state = init_state
        self.closure = CLOSURES_REGISTRY[closure_name]
        self.start_time = start_time
        self.checkpoint = checkpoint
        # creation of the CaseTracable class
        grid = self.init_state.grid
        case_attributes = {k: v for k, v in vars(case).items() if not k.startswith('__')}
        for tra in ['t', 's', 'b', 'pt']:
            tra_attr = f'{tra}_forcing'
            tra_type_attr = f'{tra}_forcing_type'
            forcing = getattr(case, tra_attr)
            if forcing is not None:
                if isinstance(forcing, tuple):
                    case_attributes[tra_type_attr] = 'borders'
                    case_attributes[tra_attr] = forcing
                elif callable(forcing) and len(inspect.signature(forcing).parameters) == 1:
                    case_attributes[tra_type_attr] = 'constant'
                    vec_fun = vmap(forcing)
                    case_attributes[tra_attr] = grid.hz*vec_fun(grid.zr)
                elif callable(forcing) and len(inspect.signature(forcing).parameters) == 2:
                    case_attributes[tra_type_attr] = 'variable'
                    time = jnp.linspace(0, (self.nt-1)*self.dt, self.nt)
                    zr_grid, time_grid = jnp.meshgrid(grid.zr, time)
                    case_attributes[tra_attr] = grid.hz*forcing(zr_grid, time_grid)
            else:
                case_attributes[tra_type_attr] = None
        self.case_tracable = CaseTracable(**case_attributes)

    def tra_promote(self, promotions: dict[str, str]) -> SingleColumnModel:
        """
        Increase the dimension type of the tracers.

        It is usefull to use :func:`~jax.vmap` on several instances of :class:`SingleColumnModel`
        to do batch computing (run the model in parallel).

        Parameters
        ----------
        promotions : dict[str, str]
            The keys of the dictionnary are the name of the tracer variables to modify the
            dimensions of the forcing, one of {:code:`'t'`, :code:`'s'`, :code:`'b'`, :code:`'pt`'},
            the values are the new dimensions of the forcings, possible values
            {:code:`'constant'`, :code:`'variable'`}.
        
        Returns
        -------
        model : SingleColumnModel
            The :code:`self` object with the several promoted forcings.
        """
        case_tracable = self.case_tracable
        for tra, prom in promotions.items():
            grid = self.init_state.grid
            if prom == 'constant':
                case_tracable = case_tracable.tra_promote_borders_constant(tra, grid)
            elif prom == 'variable':
                initial_type = getattr(case_tracable, f'{tra}_forcing_type') == 'borders'
                if initial_type == 'borders':
                    case_tracable = case_tracable.tra_promote_borders_variable(tra, grid, self.nt)
                elif initial_type == 'constant':
                    case_tracable = case_tracable.tra_promote_constant_variable(tra, self.nt)
        return eqx.tree_at(lambda t: t.case_tracable, self, case_tracable)

    def step(
        self,
        state: State,
        closure_state: ClosureStateAbstract,
        time: float,
        closure_parameters: ClosureParametersAbstract
    ) -> StatesTime:
        r"""
        Runs one time-step of the model.

        This functions first call the closure to compute the eddy-diffusivity and viscosity, and
        then integrate the equations of tracers and momentum. It modifies the :code:`state` with
        these new values and then returns the new :code:`state` and :code:`closure_state`.
        
        Parameters
        ----------
        state : State
            State of the water column at the current :code:`time`.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables at the current :code:`time`.
        time : float
            Time of the current iteration (the mehtod integrates from this time to the next one).
        closure_parameters : ClosureParametersAbstract
            A set of parameters of the used closure.
        
        Returns
        -------
        state : State
            State of the water column after the integration.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables after the integration.
        time : float
            Value of the time after the integration.
        """
        dt = self.dt
        case_tracable = self.case_tracable
        # advance closure state (compute eddy-diffusivity and viscosity)
        closure_state = self.closure.step_fun(
            state, closure_state, dt, closure_parameters, case_tracable
        )
        # advance tracers
        i_time = int(time/self.dt)
        state = advance_tra_ed(state, closure_state.akt, dt, case_tracable, i_time)
        # advance velocities
        state = advance_dyn_cor_ed(state, closure_state.akv, dt, case_tracable)
        time += self.dt
        return state, closure_state, time

    @jax.checkpoint  # pyright: ignore[reportPrivateImportUsage]
    def step_check(
        self,
        state: State,
        closure_state: ClosureStateAbstract,
        time: float,
        closure_parameters: ClosureParametersAbstract
    ) -> StatesTime:
        r"""
        Checkpointed version of :meth:`run` to save memory during the gradient computation.
        
        Parameters
        ----------
        state : State
            State of the water column at the current :code:`time`.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables at the current :code:`time`.
        time : float
            Time of the current iteration (the mehtod integrates from this time to the next one).
        closure_parameters : ClosureParametersAbstract
            A set of parameters of the used closure.
        
        Returns
        -------
        state : State
            State of the water column after the integration.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables after the integration.
        time : float
            Value of the time after the integration.
        """
        return self.step(state, closure_state, time, closure_parameters)

    def run_partial(
            self,
            state0: State,
            closure_state0: ClosureStateAbstract,
            time0: float,
            n_steps: int,
            closure_parameters: ClosureParametersAbstract
        ) ->  StatesTime:
        r"""
        Runs a certain number of time steps.

        Computes a loop of integration for a number of time steps of :code:`n_steps`, and return
        the last states of the loop.
        
        Parameters
        ----------
        state0 : State
            State of the water column at the beginning of the integration loop.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables at the beginning of the integration
            loop.
        time0 : float
            Begining time of the integration loop.
        n_steps : int
            Number of integration steps.
        closure_parameters : ClosureParametersAbstract
            A set of parameters of the used closure.    
        
        Returns
        -------
        state : State
            State of the water column after a number of :code:`n_steps` integration steps.
        closure_state : ClosureStateAbstract
            State of the water column for the closure variables after a number of :code:`n_steps`
            integration steps.
        time : float
            Value of the time after a number of :code:`n_steps` integration steps.
        """
        if self.checkpoint:
            step_fun = self.step_check
        else:
            step_fun = self.step
        def scan_fn(carry: StatesTime, _: FloatJax) -> tuple[StatesTime, None]:
            state, closure_state, time = carry
            state, closure_state, time = step_fun(state, closure_state, time, closure_parameters)
            return (state, closure_state, time), None
        carry, _ = lax.scan(scan_fn, (state0, closure_state0, time0), jnp.arange(n_steps))
        (state, closure_state, time) = carry
        return state, closure_state, time

    def _state_concat_to_traj(self, states: State, times: ArrNt) -> Trajectory:
        """
        Convert the concatenations of :class:`~space.State` in a :class:`~space.Trajectory`.
        
        Use to get a trajectory from the output of the :func:`~jax.lax.scan` function which
        computes the concatenation of several :class:`~space.State` instances.

        Parameters
        ----------
        states : State
            A concatenation of :class:`~space.State` instances. More specifically, every leafs of
            the pytree is an array on the first axis of all the values of the the
            :class:`~space.State` instances.
        times : float :class:`~jax.Array` of shape (nt)
            Values of the different times corresponding at each :class:`~space.State` instances.
        
        Returns
        -------
        trajectory : Trajectory
            Trajectory corresponding to the concatenation of the states.
        """
        var_dict = {}
        for var in VARIABLE_NAMES:
            if getattr(states, var) is not None:
                var0 = jnp.expand_dims(getattr(self.init_state, var), 0)
                var_computed = getattr(states, var)
                var_dict[var] = jnp.concat([var0, var_computed])
        times = jnp.concat([jnp.array([self.start_time]), times])
        return Trajectory(self.init_state.grid, times, **var_dict)


    def run(self, closure_parameters: ClosureParametersAbstract) -> Trajectory:
        r"""
        Main run the model.

        Computes the :attr:`nt` integration steps of lenght :attr:`dt` from the initial state
        :attr:`init_state` with and doing an output of the state every :attr:`p_out` steps with the
        physical case and forcings corresponding to :attr:`case_tracable`. The closure for eddy-
        diffusivity used is :attr:`closure` with the values of the parameters described by the
        parameter :code:`closure_parameters`.
        
        Parameters
        ----------
        closure_parameters : ClosureParametersAbstract
            The set of parameters to use for the computation of the closure of eddy-diffusivity.
        
        Returns
        -------
        trajectory : Trajectory
            The trajectory with all the output steps of the integration loop.
        """
        init_closure_state = self.closure.state_class(self.init_state.grid, closure_parameters)

        def scan_fn(carry: StatesTime, _: FloatJax) -> tuple[StatesTime, StatesTime]:
            state, closure_state, time = carry
            state, closure_state, time = self.run_partial(
                state, closure_state, time, self.p_out, closure_parameters
            )
            return (state, closure_state, time), (state, closure_state, time)

        n_steps_out = self.nt//self.p_out
        (_, _, _), (states, _, times) = lax.scan(
            scan_fn, (self.init_state, init_closure_state, self.start_time), jnp.arange(n_steps_out)
        )
        return self._state_concat_to_traj(states, jnp.array(times))

    @jit
    def jit_run(self, closure_parameters: ClosureParametersAbstract) -> Trajectory:
        r"""
        Jitted version of :meth:`run`.

        This method does exacly like :meth:`run` but :func:`~jax.jit` is applied on it, which
        means that the first call of this method will be the compilation of the function, and the
        next ones will be the compiled execution of the function which are supposed to be faster.
        There will be a compilation each time that this method will be call for a new "shape" of the
        :class:`SingleColumnModel` instance (which means that all the leafs of the pytree have the
        same shape as :class:`~jax.Array`), but even if this method is call on different instances,
        if they have the same shape, the compilation will be done only one time. Moreover, this
        method should use only for direct methods, if one wants to apply a :func:`~jax.grad` over
        it's better to put the :func:`~jax.jit` outside.
        
        Parameters
        ----------
        closure_parameters : ClosureParametersAbstract
            cf. :meth:`run`.
        
        Returns
        -------
        trajectory : Trajectory
            cf. :meth:`run`.
        """
        return self.run(closure_parameters)


def lmd_swfrac(hz: ArrNz) -> ArrNzp1:
    r"""
    Compute solar forcing.

    Compute fraction of solar shortwave flux penetrating to specified depth due to exponential decay
    in Jerlov water type. This function is called once before running the model.

    Parameters
    ----------
    hz : float :class:`~jax.Array` of shape (nz)
        Thickness of cells from deepest to shallowest :math:`[\text m]`.

    Returns
    -------
    swr_frac : float :class:`~jax.Array` of shape (nz+1)
        Fraction of solar penetration throught the water column :math:`[\text{dimensionless}]`.
    """
    nz, = hz.shape
    mu1 = 0.35
    mu2 = 23.0
    r1 = 0.58
    attn1 = -1.0 / mu1
    attn2 = -1.0 / mu2

    xi1 = attn1 * hz
    xi2 = attn2 * hz

    def lax_step(sdwk, k):
        sdwk1, sdwk2 = sdwk
        sdwk1 = lax.cond(xi1[nz-k] > -20, lambda x: x*jnp.exp(xi1[nz-k]), lambda x: 0.*x, sdwk1)
        sdwk2 = lax.cond(xi2[nz-k] > -20, lambda x: x*jnp.exp(xi2[nz-k]), lambda x: 0.*x, sdwk2)
        return (sdwk1, sdwk2), sdwk1+sdwk2

    _, swr_frac = lax.scan(lax_step, (r1, 1.0 - r1), jnp.arange(1, nz+1))
    return jnp.concat((swr_frac[::-1], jnp.array([1.])))


def tracer_flux(
        tracer: str,
        case_tracable: CaseTracable,
        grid: Grid,
        i_time: int
    ) -> ArrNz:
    r"""
    Computes flux of the tracer forcing.
    
    This function get the flux of the forcing at a certain time depending on the type of the
    forcing, the flux being the derivative of the forcing along the depth.
    
    Parameters
    ----------
    tracer : str
        Name of the tracer variable of the concerned forcing. One of {:code:`'t'`, :code:`'s'`,
        :code:`'b'`, :code:`'pt`'}.
    case_tracable : CaseTracable
        Physical case which contains the forcings type and values.
    grid : Grid
        Vertical grid of the water column.
    i_time : int
        Index of the time iteration corresponding to the index in the forcing.
    
    Returns
    -------
    df : float :class:`~jax.Array` of shape (nz)
        Flux of the forcing of the tracer. At each cell it represents the difference between the
        input and the ouput flux.
    
    Raises
    ------
    ValueError
        If the forcing type is not one of {'borders', 'constant', 'variable'}.
    """
    forcing = getattr(case_tracable, f'{tracer}_forcing')
    forcing_type = getattr(case_tracable, f'{tracer}_forcing_type')
    match forcing_type:
        case 'borders':
            df = add_boundaries(-forcing[0], jnp.zeros(grid.nz-2), forcing[1])
        case 'constant':
            df = forcing
        case 'variable':
            df = forcing[i_time, :]
        case _:
            mess = f'Forcing type of variable {tracer} should be one of' + \
                "{'borders', 'constant', 'variable'}."
            raise ValueError(mess)
    return df


def advance_tra_ed(
        state: State,
        akt: ArrNzp1,
        dt: float,
        case_tracable: CaseTracable,
        i_time: int,
    ) -> State:
    r"""
    Integrate vertical diffusion term for tracers.

    First the flux divergences are computed taking in account the forcings. Then the diffusion
    equation of the tracers system is solved, and the tracers at next time-step are returned. The
    solved equation is for :math:`C` a tracer :

    :math:`\partial _z ( K_m \partial _z C) + \partial _t C + F = 0`

    where :math:`F` is the representation of the forcings. This equation is solved for every tracer
    indicated in :attr:`~case.Case.eos_tracers` and the passive tracer if :attr:`~case.Case.do_pt`
    is set.
    
    Parameters
    ----------
    state : State
        State of the water column at the current iteration.
    akt : float :class:`~jax.Array` of shape (nz)+1
        Eddy-diffusivity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    dt : float
        Time-step of the integration step.
    case_tracable : CaseTracable
        Physical case which contains the forcings type and values.
    i_time : int
        Index of the time iteration corresponding to the index in the forcing.
    
    Returns
    -------
    state : State
        The state of the water column with the values of the tracers after the integration and the
        diffusion by the eddy-diffusivity.
    """
    hz = state.grid.hz
    tracers = [tra for tra in case_tracable.eos_tracers]
    if case_tracable.do_pt:
        tracers.append('pt')
    def get_pytree_fun(tracer: str):
        return lambda t: getattr(t, tracer)
    for tracer in tracers:
        tra = getattr(state, tracer)
        df = tracer_flux(tracer, case_tracable, state.grid, i_time)
        df = hz*tra + dt*df
        tra = diffusion_solver(akt, hz, df, dt)
        state = eqx.tree_at(get_pytree_fun(tracer), state, tra)
    return state


def advance_dyn_cor_ed(
        state: State,
        akv: ArrNzp1,
        dt: float,
        case_tracable: CaseTracable
    ) -> State:
    r"""
    Integrate vertical diffusion and Coriolis terms for momentum.

    First the Coriolis term is computed, then the momentum forcings are applied and finally, the
    diffusion equation is solved. The momentum at next time-step is returned. The equation which is
    solved is :

    :math:`\partial_z (K_v \partial_z U) + F_{\text{cor}}(U) + F = 0`

    where :math:`F_{\text{cor}}` represent the Coriolis effect, and :math:`F` represent the effect
    of the forcings on the momentum.
    
    Parameters
    ----------
    state : State
        State of the water column at the current iteration.
    akv : float :class:`~jax.Array` of shape (nz+1)
        Eddy-viscosity on the interfaces of the cells
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    dt : float
        Time-step of the integration step.
    case_tracable : CaseTracable
        Physical case which contains the forcings type and values.
    
    Returns
    -------
    state : State
        The state of the water column with the values of the momentum after the integration and the
        diffusion by the eddy-viscosity.
    """
    gamma_cor = 0.55
    fcor = case_tracable.fcor
    u = state.u
    v = state.v
    hz = state.grid.hz

    # 1 - Compute Coriolis term
    cff = (dt * fcor) ** 2
    cff1 = 1 / (1 + gamma_cor * gamma_cor * cff)
    fu = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*u + dt*fcor*v)
    fv = cff1 * hz * ((1-gamma_cor*(1-gamma_cor)*cff)*v - dt*fcor*u)

    # 2 - Apply surface and bottom forcing
    fu = fu.at[-1].add(dt * case_tracable.ustr_sfc)
    fv = fv.at[-1].add(dt * case_tracable.vstr_sfc)
    fu = fu.at[0].add(-dt * case_tracable.ustr_btm)
    fv = fv.at[0].add(-dt * case_tracable.vstr_btm)

    # 3 - Implicit integration for vertical viscosity
    u = diffusion_solver(akv, hz, fu, dt)
    v = diffusion_solver(akv, hz, fv, dt)

    # 4 - Update the state
    state = eqx.tree_at(lambda t: t.u, state, u)
    state = eqx.tree_at(lambda t: t.v, state, v)

    return state


def diffusion_solver(
        ak: ArrNzp1,
        hz: ArrNz,
        f: ArrNz,
        dt: float
    ) -> ArrNz:
    r"""
    Solve a diffusion problem with finite volumes.

    The diffusion problems can be written

    :math:`\partial _z (K \partial _z X) + \dfrac f {\Delta t \Delta x} = 0`

    where we are searching for :math:`X` and where :math:`f` represents the temporal derivative and
    forcings. This function transforms this problem in a tridiagonal system and then solve it.

    Parameters
    ----------
    ak : float :class:`~jax.Array` of shape (nz+1)
        Diffusion at the cell interfaces :math:`K` in
        :math:`\left[\text m ^2 \cdot \text s ^{-1}\right]`.
    hz : float :class:`~jax.Array` of shape (nz)
        Thickness of cells from deepest to shallowest :math:`\left[\text m\right]`.
    f : float :class:`~jax.Array` of shape (nz)
        Right-hand flux of the equation :math:`f` in :math:`[[X] \cdot \text m ]`.
    dt : float
        Time-step of discretisation :math:`[\text s]`.
    
    Returns
    -------
    x : float :class:`~jax.Array` of shape (nz)
        Solution of the diffusion problem :math:`X` in :math:`\left[[X]\right]`.
    """
    # fill the coefficients for the tridiagonal matrix
    a_in = -2.0 * dt * ak[1:-2] / (hz[:-2] + hz[1:-1])
    c_in = -2.0 * dt * ak[2:-1] / (hz[2:] + hz[1:-1])
    b_in = hz[1:-1] - a_in - c_in

    # bottom boundary condition
    c_btm = -2.0 * dt * ak[1] / (hz[1] + hz[0])
    b_btm = hz[0] - c_btm

    # surface boundary condition
    a_sfc = -2.0 * dt * ak[-2] / (hz[-2] + hz[-1])
    b_sfc = hz[-1] - a_sfc

    # concatenations
    a = add_boundaries(jnp.array([0.]), a_in, a_sfc)
    b = add_boundaries(b_btm, b_in, b_sfc)
    c = add_boundaries(c_btm, c_in, jnp.array([0.]))

    x = tridiag_solve(a, b, c, f)

    return x

"""
Abstraction for calibration databases.


This module the objects that are used in Tunax to describe a :class:`Database` of observations used
for a calibration. By *datas* (:class:`Data`), we refer ton the union of a trajectory and a physical
case which represent a measurment or a reference as a Large Eddy Simulation (LES) for example. And
by *observations* (:class:`Obs`), we refer to the union of a trajectory and the Tunax model which
corresponds to it. These classes can be obtained by the prefix :code:`tunax.database.` or directly
by :code:`tunax.`.

"""

from __future__ import annotations
import warnings
from typing import Union, Optional, Tuple, List, Dict, TypeAlias, cast

import yaml
import xarray as xr
import equinox as eqx
import numpy as np
import jax.numpy as jnp
from h5py import File as H5pyFile
from h5py import Dataset as H5pyDataset
from h5py import Group as H5pyGroup
from jaxtyping import Array, Float

from tunax.space import Grid, Trajectory, TRACERS_NAMES, VARIABLE_NAMES
from tunax.case import Case
from tunax.functions import _format_to_single_line
from tunax.model import SingleColumnModel

DimsType: TypeAlias = Tuple[Optional[int]]
"""Type that represent the dimensions on which load the datas from a file."""


def _get_var_jl(
        jl_file: H5pyFile,
        var_names: Dict[str, str],
        var: str,
        n: int,
        dims: Union[DimsType, Dict[str, DimsType]] = (None,),
        suffix: str = ''
    ) -> Float[Array, 'n']:
    """
    This function retrieves the right value of a variable in a .jld2 file.

    This function retrives first the raw data that corresponding to the variable var (name from
    Tunax) in accord to the names registry var_names, with an eventual suffix at the end. In a
    second time it will project in the good dimensions the raw data with the indications in dims to
    get a one-dimension array. And finally it will remove the eventual borders by taking the middle
    part of the array of lenght n.
    
    Parameters
    ----------
    jl_file : H5pyFile
        A .jld2 file loaded with the package h5py.
    var_names : Dict[str, str]
        The reference on which search the variable on the file, the keys are the Tunax names of
        variables and the values are the names in the file (which are actually paths).
    var : str
        The name of the variable to search in terms of Tunax.
    n : int
        The excpected lenght of the variable. The function will keep the middle part of lenght n
        on the array that it will directly extract from the file.
    dims : DimsType or Dict[str, DimsType], default=(None,)
        It contains the dimensions on which search the right array. If it's a dictionnary, it's like
        var_names, the keys are the names of the variables in terms of Tunax and the values are the
        dimensions for each variable. Then we have a Tuple of int or Nones which corresponds at
        every axis of the raw data from the file. If an axis is indexed with None, it means that we
        keep this dimension, if an axis is indexed with an integer, it means that we reduce this
        axis to the value of the raw data on this index.
    suffix : str, default=''
        A string suffix to add at the end of the path that is in var_names.

    Returns
    -------
    arr : float :class:`~jax.Array` of shape (n)
        Value of the right array in the :code:`.jld2` file.
    
    Raises
    ------
    ValueError
        If the lenght of the tuple dims doesn't have the same number of elements than the number of
        axis in the raw data from the file.

    Warns
    -----
    Odd shift
        If the number n hasn't the same parity as the lenght of the raw data array. Then the borders
        are not symetric.
    """
    jl_var = cast(H5pyDataset, jl_file[f'{var_names[var]}{suffix}'])
    if isinstance(dims, dict):
        dims = dims[var]
    if len(jl_var.shape) != len(dims):
        raise ValueError(_format_to_single_line(f"""
            The tuple parameter `dims` of value {dims} must have the length of the number of
            dimension of the variable {var} array in the `jl_file`.
        """))
    dims_slice = tuple(slice(None) if x is None else x for x in dims)
    jl_var_1d = jl_var[dims_slice]
    double_shift = jl_var_1d.shape[0] - n
    shift = double_shift//2
    if double_shift%2 == 1:
        warnings.warn(_format_to_single_line(f"""
            The length array from the `jl_file` of the variable {var} minus `n` is an odd number :
            the removed boundaries are taken 1 point thinner on the bottom side than on the surface
            side.
        """))
        return jl_var_1d[shift:-shift-1]
    return jl_var_1d[shift:-shift]


class Data(eqx.Module):
    """
    Abstraction to represent an element of the database from the point of view of Tunax.

    This abstraction is the link between the time-series of :class:`Trajectory` and a physical
    situation described by :class:`Case`. It can eventually contains metadatas. Typically this class
    appears when one want to import the different element of a database of observations or
    simulations. The constructor takes all the attributes as parameters.
    
    Attributes
    ----------
    trajectory : Trajectory
        The time-series of the variables that represent this data.
    case : Case
        The physical case that represent this data.
    metadatas : Dict[str, float], default={}
        Some metadatas that we want to use later. It can be some values of the :attr:`case` that
        we want to set by hand later.

    Raises
    ------
    ValueError
        If the :attr:`~space.Trajectory.time` of :attr:`trajectory` is not build with constant
        time-steps.

    """

    trajectory: Trajectory
    case: Case
    metadatas: Dict[str, float] = eqx.field(static=True)

    def __init__(
            self,
            trajectory: Trajectory,
            case: Case,
            metadatas: Optional[Dict[str, float]]=None
        ) -> None:
        time = trajectory.time
        steps = time[1:] - time[:-1]
        if not jnp.all(steps == steps[0]):
            raise ValueError('Tunax only handle constant output time-steps')
        self.trajectory = trajectory
        self.case = case
        if metadatas is None:
            metadatas = {}
        self.metadatas = metadatas

    @classmethod
    def from_nc_yaml(
            cls,
            nc_path: str,
            yaml_path: str,
            var_names: Dict[str, str],
            eos_tracers: str = 't',
            do_pt: bool = False
        ) -> Data:
        """
        Create a :class:`Data` instance from a *netcdf* and a *yaml* files.

        This class method build a trajectory from the :code:`.nc` file :code:`nc_path`, it build the
        physical parameters from the configuration file :code:`yaml_path`. :code:`var_names` is used
        to do the link between Tunax name convention and the one from the used database.

        Parameters
        ----------
        nc_path : str
            Path of the *netcdf* file that contains the time-series of the observation trajectory.
            The file should contains at least the three dimensions :attr:`~space.Grid.zr`,
            :attr:`~space.Grid.zw` and :attr:`~space.Trajectory.time`. The time-series can be
            created with default values if they are not present in the file (only for
            :attr:`space.Trajectory.u` and :attr:`space.Trajectory.v`). Otherwise, they must have
            the good dimensions described in :class:`~space.Trajectory`.
        yaml_path : str
            Path of the *yaml* file that contains the parameters and forcing that describe the
            observation. The parameters should be float numbers and directly accessible from the
            root of the file with a key. Only the parameters that are described in
            :class:`~case.Case` will be takend in account.
        var_names : Dict[str, str]
            Link between the convention names in Tunax and the ones in the database. The keys are
            the Tunax names and the values are the names in the database. It works for variables of
            the :class:`~space.Trajectory` and fornthe parameters of :class:`~case.Case`. It must at
            least contains entries for :attr:`~space.Grid.zr` :attr:`~space.Grid.zw` and
            :attr:`~space.Trajectory.time`
        eos_tracers : str, default='t'
            Tracers used for the equation of state, cf. :attr:`~case.Case.eos_tracers`.
        do_pt : bool, default=False
            Compute or not a passive tracer, cf. :attr:`~case.Case.do_pt`.
        
        Returns
        -------
        data : Data
            An object that represent these files.
        """
        ds = xr.load_dataset(nc_path)
        # dimensions
        zr = jnp.array(ds[var_names['zr']].values)
        zw = jnp.array(ds[var_names['zw']].values)
        grid = Grid(zr, zw)
        time = jnp.array(ds[var_names['time']].values)
        nt, = time.shape
        nz = grid.nz
        # variables
        u_name = var_names['u']
        if u_name == '':
            u = jnp.full((nt, nz), 0.)
        else:
            u = jnp.array(ds[u_name].values)
        v_name = var_names['v']
        if v_name == '':
            v = jnp.full((nt, nz), 0.)
        else:
            v = jnp.array(ds[v_name].values)
        tracers_dict = {}
        for tracer in TRACERS_NAMES:
            if tracer in var_names.keys():
                tracers_dict[tracer] = jnp.array(ds[var_names[tracer]].values)
        # writing trajectory
        trajectory = Trajectory(grid, time, u, v, **tracers_dict)
        # writing the case
        with open(yaml_path, 'r', encoding='utf-8') as f:
            metadatas = yaml.safe_load(f)
        case = Case()
        case_attributes = list(vars(case).keys())
        def get_pytree_fun(att: str):
            return lambda t: getattr(t, att)
        for att in case_attributes:
            if att in var_names.keys():
                case = eqx.tree_at(get_pytree_fun(att), case, metadatas[var_names[att]])
        case = eqx.tree_at(lambda t: getattr(t, 'eos_tracers'), case, eos_tracers)
        case = eqx.tree_at(lambda t: getattr(t, 'do_pt'), case, do_pt)

        return cls(trajectory, case, metadatas)

    @classmethod
    def from_jld2(
            cls,
            jld2_path: str,
            names_mapping: Dict[str, Dict[str, str]],
            nz: Optional[int] = None,
            dims: Union[DimsType, Dict[str, DimsType]] = (None,),
            eos_tracers: str = 't',
            do_pt: bool = False
        ) -> Data:
        """
        Creates a :class:`Data` instance from a :code:`.jld2` file.

        For the scalar parameters, the values must be registered in the file simply with their path
        in the file, separated with :code:`/` in the same string. For the timeseries and the time,
        the arrays of each time step must be register with a path that ends with the reference of
        the time. For the other variables, just register the normal path. The time is appriximated
        to the order of the second.

        Parameters
        ----------
        jld2_path : str
            Path of the *netcdf* file that contains the time-series of the observation trajectory
            and the physical parameters and forcings.
        names_mapping: Dict[str, Dict[str, str]]
            Contains the link between the Tunax names of variables and the path of the variables in
            the file. There are 3 first entries :

            - :code:`variables` :for all the variables corresponding to the :class:`space.Grid` and
              the :class:`space.Trajectory`. For the grid attributes (:attr:`space.Grid.zr` and
              :attr:`space.Grid.zw`) the path should correspond directly to the array in the file.
              For the time and the time-series, the given path corresponds to a path with all the
              reference (with a number in string) of the time, and then in these path with the time
              we have the array of the variable (or the float corresponding to the value of the
              time) at the time with this reference. Then the 2D arrays are rebuild by
              concatenation. The references of the time are get with the path of the time data.

            - :code:`parameters` : for all the scalar entries corresponding directly to the
              parameters of :class:`case.Case`

            - :code:`metadatas` : for the scalar entries that we want to keep in the
              :attr:`metadatas` for later.
              
        nz : int, optionnal, default=None
            Expected number of steps of the grid of the water column. The method will remove the
            borders of the raw data from the file to keep only the middle part of this lenght. If
            nothing is entered for this parameter, all the raw data are kept.
        dims : DimsType or Dict[str, DimsType], default=(None,)
            It contains the dimensions on which search the right arrays. If it's a dictionnary,
            it's like :code:`var_names`, the keys are the names of the variables in terms of Tunax
            and the values are the dimensions for each variable. Then we have a Tuple of int or
            Nones which corresponds at every axis of the raw data from the file. If an axis is
            indexed with None, it means that we keep this dimension, if an axis is indexed with an
            integer, it means that we reduce this axis to the value of the raw data on this index.
        eos_tracers : str, default='t'
            Tracers used for the equation of state, cf. :attr:`~case.Case.eos_tracers`.
        do_pt : bool, default=False
            Compute or not a passive tracer, cf. :attr:`~case.Case.do_pt`.
        
        Returns
        -------
        data : Data
            An object that represent these file.
        """
        var_map = names_mapping['variables']
        par_map = names_mapping['parameters']
        metadata_map = names_mapping['metadatas']
        jl = H5pyFile(jld2_path, 'r')
        # récupération de la bonne valeur de nz
        if nz is None:
            ds = cast(H5pyDataset, jl[par_map['nz']])
            nz = int(ds[()])
        # variables grid et time
        zr = jnp.array(_get_var_jl(jl, var_map, 'zr', nz, dims))
        zw = jnp.array(_get_var_jl(jl, var_map, 'zw', nz+1, dims))
        time_group = var_map['time']
        gr = cast(H5pyGroup, (jl[time_group]))
        time_str_list = list(gr.keys())
        time_str_list = [int(i) for i in time_str_list]
        time_str_list.sort()
        time_str_list = [str(i) for i in time_str_list]
        time_float_list = []
        for time_str in time_str_list:
            ds = cast(H5pyDataset, jl[f'{time_group}/{time_str}'])
            time_val = float(int(ds[()]))
            time_float_list.append(float(time_val))
        time = jnp.array(time_float_list)
        # variables
        variables_dict = {}
        for var_name in VARIABLE_NAMES:
            if var_name not in var_map:
                continue
            var_list = []
            for time_str in time_str_list:
                var_time = _get_var_jl(jl, var_map, var_name, nz, dims, f'/{time_str}')
                var_list.append(var_time)
            variables_dict[var_name] = jnp.vstack(var_list)
        # trajectory
        trajectory = Trajectory(Grid(zr, zw), time, **variables_dict)

        # parameters
        params = {}
        case_params_list = [nom for nom in vars(Case).keys()]
        for par_name, jl_name in par_map.items():
            if par_name in case_params_list:
                ds = cast(H5pyDataset, jl[jl_name])
                params[par_name] = float(ds[()])
        case = Case(eos_tracers=eos_tracers, do_pt=do_pt, **params)

        # metadatas
        metadatas = {}
        for metadata_name, jl_name in metadata_map.items():
            ds = cast(H5pyDataset, jl[jl_name])
            jl_val = ds[()]
            if isinstance(jl_val, np.floating) or isinstance(jl_val, np.integer):
                metadatas[metadata_name] = float(jl_val)

        # return time
        return cls(trajectory, case, metadatas)

    def cut(self, out_nt_cut: int) -> List[Data]:
        """
        Cuts the :attr:`Trajectory` in sub-trajectories, cf. :meth:`space.Trajectory.cut`.

        Parameters
        ----------
        out_nt_cut : int
            Number of output steps of the sub-trajectories.
        
        Returns
        -------
        traj_list : List[Data]
            List of :class:`Data` instances with the sub-trajectories in the chronological order.
        """
        traj_list = self.trajectory.cut(out_nt_cut)
        return [Data(traj, self.case, self.metadatas) for traj in traj_list]


class Weights(eqx.Module):
    """
    Representation of the weights to put on every variable for the computing of the lost function.
    The constructor takes all the attributes as parameters.

    Attributes
    ----------
    weight_u : float, default=0.
        Weight on zonal velocity.
    weight_v : float, default=0.
        Weight on meridionnal velocity.
    weight_t : float, default=0.
        Weight on temperature.
    weight_s : float, default=0.
        Weight on salinity.
    weight_b : float, default=0.
        Weight on buoyancy.
    weight_pt : float, default=0.
        Weight on passive tracer.
    """
    weight_u: float = 0.
    weight_v: float = 0.
    weight_t: float = 0.
    weight_s: float = 0.
    weight_b: float = 0.
    weight_pt: float = 0.


class Obs(eqx.Module):
    """
    This class represents and element of the database from the point of view of the loss function.
     
    This class prepares everything to make the loss function able to compute the loss for this
    element of the database. Indeed this class makes the link between the :class:`Trajectory`
    corresponding to this element, a model (with a grid a time parameters) corresponding to this
    trajectory, and the weights that we want to put on each variable. The constructor takes all the
    attributes as parameters.

    Attributes
    ----------
    trajectory: Trajectory
        The time-series of the variables that represent this observation.
    model: SingleColumnModel
        A model built on this trajectory and on a physical case with the time and geometrical
        parameters.
    weights: Weights
        The weights to give to the loss function.
    """
    trajectory: Trajectory
    model: SingleColumnModel
    weights: Weights

    @classmethod
    def from_data(cls, data: Data, dt: float, weights: Weights, checkpoint: bool=False):
        """
        Create a Obs instance from a Data one adding :class:`Weights` and a :code:`dt`.

        This function builds the other time parameters of the model from the trajectory.

        Parameters
        ----------
        data : Data
            A data containing the trajectory and the physical case that we want to apply on our
            model.
        dt : float
            The integration time-step that we want for our model.
        weights : Weights
            The weights to give to the loss function.
        checkpoint : bool, default=False
            Use the :func:`~jax.checkpoint` on the partial run method. Used for economize the memory
            when computing the gradient, especially on GPUs.
        """
        time = data.trajectory.time
        nt = int((time[-1]-time[0])/dt)
        out_dt = float(time[1] - time[0])
        p_out = int(out_dt/dt)
        init_state = data.trajectory.extract_state(0)
        start_time = float(time[0])
        model = SingleColumnModel(
            nt, dt, p_out, init_state, data.case, 'k-epsilon', start_time, checkpoint
        )
        return Obs(data.trajectory, model, weights)

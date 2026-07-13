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
from pathlib import Path
from dataclasses import replace
from typing import Union, Optional, Tuple, List, Dict, TypeAlias, Callable, Any, cast

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


DimsType: TypeAlias = tuple[Union[None, int], ...]
"""TypeAlias : tuple of integers or None, used for of the dimensions in the loaders."""


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
    metadatas : Dict[str, float :class:`~jax.Array`], default={}
        Some metadatas that we want to use later. They can be floats or arrays but always written in
        a JAX array.

    Raises
    ------
    ValueError
        If the :attr:`~space.Trajectory.time` of :attr:`trajectory` is not build with constant
        time-steps.

    """

    trajectory: Trajectory
    case: Case
    metadatas: Dict[str, jnp.ndarray] = eqx.field(static=True)

    @staticmethod
    def _proj_dims(
            arr: Union[H5pyDataset, np.ndarray],
            dims: DimsType,
            n: int
        ) -> Float[Array, 'n']:
        """
        Projection of the loaded data on the right dimension (only one dimension).

        First, the one dimension is selected in `arr`, the one corresponding to the `None` in `dims`
        and projected in the indexes of the other parameters. Then the borders are deleted to make
        keep the middle part of the array of shape `n`. The function also do the conversion in JAX.

        Parameters
        ----------
        arr : h5py Dataset or numpy array
            Array on which do the projection.
        dims : typle of None and integers
            Must contains exactly one None on the dimension on which select the array. The other
            integers are the indexes of selection of the data.
        n : int
            Size of the output array
        
        Returns
        -------
        projected_arr : float :class:`~jax.Array` of shape (n)
            Array arr projected on the good dimension and cuted on the edges to get the right size.
        """
        if len(arr.shape) != len(dims):
            raise ValueError(_format_to_single_line(f"""
                The tuple parameter `dims` of value must have the length of the array `arr` readed
                from the file.
            """))
        if dims.count(None) != 1:
            raise ValueError(_format_to_single_line(f"""
                The tuple parameter `dims` must contains exactly one None.
            """))
        dims_slice = tuple(slice(None) if x is None else x for x in dims)
        arr_1d = arr[dims_slice]
        double_shift = arr_1d.shape[0] - n
        shift = double_shift//2
        if double_shift%2 == 1:
            warnings.warn(_format_to_single_line(f"""
                The size of the loaded array minus n is an odd number : the removed boundaries are
                taken 1 point thinner on the bottom side than on the surface side.
            """))
        return jnp.array(arr_1d[shift:shift+n])


    @staticmethod
    def _get_var_jl(
            jl_file: H5pyFile,
            var_names: Dict[str, str],
            var: str,
            n: int,
            dims_mapping: Union[DimsType, Dict[str, DimsType]] = (None,),
            suffix: str = ''
        ) -> Float[Array, 'n']:
        """
        This function retrieves the right value of a variable in a .jld2 file.

        This function retrives first the raw data that corresponding to the variable var (name from
        Tunax) in accord to the names registry var_names, with an eventual suffix at the end. In a
        second time it will project in the good dimensions the raw data with the indications in dims
        to get a one-dimension array. And finally it will remove the eventual borders by taking the
        middle part of the array of lenght n.
        
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
        dims_mapping : DimsType or Dict[str, DimsType], default=(None,)
            It contains the dimensions on which search the right array. If it's a dictionnary, it's
            like var_names, the keys are the names of the variables in terms of Tunax and the values
            are the dimensions for each variable. Then we have a Tuple of int or Nones which
            corresponds at every axis of the raw data from the file. If an axis is indexed with
            None, it means that we keep this dimension, if an axis is indexed with an integer, it
            means that we reduce this axis to the value of the raw data on this index.
        suffix : str, default=''
            A string suffix to add at the end of the path that is in var_names.

        Returns
        -------
        arr : float :class:`~jax.Array` of shape (n)
            Value of the right array in the :code:`.jld2` file.
        """
        jl_var = cast(H5pyDataset, jl_file[f'{var_names[var]}{suffix}'])
        if isinstance(dims_mapping, dict):
            dims = dims_mapping[var]
        else:
            dims = dims_mapping
        return Data._proj_dims(jl_var, dims, n)


    @classmethod
    def _load_jld2(
            cls,
            jld2_path: str,
            names_mapping: Dict[str, Dict[str, str]],
            nz: Optional[int] = None,
            dims_mapping: Union[DimsType, Dict[str, DimsType]] = (None,)
        ) -> Tuple[Data, Dict[str, jnp.ndarray]]:
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
        names_mapping : Dict[str, Dict[str, str]]
            Contains the link between the Tunax names of variables and the path of the variables in
            the file. There are 4 first entries :
            - :code:`variables` : for all the variables corresponding to the :class:`space.Grid` and
              the :class:`space.Trajectory`. For the grid attributes (:attr:`space.Grid.zr` and
              :attr:`space.Grid.zw`) the path should correspond directly to the array in the file.
              For the time and the time-series, the given path corresponds to a path with all the
              reference (with a number in string) of the time, and then in these path with the time
              we have the array of the variable (or the float corresponding to the value of the
              time) at the time with this reference. Then the 2D arrays are rebuild by
              concatenation. The references of the time are get with the path of the time data.
            - :code:`parameters` : for all the scalar entries corresponding directly to the
              parameters of :class:`case.Case`
            - :code:`metadatas` : for the entries that we want to keep in the :attr:`metadatas` for
            later.
            - :code:`adjust_params` : for the parameters entries that are used by the adjusting
            function later.
        nz : int, optionnal, default=None
            Expected number of steps of the grid of the water column. The method will remove the
            borders of the raw data from the file to keep only the middle part of this lenght. If
            nothing is entered for this parameter, the value is loaded from the variables mapping,
            if nothing is registered, the value will be loaded from the size of the entry
            :code:`'zr'`.
        dims_mapping : DimsType or Dict[str, DimsType], default=(None,)
            It contains the dimensions on which search the right arrays. If it's a dictionnary,
            it's like :code:`var_names`, the keys are the names of the variables in terms of Tunax
            and the values are the dimensions for each variable. Then we have a Tuple of int or
            Nones which corresponds at every axis of the raw data from the file. If an axis is
            indexed with None, it means that we keep this dimension, if an axis is indexed with an
            integer, it means that we reduce this axis to the value of the raw data on this index.
        
        Returns
        -------
        data : Data
            An object containing the trajectory, the physical case and the metadata in this file.
        adjust_parameters_load : Dict[str, jnp.ndarray]
            An dictionnary with the parameters of an eventual adjusting function used in
            :meth:`load`.
        """
        var_map = names_mapping['variables']
        par_map = names_mapping['parameters']
        # read .jl file
        jl = H5pyFile(jld2_path, 'r')
        # load nz
        if nz is None:
            if 'nz' in var_map.keys():
                ds = cast(H5pyDataset, jl[var_map['nz']])
                nz = int(ds[()])
            else:
                if isinstance(dims_mapping, dict) and 'zr' in dims_mapping.keys():
                    i_zr = dims_mapping['zr'].index(None)
                else:
                    i_zr = 0
                ds = cast(H5pyDataset, jl[var_map['zr']])
                nz = int(ds.shape[i_zr])
        # load grid et time
        zr = jnp.array(Data._get_var_jl(jl, var_map, 'zr', nz, dims_mapping))
        zw = jnp.array(Data._get_var_jl(jl, var_map, 'zw', nz+1, dims_mapping))
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
        # load variables
        variables_dict = {}
        for var_name in VARIABLE_NAMES:
            if var_name not in var_map:
                continue
            var_list = []
            for time_str in time_str_list:
                var_time = Data._get_var_jl(jl, var_map, var_name, nz, dims_mapping, f'/{time_str}')
                var_list.append(var_time)
            variables_dict[var_name] = jnp.vstack(var_list)
        # generate trajectory
        trajectory = Trajectory(Grid(zr, zw), time, **variables_dict)
        # generate parameters
        params = {}
        for par_name, jl_name in par_map.items():
            if jl_name in jl.keys():
                ds = cast(H5pyDataset, jl[jl_name])
                params[par_name] = float(ds[()])
        case = Case(**params)
        # generate metadatas
        metadatas = {}
        for metadata_name, jl_name in names_mapping['metadatas'].items():
            if jl_name in jl.keys():
                ds = cast(H5pyDataset, jl[jl_name])
                jl_val = ds[()]
                metadatas[metadata_name] = jnp.array(jl_val)
        # adjust parameters
        adjust_parameters_load = {}
        for par_name, jl_name in names_mapping['adjust_params'].items():
            if jl_name in jl.keys():
                ds = cast(H5pyDataset, jl[jl_name])
                jl_val = ds[()]
                if isinstance(jl_val, np.ndarray) or isinstance(jl_val, int) or \
                isinstance(jl_val, float):
                    adjust_parameters_load[par_name] = jnp.array(jl_val)
        return cls(trajectory, case, metadatas), adjust_parameters_load

    @classmethod
    def _load_nc(
            cls,
            nc_path: str,
            names_mapping: Dict[str, Dict[str, str]],
        ) -> Tuple[Data, Dict[str, jnp.ndarray]]:
        """
        Pre-load of a netCDF file.

        This class method build a trajectory and the physical parameters from the :code:`.nc`
        file :code:`nc_path`. :code:`names_mapping` is used to do the link between Tunax name
        convention and the one from the used database. This is only a pre-load because the user
        might want to adjust the Data later with the method :meth:`load`.

        Parameters
        ----------
        nc_path : str
            Path of the *netcdf* file that contains the time-series of the observation trajectory.
            The file should contains at least the three dimensions :attr:`~space.Grid.zr`,
            :attr:`~space.Grid.zw` and :attr:`~space.Trajectory.time`. The time-series must have the
            good dimensions described in :class:`~space.Trajectory`.
        names_mapping : Dict[str, Dict[str, str]]
            Contains the link between the Tunax names of variables and the path of the variables in
            the file. There are 4 first entries :
            - :code:`variables` : for all the variables corresponding to the :class:`space.Grid` and
              the :class:`space.Trajectory`.
            - :code:`parameters` : for all the scalar entries corresponding directly to the
              parameters of :class:`case.Case`
            - :code:`metadatas` : for the entries that we want to keep in the :attr:`metadatas` for
            later.
            - :code:`adjust_params` : for the parameters entries that are used by the adjusting
            function later.
        
        Returns
        -------
        data : Data
            An object containing the trajectory, the physical case and the metadata in this file.
        adjust_parameters_load : Dict[str, jnp.ndarray]
            An dictionnary with the parameters of an eventual adjusting function used in
            :meth:`load`.
        """
        var_map = names_mapping['variables']
        par_map = names_mapping['parameters']
        # read .nc file
        ds = xr.load_dataset(nc_path)
        # load grid et time
        zr = jnp.array(ds[var_map['zr']].values)
        zw = jnp.array(ds[var_map['zw']].values)
        grid = Grid(zr, zw)
        time = jnp.array(ds[var_map['time']].values)
        # load variables
        variables_dict = {}
        for var_name in VARIABLE_NAMES:
            if var_name not in var_map:
                continue
            variables_dict[var_name] = jnp.array(ds[var_map[var_name]].values)
        # generate trajectory
        trajectory = Trajectory(grid, time, **variables_dict)
        # generate case
        params = {}
        for par_name, nc_name in par_map.items():
            if nc_name in ds.keys():
                params[par_name] = float(ds[nc_name])
        case = Case(**params)
        # generate metadatas
        metadatas = {}
        for metadata_name, nc_name in names_mapping['metadatas'].items():
            if nc_name in ds.keys():
                metadatas[metadata_name] = jnp.array(ds[nc_name])
        # adjust parameters
        adjust_parameters_load = {}
        for par_name, nc_name in names_mapping['adjust_params'].items():
            if nc_name in ds.keys():
                adjust_parameters_load[par_name] = jnp.array(ds[nc_name])
        return cls(trajectory, case, {}), adjust_parameters_load
    
    @classmethod
    def load(
            cls,
            file_path: str,
            names_mapping: Dict[str, Dict[str, str]],
            adjust_fun: Optional[Callable[[Data, Dict[str, Any]], Data]] = None,
            adjust_fun_pars_out: Optional[Dict[str, Any]] = None,
            nz: Optional[int] = None,
            dims_mapping: Union[DimsType, Dict[str, DimsType]] = (None,)
        ) -> Data:
        """
        Creates a :class:`Data` instance from a :code:`.jld2` or a :code:`.nc` file.

        A trajectory, a physical case, and some metadatas are loaded from the file to create the
        Tunax Data instance. Some parameters can be loaded from the file to be used in a adjusting
        function which deals with the parameters and attribute that needs a transformation after
        loading (for example variable forcings). The rtacers for the equation of state (eos) are
        automatically chosen with the presence of the tracers in the file (by order of priority 'b',
        't', 's' and 'ts' are chosen). do_pt is set on True if there is a passive tracer on the
        file.

        Parameters
        ----------
        file_path : str
            Path of the file to load that contains the time-series of the observation trajectory
            and the physical parameters and forcings. It must end with :code:`.jld2` or a
            :code:`.nc`.
        names_mapping : Dict[str, Dict[str, str]]
            Contains the link between the Tunax names of variables and the path of the variables in
            the file. There are 4 first entries :
            - :code:`variables` :for all the variables corresponding to the :class:`space.Grid` and
              the :class:`space.Trajectory`.
            - :code:`parameters` : for all the scalar entries corresponding directly to the
              parameters of :class:`case.Case`
            - :code:`metadatas` : for the entries that we want to keep in the :attr:`metadatas` for
            later.
            - :code:`adjust_params` : for the parameters entries that are used by the adjusting
            function later.
            For :code:`.nc` files these are strings, for :code:`.jld2` the syntax is special, cf.
            :meth:`_load_jld2` method.
        adjust_fun : Callable[[Data, Dict[str, Any]], Data], optionnal, default=None
            This is a adjusting function, taking the loaded data and some parameters in a
            dictionnary, it returns a modified version of the data. This function can be used to
            implement variable forcings or special transformations of parameters. The parameters are
            the mix of :par:`adjust_fun_pars_out` and the parameters that are loaded from the file
            with the :code:`adjust_params` of :par:`names_mapping`.
        adjust_fun_pars_out : Dict[str, Any], optionnal, default = None
            Parameters to put in :par:`adjust_fun`.
        nz : int, optionnal, default=None
            Only for :code:`.jld2` files, cf. :meth:`_load_jld2` method.
        dims_mapping : DimsType or Dict[str, DimsType], optionnal, default=(None,)
            Only for :code:`.jld2` files, cf. :meth:`_load_jld2` method.
        
        Returns
        -------
        data : Data
            An object that represent these file.
        """
        if Path(file_path).suffix == '.nc':
            data, adjust_pars_load = cls._load_nc(file_path, names_mapping)
        elif Path(file_path).suffix == '.jld2':    
            data, adjust_pars_load = cls._load_jld2(file_path, names_mapping, nz, dims_mapping)
        else:
            raise ValueError(_format_to_single_line(f"""
                The load method only handle .jld2 and .nc files.
            """))
        # check for constant time step
        time = data.trajectory.time
        steps = time[1:] - time[:-1]
        if not jnp.all(steps == steps[0]):
            raise ValueError('Tunax only handle constant output time-steps')
        # detection of passive tracer
        if data.trajectory.pt is not None:
            data = replace(data, case=replace(data.case, do_pt=True))
        # detection of eos tracers
        if data.trajectory.b is not None:
            data = replace(data, case=replace(data.case, eos_tracers='b'))
        elif data.trajectory.t is not None and data.trajectory.s is None:
            data = replace(data, case=replace(data.case, eos_tracers='t'))
        elif data.trajectory.t is None and data.trajectory.s is not None:
            data = replace(data, case=replace(data.case, eos_tracers='s'))
        elif data.trajectory.t is not None and data.trajectory.s is not None:
            data = replace(data, case=replace(data.case, eos_tracers='ts'))
        else:
            raise ValueError('There must be at least one tracer for equation of state')
        # adjusting data
        if adjust_fun is not None:
            if adjust_fun_pars_out is None :
                adjust_fun_pars_out = {}
            adjust_pars = adjust_pars_load | adjust_fun_pars_out
            data = adjust_fun(data, adjust_pars)
        return data
        

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

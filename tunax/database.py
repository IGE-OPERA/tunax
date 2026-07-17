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
from collections.abc import Callable
from typing import Any, cast

import xarray as xr
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from h5py import File as H5pyFile
from h5py import Group as H5pyGroup

from tunax.space import Grid, Trajectory, VARIABLE_NAMES
from tunax.case import Case
from tunax.functions import _format_to_single_line
from tunax.model import SingleColumnModel


type TransAxisType = tuple[tuple[bool, int, tuple[int, ...]], ...]
"""Details the projection, transposition and slicing of an array, cf. :meth:`Data.transform_arr`"""
type TransAxisParType = tuple[tuple[bool, int, tuple[int | str, ...]], ...]
"""Details the projection, transposition and slicing of an array, parametrisable version."""


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
    metadatas : dict[str, float :class:`~jax.Array`]
        Some metadatas that we want to use later. They can be floats or arrays but always written in
        a :class:`jax.Array`.

    Raises
    ------
    ValueError
        If the :attr:`~space.Trajectory.time` of :attr:`trajectory` is not build with constant
        time-steps.

    """

    trajectory: Trajectory
    case: Case
    metadatas: dict[str, Array] = eqx.field(static=True)

    @staticmethod
    def transform_arr(
            arr: Array,
            trans_axis: TransAxisType
        ) -> Array:
        """
        Projection, transposition and slice of the input array.

        The input array `arr` is transformed by projection on some axis, transposition of the
        resulting axis, and eventual slicing of them. There is an option for automatic slice giving
        expected size of the output array.

        Parameters
        ----------
        arr : float :class:`~jax.Array`
            Array on which do the projection.
        trans_axis : TransAxisType
            This object contains the information of the transformation of each axis. The number of
            element must be the same as the number of axis of `arr`. Each element has 3 components :
            - A boolean, indicates whether we keep this axis on the ouput (if True), or if we
            project on a certain index of this axis.
            - An integer. If the boolean is True, it indicates the position of the axis in the
            output array (there will be a transposition if these numbers are not in the order). I
            the boolean is False, it indicates the index on which do the projection of `arr` for the
            output array.
            - A tuple of integer indicating the slicing to do. There are 4 possibilities :
                - If the tuple is empty, there is no slicing, all the data of the axis is kept.
                - If the tuple has 1 integer, it indicates the expected size of the array, it will
                keep the center of array by removing symetrically the bounds of the array on this
                axis.
                - If the tuple has 2 or 3 integers, it behaves like a normal slice (start, stop) or
                (start, stop, step).
        
        Returns
        -------
        transformed_arr : :class:`~jax.Array`
            Array `arr` projected, transposed and sliced on all the axis.
        """
        if len(arr.shape) != len(trans_axis):
            raise ValueError(_format_to_single_line("""
                The tuple parameter `trans_axis` must have the length of the number of dimensions of
                the input array `arr`.
            """))
        slices = []
        trans = []
        for i_axis, axis in enumerate(trans_axis):
            do_axis, i_proj_trans, sl_i = axis
            if do_axis:
                if len(sl_i) == 0:
                    sl = slice(None)
                elif len(sl_i) == 1:
                    size_in, size_out = arr.shape[i_axis], sl_i[0]
                    double_bouds = size_in - size_out
                    if double_bouds < 0:
                        raise ValueError(_format_to_single_line(f"""
                            The size of the input array is smaller than the expected size of the
                            output array for axis n°{i_axis}.
                        """))
                    if double_bouds%2 == 1:
                        warnings.warn(_format_to_single_line(f"""
                            The size of the input array minus the expected size of the output array
                            is an odd number : the removed boundaries are taken 1 point thinner on
                            the left than on the right, for axis n°{i_axis}.
                        """))
                    bounds = double_bouds//2
                    sl = slice(bounds, bounds+size_out)
                elif len(sl_i) == 2 or len(sl_i) == 3:
                    sl = slice(*sl_i)
                else:
                    raise ValueError(_format_to_single_line(f"""
                            The number of element in the slice typle of `trans_axis` for axis
                            n°{i_axis} must be at maximum 3.
                        """))
                slices.append(sl)
                trans.append(i_proj_trans)
            else:
                slices.append(i_proj_trans)
        proj_sliced_arr = arr[tuple(slices)]
        return jnp.transpose(proj_sliced_arr, axes=trans)

    @staticmethod
    def _get_val(
            file: H5pyFile | xr.Dataset,
            var_name: str,
            var_map: dict[str, str],
            trans_axis_mapping: dict[str, TransAxisType] | None = None,
            suffix: str = ''
        ) -> Array:
        """
        Shortcut for the loading a data in a read file.

        Parameters
        ----------
        file : H5pyFile or xr.Dataset
            Imported file, works with .jld2 file (type = H5pyFile) or .nc file (type=xr.Dataset).
        var_name : str
            Name of the variable in Tunax.
        data_map : dict[str, str]
            Contains the link between the Tunax names of variables and the path of the variables in
            the file. This is on of the 4 sub dictionnary of the parameter `names_mapping` of
            `Data._load`.
        trans_axis_mapping : dict[str, TransAxisType], default = None
            Mapping of the transformation (projection, transposition and slicing) for each variable
            that has to be transformed by Data.transform_arr. If `var_name` is not in this
            dictionnary or if this parameter is None, no transformation is applied.
        suffix : str = ''
            Suffix to add at the end of the name in the file. Can be used for the variables that
            are separated in time for example.
        
        Returns
        -------
        arr : :class:`~jax.Array`
            Array of the variable `var_name` in the `file` potentially transformed by the parameters
            in `trans_axis_mapping`.
        """
        arr = jnp.array(file[f'{var_map[var_name]}{suffix}'])
        if trans_axis_mapping is None or var_name not in trans_axis_mapping.keys():
            return arr
        return Data.transform_arr(arr, trans_axis_mapping[var_name])

    @staticmethod
    def replace_nz_expr(sl: tuple[int | str, ...], nz: int) -> tuple[int, ...]:
        """
        Replace 'nz' by the value of `nz` in the tuple `sl`.

        This method only modify the tuple when it's a singleton containing the string 'nz' plus or
        minus 1 or 2. If it's another string or or if `sl` doesn't contains only integers, it will
        returns an error.

        Parameters
        ----------
        sl : tuple of integers or string
            Can be a singleton with the string 'nz' (+ or - 1 or 2) or a tuple with only integers.
        nz : int
            Value of nz to replace in the tuple.
        
        Returns
        -------
        sl_eval : tuple of integers
            Tuple `sl` with the replaced value of `nz` in the 'nz' string.
        """
        if len(sl) == 1:
            match sl[0]:
                case 'nz':
                    return (nz,)
                case 'nz+1':
                    return (nz+1,)
                case 'nz+2':
                    return (nz+2,)
                case 'nz-1':
                    return (nz-1,)
                case 'nz-2':
                    return (nz-2,)
            if isinstance(sl[0], str):
                raise ValueError(_format_to_single_line("""
                    Only nz + or - 1 or 2 is possible for transformation.
                """))
            return cast(tuple[int, ...], sl)
        else:
            if all(isinstance(x, int) for x in sl):
                return cast(tuple[int, ...], sl)
            raise ValueError(_format_to_single_line("""
                If the input tuple contains more than 1 element, they all must be integers.
            """))

    @classmethod
    def load(
            cls,
            file_path: Path | str,
            names_mapping: dict[str, dict[str, str]],
            trans_axis_mapping: dict[str, TransAxisParType] | None = None,
            adjust_fun: Callable[[Data, dict[str, Any]], Data] | None = None,
            adjust_fun_pars_out: dict[str, Any] | None = None,
            time_sep: bool = False
        ) -> Data:
        """
        Global loader of :code:`.nc` and :code:`.jld2` files in a Data object.

        The file must contain at least the element to build a :class:`~space.Grid` and a
        :class:`~space.Trajectory`. It might also contains some parameters used to describe the
        :class:`~case.Case`, and some other used do add :attr:`metadatas`. It uses the mapping
        :code:`names_mapping` to make the link between the Tunax variables and the keys to access to
        the file variables. It is also possible to adjust the Data obtained by the simple file
        loading, using the function :code:`adjust_fun` which can modify all the Data, for example
        if we want to compute variable forcings we have to use it. Note that this function  can
        take arguments read from the file and arguments give by the user in
        :code:`ajust_fun_pars_out`. If the data is the file needs some transformation before going
        in the :class:`~space.Trajectory`, one can use the parameter :code:`trans_axis_mapping`.
        Some :code:`.jld2` files come with the timeseries separated in the different time-steps, one
        can use :code:`time_sep` to handle this. Finally, the presence of a passive tracer
        (:attr:`~case.Case.do_pt`) and the equation of state tracers
        (:attr:`~case.Case.eos_tracers`) are automatically detected (by order of priority 'b',
        't', 's' and 'ts').

        arameters
        ----------
        file_path : Path or str
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
            For :code:`.nc` files these are strings, for :code:`.jld2` they can use '/' to describe
            the tree structure.
        trans_axis_mapping : dict[str, TransAxisParType], optionnal
            Can contain the information of the transformation (projection, transposition and
            slicing) to apply when the variables are loaded. For each variable that has to be
            transformed, an object TransAxisParType is used, check :meth:`transform_arr` for the
            syntax. There is a possibility to just ask for an array lenght. Note that :code:`nz` can
            be imported from the file and its valuecan replace the strings :code:`'nz` in the object
            using :meth:`replace_nz_expr`.
        adjust_fun : Callable[[Data, Dict[str, Any]], Data], optionnal
            This is a adjusting function, taking the loaded data and some parameters in a
            dictionnary, it returns a modified version of the data. This function can be used to
            implement variable forcings or special transformations of parameters. The parameters are
            the mix of :par:`adjust_fun_pars_out` and the parameters that are loaded from the file
            with the :code:`adjust_params` of :par:`names_mapping`.
        adjust_fun_pars_out : Dict[str, Any], optionnal
            Parameters to put in :par:`adjust_fun` in addition to those read in the file.
        time_sep : bool, default=False
            Works only for :code:`.jld2` files. If the timeseries of the variables are indexed as
            one 1D array by time-step, put this attribute to True. It will read the indexes of the
            time and read the 1D arrays separately to concatenate them at the end.
        
        Returns
        -------
        data : Data
            An object that represent these file.

        Warning
        -------
        Be aware, this function ignore the parameters that are not convertible in JAX arrays.
        """
        # read file
        if Path(file_path).suffix == '.nc':
            file = xr.load_dataset(file_path)
        elif Path(file_path).suffix == '.jld2':
            file = H5pyFile(file_path, 'r')
        else:
            raise ValueError(_format_to_single_line("""
                The load method only handle .jld2 and .nc files.
            """))
        # mappings
        var_map = names_mapping['variables']
        par_map = names_mapping['parameters']
        md_map = names_mapping['metadatas']
        ap_map = names_mapping['adjust_params']
        # load nz
        if 'nz' in var_map.keys():
            nz = int(Data._get_val(file, 'nz', var_map))
        else:
            trans_axis_mapp = cast(dict[str, TransAxisType] | None, trans_axis_mapping)
            zr = Data._get_val(file, 'zr', var_map, trans_axis_mapp)
            nz = zr.shape[0]
        # replace potential 'nz' in trans_axis_mapping
        if trans_axis_mapping is not None:
            trans_axis_mapping = {
                var: tuple((b, i, Data.replace_nz_expr(sl, nz)) for b, i, sl in proj_axis)
                for var, proj_axis in trans_axis_mapping.items()
            }
        # load grid
        trans_axis_mapp = cast(dict[str, TransAxisType] | None, trans_axis_mapping)
        zr = Data._get_val(file, 'zr', var_map, trans_axis_mapp)
        zw = Data._get_val(file, 'zw', var_map, trans_axis_mapp)
        # load time
        if time_sep:
            gr = file[var_map['time']]
            assert isinstance(gr, H5pyGroup)
            time_int_sorted_list = sorted([int(i) for i in list(gr.keys())])
            time_str_list = [str(i) for i in time_int_sorted_list]
            time_float_list = []
            for time_str in time_str_list:
                time_float_list.append(float(Data._get_val(
                    file, 'time', var_map, None, f'/{time_str}')
                ))
            time = jnp.array(time_float_list)
        else:
            time = Data._get_val(file, 'time', var_map, trans_axis_mapp)
        # load variables
        variables_dict = {}
        for var_name in VARIABLE_NAMES:
            if var_name not in var_map:
                continue
            if time_sep:
                var_list = []
                for time_str in time_str_list: # pyright: ignore[reportPossiblyUnboundVariable]
                    var_list.append(Data._get_val(
                        file, var_name, var_map, trans_axis_mapp, f'/{time_str}'
                    ))
                variables_dict[var_name] = jnp.vstack(var_list)
            else:
                variables_dict[var_name] = Data._get_val(
                    file, var_name, var_map, trans_axis_mapp
                )
        # generate trajectory
        trajectory = Trajectory(Grid(zr, zw), time, **variables_dict)
        # generate parameters
        params = {}
        for par_name, file_par_name in par_map.items():
            if file_par_name in file.keys():
                try:
                    params[par_name] = float(Data._get_val(file, par_name, par_map))
                except TypeError:
                    pass
        case = Case(**params)
        # generate metadatas
        metadatas = {}
        for metadata_name, file_md_name in md_map.items():
            if file_md_name in file.keys():
                metadatas[metadata_name] = Data._get_val(file, metadata_name, md_map)
        # adjust parameters
        adjust_pars_load = {}
        for par_name, file_par_name in ap_map.items():
            if file_par_name in file.keys():
                try:
                    adjust_pars_load[par_name] = Data._get_val(file, par_name, ap_map)
                except TypeError:
                    pass
        # creation of data instance
        data = cls(trajectory, case, metadatas)
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

    def cut(self, out_nt_cut: int) -> list[Data]:
        """
        Cuts the :attr:`Trajectory` in sub-trajectories, cf. :meth:`space.Trajectory.cut`.

        Parameters
        ----------
        out_nt_cut : int
            Number of output steps of the sub-trajectories.
        
        Returns
        -------
        traj_list : list[Data]
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

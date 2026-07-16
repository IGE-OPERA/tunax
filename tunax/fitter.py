"""
Abstraction usefull for the calibration of the closures.

This module contains the classes :class:`FittableParameter` and :class:`FittableParametersSet` which
are used to make a link between the optimization part and the closures structures. These class can
be obtained by the prefix :code:`tunax.fitter.` or directly by :code:`tunax.`.

"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float, Array

from tunax.closure import ClosureParametersAbstract, Closure
from tunax.closures_registry import CLOSURES_REGISTRY


class FittableParameter(eqx.Module):
    """
    Calibration configuration for one parameter.

    An instance of this class must be created for every parameter of the
    closure that will be calibrated and for every not default values during
    calibration. The constructor takes all the attributes as parameters.

    Attributes
    ----------
    do_fit : bool
        The parameter will be calibrated.
    val : float, default=0.
        If :attr:`do_fit` : the initial value for calibration (at the first
        step of the calibration) ; if :attr:`do_fit` is false : the constant
        value to take for this parameter if it's not the default one in the
        closure.

    """

    do_fit: bool
    val: float = 0.


class FittableParametersSet(eqx.Module):
    """
    Complete closure calibration parameters.

    This class is the set of all the configurations on the closure parameters
    for the calibration. It makes the link between the array on which the
    optimizer works and the closure parameters class.

    Parameters
    ----------
    coef_fit_dict : dict[str, FittableParameter]
        cf. :attr:`coef_fit_dict`.
    closure_name : str
        Name of the chosen closure, must be a key of
        :attr:`~closures_registry.CLOSURES_REGISTRY`, see its documentation
        for the available closures.

    Attributes
    ----------
    coef_fit_dico : dict[str, FittableParameter]
        The set of all the configurations of all the parameters that will be
        calibrated and the one constants but not with the default value of the
        closure.
    closure : Closure
        The abstraction that represent the used closure.
    
    """

    coef_fit_dict: dict[str, FittableParameter]
    closure: Closure

    def __init__(
            self,
            coef_fit_dict: dict[str, FittableParameter],
            closure_name: str
        ) -> None:
        self.coef_fit_dict = coef_fit_dict
        self.closure = CLOSURES_REGISTRY[closure_name]

    @property
    def n_calib(self) -> int:
        """
        Number of variables that are calibrated.

        Returns
        -------
        nc : int
            Number of variables that are calibrated.
        """
        nc = 0
        for coef_fit in self.coef_fit_dict.values():
            if coef_fit.do_fit:
                nc += 1
        return nc

    def fit_to_closure(
            self,
            x: Float[Array, 'nc']
        ) -> ClosureParametersAbstract:
        """
        Transforms an fitted array in a set of closure parameters.

        This method copy the fixed non-default values of :attr:`coef_fit_dict`
        and copy the calibrated values from :code:`x`. Which is simply the
        parameters values in the order that is indicated by
        :attr:`coef_fit_dict`.

        Parameters
        ----------
        x : float :class:`~jax.Array` of shape (nc)
            The array on which the optimize works to find the best values. It
            is the array of the parameters that are calibrated.

        Returns
        -------
        clo_params : ClosureParametersAbstract
            The instance of the closure parameters class (child class of
            :class:`~closure.ClosureParametersAbstract`) with the modifications
            of the calibration step.
        """
        clo_coef_dico = {}
        i_x = 0
        for coef_name, coef_fit in self.coef_fit_dict.items():
            if coef_fit.do_fit:
                clo_coef_dico[coef_name] = x[i_x]
                i_x += 1
            else:
                clo_coef_dico[coef_name] = coef_fit.val
        return self.closure.parameters_class(**clo_coef_dico)

    def gen_init_val(self) -> Float[Array, 'nc']:
        """
        Produce the fitted array for the first calibration step.

        This method simply copy the initial values of the calibrated
        coefficients in an array :code:`x` which will be used as the first
        calibration step for the optimizer.

        Returns
        -------
        x : float :class:`~jax.Array` of shape (nc)
            The initial vector for the optimizer at the first step of
            calibration.
        """
        x = []
        for coef_fit in self.coef_fit_dict.values():
            if coef_fit.do_fit:
                x.append(coef_fit.val)
        return jnp.array(x)

r"""
Registry of available closures.

This module only contains a constant variable which lists all the available closures. It can be
obtained by the prefix :code:`tunax.closures_registry.` or directly by :code:`tunax.`.

Attributes
==========
CLOSURES_REGISTRY : dict[str, Closure]
    This variable is a dictionnary whose keys are the name of the closures (they must be exaclty the
    same than :attr:`~closure.Closure.name`), and whose values are the corresponding
    :class:`~closure.Closure` instance of the closure. The closure can be used by the forward model
    and the fitter thanks to this constant. When the user add a the code for a closure in
    :code:`closures/` they must

    1. import here the parameters and state classes (child of
       :class:`~closure.ClosureParametersAbstract` and of :class:`~closure.ClosureStateAbstract`)
       and the step function of the closure,
    2. from these objects, create the :class:`~closure.Closure` instance of the closure,
    3. add an entry at this dictionnary

    The current available closures are :
    
    - :code:`k-epsilon` for :math:`k-\varepsilon` closure cf. :mod:`closures.k_epsilon`

"""

from tunax.closure import Closure
from tunax.closures.k_epsilon import KepsParameters, KepsState, keps_step


CLOSURES_REGISTRY: dict[str, Closure] = {
    'k-epsilon': Closure('k-epsilon', KepsParameters, KepsState, keps_step)
}

"""
Usefull calculation functions.

The functions in this module are supposed to be used in various other modules. They can be called by
the prefix :code:`tunax.functions.` or directly by :code:`tunax.`.
"""

import jax.numpy as jnp
from jax import lax, Array
from jaxtyping import Float


type FloatJax = Float[Array, '']
"""Type that represent a float in a :class:`~jax.Array`, used only for the code linter."""


def tridiag_solve(
        a: Float[Array, 'n'],
        b: Float[Array, 'n'],
        c: Float[Array, 'n'],
        f: Float[Array, 'n']
    ) -> Float[Array, 'n']:
    r"""
    Solve a trigiagonal problem.

    The tridiagonal problem can be written :math:`\mathbb MX = F` where
    :math:`\mathbb M = \begin{pmatrix} b_1 & c_1 & & \\
    a_2 & \ddots & \ddots & \\
    & \ddots & \ddots & c_{n-1} \\
    & & a_n & b_n
    \end{pmatrix}`
    and :math:`F = \begin{pmatrix} f_1 \\ \vdots \\ f_n \end{pmatrix}`. The problem is solved by
    recurrence using :mod:`jax.lax.scan` 

    Parameters
    ----------
    a : float :class:`~jax.Array` of shape (n)
        Left diagonal of :math:`\mathbb M`, the first element is not used.
    b : float :class:`~jax.Array` of shape (n)
        Middle diagonal of :math:`\mathbb M`.
    c : float :class:`~jax.Array` of shape (n)
        Right diagonal of :math:`\mathbb M`, the last element is not used.
    f : float :class:`~jax.Array` of shape (n)
        Right hand of the equation :math:`F`.
    
    Returns
    -------
    x : float :class:`~jax.Array` of shape (n)
        Solution :math:`X` of tridiagonal problem.
    """
    def forward_scan_scal(
        carry: tuple[FloatJax, FloatJax],
        x: Float[Array, '4']
    ) -> tuple[tuple[FloatJax, FloatJax], tuple[FloatJax, FloatJax]]:
        f_im1, q_im1 = carry
        a, b, c, f = x
        cff = 1./(b+a*q_im1)
        f_i = cff*(f-a*f_im1)
        q_i = -cff*c
        carry = f_i, q_i
        return carry, carry
    init = f[0]/b[0], -c[0]/b[0]
    xs = jnp.stack([a, b, c, f])[:, 1:].T
    _, (f, q) = lax.scan(forward_scan_scal, init, xs)
    f = jnp.concat([init[0], f])
    q = jnp.concat([init[1], q])

    def reverse_scan_scal(carry: FloatJax, x: Float[Array, '2']) -> tuple[FloatJax, FloatJax]:
        q_rev, f_rev = x
        carry = f_rev + q_rev*carry
        return carry, carry
    init = f[-1]
    xs = jnp.stack([q[::-1], f[::-1]])[:, 1:].T
    _, x = lax.scan(reverse_scan_scal, init, xs)
    x = jnp.concat([init, x])

    return x[::-1]


def add_boundaries(
        vec_btm: float | FloatJax,
        vec_in: Float[Array, 'n-2'],
        vec_sfc: float | FloatJax
    ) -> Float[Array, 'n']:
    """
    Concatenate the three parts of a vector : surface, bottom and inside.

    This functions is made to avoid loops and make JAX more efficient by writing the calculations
    with vectorization when possible.

    Parameters
    ----------
    vec_btm : float or :class:`~jax.Array` of shape ().
        Bottom value of the vector.
    vec_in : float :class:`~jax.Array` of shape (n-2)
        Middle values of the vector.
    vec_sfc : float or :class:`~jax.Array` of shape ().
        Surface value of the vector, must have the same type than :code:`vec_btm`.

    Returns
    -------
    vec : float :class:`~jax.Array` of shape (n)
        Concatenated vector.
    """
    if isinstance(vec_btm, float):
        return jnp.concat([jnp.array([vec_btm]), vec_in, jnp.array([vec_sfc])])
    else:
        return jnp.concat([vec_btm, vec_in, vec_sfc])


def _format_to_single_line(text: str) -> str:
    """
    Transforms a multiple line text in a line string by removing indentations.

    In the code the error and warning messages are written on multiple lines with indentations to
    respect the PEP8 maximum line length of 79 characters and the consistency of indentations. This
    function is used to show correctly these messages on one line and without the indentations.

    Parameters
    ----------
    text : str
        Text on multiple lines to transform.

    Returns
    -------
    line : str
        Text on a single line removed from indentations.
    """
    lines = text.splitlines()
    stripped_lines = [line.strip() for line in lines]
    single_line = " ".join(stripped_lines)
    return " ".join(single_line.split())

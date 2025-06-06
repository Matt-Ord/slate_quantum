from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)

if TYPE_CHECKING:
    from slate_core import TupleMetadata

    from slate_quantum.operator._operator import Operator, OperatorBasis

import numpy as np

from slate_quantum.operator._build._momentum import p as build_p
from slate_quantum.operator._build._position import x as build_x
from slate_quantum.operator._linalg import matmul


def caldeira_leggett_shift[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    *,
    friction: float,
) -> Operator[
    OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
]:
    r"""
    Build the shift operator for the Caldeira-Leggett model.

    This is the required Hamiltonian-like shift to the Caldeira-Leggett master equation.
    The shift operator is defined as:
    .. math::
        S = \\frac{\\gamma}{2} \\left(\\hat{p}\\hat{x} + \\hat{x}\\hat{p}\\right)

    where :math:`\\gamma` is the friction coefficient, :math:`\\hat{p}` is the momentum operator,
    and :math:`\\hat{x}` is the position operator.
    """
    assert metadata.n_dim == 1, "Currently only supports 1D systems."
    x = build_x(metadata, axis=0)
    p = build_p(metadata, axis=0)
    return ((matmul(p, x) + matmul(x, p)) * (friction / 2)).as_type(np.complex128)


def caldeira_leggett_collapse[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    *,
    friction: float,
    temperature: float,
    mass: float,
) -> Operator[
    OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
]:
    r"""
    Build the collapse operator for the Caldeira-Leggett model.

    This is the required collapse operator for the Caldeira-Leggett master equation.
    The collapse operator is defined as:
    .. math::
        \\hat{L} = \\sqrt{\\frac{4 m \\gamma k_B T}{\\hbar^2}} \\hat{x} + i \\sqrt{\\frac{\\gamma}{4 m k_B T}} \\hat{p}

    where :math:`\\gamma` is the friction coefficient, :math:`k_B` is the Boltzmann constant,
    :math:`T` is the temperature, :math:`\\hat{x}` is the position operator, and
    :math:`\\hat{p}` is the momentum operator.

    This is a Lindblad-type operator using the convention
    .. math::
        D(\hat{L}) = \hat{L} \rho \hat{L}^\dagger - \frac{1}{2} ...
    """
    assert metadata.n_dim == 1, (
        "For systems with more than 1D, there will be multiple collapse operators."
    )
    kb_t = Boltzmann * temperature

    x = build_x(metadata, axis=0)
    x_prefactor = complex(np.sqrt(4 * mass * friction * kb_t / hbar**2))
    p = build_p(metadata, axis=0)
    p_prefactor = -1j * complex(np.sqrt(friction / (4 * mass * kb_t)))
    return (x * x_prefactor + p * p_prefactor).as_type(np.complex128)

from __future__ import annotations

from typing import Any, TypedDict, Unpack

import numpy as np
from scipy.constants import hbar  # type: ignore[import]
from slate import Array, StackedMetadata, linalg
from slate.metadata import AxisDirections, SpacedLengthMetadata

from slate_quantum.operator._build._momentum import k as build_k
from slate_quantum.operator._build._position import x as build_x
from slate_quantum.operator._build._potential import identity as build_identity
from slate_quantum.operator._operator import Operator


class BosonOptions(TypedDict, total=False):
    axis: int
    x_offset: float
    wrapped_x: bool


def create_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    factor: float = 1,
    *,
    axis: int = 0,
    x_offset: float = 0,
    wrapped_x: bool = False,
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic creation operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} - i* factor * \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    x = build_x(metadata, axis=axis, offset=x_offset, wrapped=wrapped_x)
    d_x = build_k(metadata, axis=axis)
    return (x * np.sqrt(factor) - d_x * (1j / np.sqrt(factor))) * (1 / (2**0.5))


def create_harmonic_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    m_omega: float,
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic creation operator for a harmonic particle."""
    return create_boson(metadata, factor=m_omega / hbar, **kwargs)


def create_vaccum_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic creation operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} - i \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    return create_boson(metadata, **kwargs)


def annhialate_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    factor: float = 1,
    *,
    axis: int = 0,
    x_offset: float = 0,
    wrapped_x: bool = False,
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic annhialate operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} + i* factor * \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    x = build_x(metadata, axis=axis, offset=x_offset, wrapped=wrapped_x)
    d_x = build_k(metadata, axis=axis)
    return (x * np.sqrt(factor) + d_x * (1j / np.sqrt(factor))) * (1 / (2**0.5))


def annhialate_harmoic_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    m_omega: float,
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic annihilate operator for a harmonic particle."""
    return annhialate_boson(metadata, factor=m_omega / hbar, **kwargs)


def annhialate_vaccum_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic annihilate operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} + i \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    return annhialate_boson(metadata, **kwargs)


def normal_ordered_boson[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    factor: float = 1,
    *,
    n_create: int = 1,
    n_annhialate: int = 1,
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic annhialate operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} + i* factor * \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    create = create_boson(metadata, factor=factor, **kwargs)
    out: Array[Any, Any] = build_identity(metadata)
    for _ in range(n_create):
        # TODO: array pow and array mul implementation
        out = linalg.einsum(
            "(i j),(j' k)->(i k)",
            create,
            out,
        )

    annhialate = annhialate_boson(metadata, factor=factor, **kwargs)
    for _ in range(n_annhialate):
        out = linalg.einsum(
            "(i j),(j' k)->(i k)",
            annhialate,
            out,
        )
    return Operator(out.basis, out.raw_data)


def number[M: SpacedLengthMetadata, E: AxisDirections](
    metadata: StackedMetadata[M, E],
    factor: float = 1,
    **kwargs: Unpack[BosonOptions],
) -> Operator[StackedMetadata[M, E], np.complexfloating]:
    r"""Build the bosonic annhialate operator for a free particle.

    This is equivalent to
    ...math:: \\hat{a}^\\dagger = \frac{1}{\\sqrt{2}}(\\hat{x} + i* factor * \\hat{\\partial_x})
    where :math:`\\hat{x}` is the position operator.
    """
    return normal_ordered_boson(
        metadata, factor=factor, n_create=1, n_annhialate=1, **kwargs
    )

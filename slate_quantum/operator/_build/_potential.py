from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never

import numpy as np
from slate_core import BasisMetadata, TupleMetadata, basis, ctype
from slate_core import metadata as _metadata
from slate_core.basis import (
    CroppedBasis,
    TruncatedBasis,
    Truncation,
    TupleBasisLike,
)
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum._util import outer_product
from slate_quantum.metadata import RepeatedLengthMetadata, repeat_volume_metadata
from slate_quantum.operator._diagonal import (
    PositionOperatorBasis,
    Potential,
    position_operator_basis,
    with_outer_basis,
)
from slate_quantum.operator._operator import Operator, OperatorBuilder

if TYPE_CHECKING:
    from collections.abc import Callable


def potential[M: BasisMetadata, E, CT: ctype[Never], DT: np.dtype[np.generic]](
    outer_basis: TupleBasisLike[tuple[M, ...], E, CT], data: np.ndarray[Any, DT]
) -> OperatorBuilder[PositionOperatorBasis[M, E, CT], DT]:
    """Get the potential operator."""
    return Operator.build(position_operator_basis(outer_basis), data)


_build_potential = potential


def repeat_potential(
    potential: Potential[
        SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]
    ],
    shape: tuple[int, ...],
) -> Potential[RepeatedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Create a new potential by repeating the original potential in each direction."""
    transformed_basis = basis.transformed_from_metadata(
        potential.basis.inner.outer_recast.metadata()
    )
    as_transformed = with_outer_basis(potential, transformed_basis).ok()
    converted_basis = basis.transformed_from_metadata(
        repeat_volume_metadata(potential.basis.inner.outer_recast.metadata(), shape)
    )
    repeat_basis = basis.with_modified_children(
        converted_basis,
        lambda i, y: TruncatedBasis(
            Truncation(transformed_basis.children[i].size, shape[i], 0),
            y,
        )
        .resolve_ctype()
        .upcast(),
    ).upcast()
    return _build_potential(
        repeat_basis,
        as_transformed.raw_data * np.sqrt(np.prod(shape)),
    ).ok()


def cos_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Build a cosine potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).resolve_ctype().upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    data = outer_product(*(np.array([2, 1, 1]),) * n_dim)
    return potential(
        cropped, 0.25**n_dim * height * data * np.sqrt(transformed_basis.size)
    ).ok()


def sin_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Build a cosine potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).resolve_ctype().upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    data = outer_product(*(np.array([2, 1j, -1j]),) * n_dim)
    return potential(
        cropped, 0.25**n_dim * height * data * np.sqrt(transformed_basis.size)
    ).ok()


def _square_wave_points(
    n_terms: int, lanczos_factor: float
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    prefactor = np.sinc(2 * np.fft.fftfreq(n_terms)) ** lanczos_factor
    return prefactor * np.fft.ifft(
        np.sign(np.cos(np.linspace(0, 2 * np.pi, n_terms, endpoint=False)))
    )  # type: ignore can't infer shape of ifft


def square_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
    *,
    n_terms: tuple[int, ...] | None = None,
    lanczos_factor: float = 0,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Build a square potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda i, y: y
        if n_terms is None
        else CroppedBasis(n_terms[i], y).resolve_ctype().upcast(),
    ).upcast()

    data = outer_product(
        *(_square_wave_points(n, lanczos_factor) for n in cropped.inner.shape)
    )
    return potential(
        cropped,
        0.5 * height * data * np.sqrt(transformed_basis.size),
    ).ok()


def fcc_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Generate a potential suitable for modelling an fcc surface.

    This potential contains the lowest fourier components - however for an fcc surface
    there are only six degenerate fourier components.
    """
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).resolve_ctype().upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    # TODO: generalize to n_dim  # noqa: FIX002
    assert n_dim == 2  # noqa: PLR2004

    data = np.array([[3, 1, 1], [1, 1, 0], [1, 0, 1]])
    return potential(
        cropped,
        (1 / 3) ** n_dim * data * height * np.sqrt(transformed_basis.size),
    ).ok()


def potential_from_function[
    M: SpacedLengthMetadata,
    E: AxisDirections,
    DT: np.dtype[np.generic],
](
    metadata: TupleMetadata[tuple[M, ...], E],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, DT],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> Potential[M, E, DT]:
    """Get the potential operator."""
    positions = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=offset, wrapped=wrapped
    )

    return potential(basis.from_metadata(metadata).upcast(), fn(positions)).ok()


def harmonic_potential(
    metadata: SpacedVolumeMetadata,
    frequency: float,
    *,
    offset: tuple[float, ...] | None = None,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.dtype[np.complexfloating]]:
    """Build a harmonic potential.

    V(x) = 0.5 * frequency^2 * ||x||^2
    """
    return potential_from_function(
        metadata,
        lambda x: (0.5 * frequency**2 * np.linalg.norm(x, axis=0) ** 2),
        wrapped=True,
        offset=offset,
    )

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import basis
from slate_core import metadata as _metadata
from slate_core.basis import (
    CroppedBasis,
    TruncatedBasis,
    Truncation,
    fundamental_transformed_tuple_basis_from_metadata,
)
from slate_core.metadata import (
    AxisDirections,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)

from slate_quantum._util import outer_product
from slate_quantum.metadata import RepeatedLengthMetadata, repeat_volume_metadata
from slate_quantum.operator._diagonal import Potential

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_quantum._util.legacy import StackedMetadata


def repeat_potential(
    potential: Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating],
    shape: tuple[int, ...],
) -> Potential[RepeatedLengthMetadata, AxisDirections, np.complexfloating]:
    """Create a new potential by repeating the original potential in each direction."""
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(
        potential.basis.outer_recast.metadata()
    )
    as_transformed = potential.with_outer_basis(transformed_basis)
    converted_basis = fundamental_transformed_tuple_basis_from_metadata(
        repeat_volume_metadata(potential.basis.outer_recast.metadata(), shape)
    )
    repeat_basis = basis.with_modified_children(
        converted_basis,
        lambda i, y: TruncatedBasis(
            Truncation(transformed_basis[i].size, shape[i], 0),
            y,
        ),
    )
    return Potential(
        repeat_basis,
        as_transformed.raw_data * np.sqrt(np.prod(shape)),
    )


def cos_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
    """Build a cosine potential."""
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y),
    )
    n_dim = len(cropped.shape)
    data = outer_product(*(np.array([2, 1, 1]),) * n_dim)
    return Potential(
        cropped,
        0.25**n_dim * height * data * np.sqrt(transformed_basis.size),
    )


def sin_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
    """Build a cosine potential."""
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y),
    )
    n_dim = len(cropped.shape)
    data = outer_product(*(np.array([2, 1j, -1j]),) * n_dim)
    return Potential(
        cropped,
        0.25**n_dim * height * data * np.sqrt(transformed_basis.size),
    )


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
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
    """Build a square potential."""
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda i, y: y if n_terms is None else CroppedBasis(n_terms[i], y),
    )

    data = outer_product(
        *(_square_wave_points(n, lanczos_factor) for n in cropped.shape)
    )
    return Potential(
        cropped,
        0.5 * height * data * np.sqrt(transformed_basis.size),
    )


def fcc_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
    """Generate a potential suitable for modelling an fcc surface.

    This potential contains the lowest fourier components - however for an fcc surface
    there are only six degenerate fourier components.
    """
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y),
    )
    n_dim = len(cropped.shape)
    # TODO: generalize to n_dim  # noqa: FIX002
    assert n_dim == 2  # noqa: PLR2004

    data = np.array([[3, 1, 1], [1, 1, 0], [1, 0, 1]])
    return Potential(
        cropped,
        (1 / 3) ** n_dim * data * height * np.sqrt(transformed_basis.size),
    )


def potential_from_function[M: SpacedLengthMetadata, E: AxisDirections, DT: np.generic](
    metadata: StackedMetadata[M, E],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[DT]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> Potential[M, E, DT]:
    """Get the potential operator."""
    positions = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=offset, wrapped=wrapped
    )

    return Potential(basis.from_metadata(metadata), fn(positions))


def harmonic_potential(
    metadata: SpacedVolumeMetadata,
    frequency: float,
    *,
    offset: tuple[float, ...] | None = None,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complexfloating]:
    """Build a harmonic potential.

    V(x) = 0.5 * frequency^2 * ||x||^2
    """
    return potential_from_function(
        metadata,
        lambda x: (0.5 * frequency**2 * np.linalg.norm(x, axis=0) ** 2),
        wrapped=True,
        offset=offset,
    )

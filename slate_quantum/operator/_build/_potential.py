from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate import basis
from slate import metadata as _metadata
from slate.basis import (
    CroppedBasis,
    TruncatedBasis,
    Truncation,
    fundamental_transformed_tuple_basis_from_metadata,
)
from slate.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
    StackedMetadata,
)

from slate_quantum._util import outer_product
from slate_quantum.operator._diagonal import Potential

if TYPE_CHECKING:
    from collections.abc import Callable


class RepeatedLengthMetadata(SpacedLengthMetadata):
    def __init__(self, inner: SpacedLengthMetadata, n_repeats: int) -> None:
        self._inner = inner
        self.n_repeats = n_repeats
        super().__init__(
            inner.fundamental_size * n_repeats,
            spacing=LabelSpacing(
                start=inner.spacing.start, delta=n_repeats * inner.spacing.delta
            ),
        )

    @property
    def inner(self) -> SpacedLengthMetadata:
        return self._inner


type RepeatedVolumeMetadata = StackedMetadata[RepeatedLengthMetadata, AxisDirections]


def _get_repeat_basis_metadata(
    metadata: SpacedVolumeMetadata, shape: tuple[int, ...]
) -> RepeatedVolumeMetadata:
    return StackedMetadata(
        tuple(
            RepeatedLengthMetadata(
               d, s
            )
            for (s, d) in zip(shape, metadata.children, strict=True)
        ),
        metadata.extra,
    )


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
        _get_repeat_basis_metadata(potential.basis.outer_recast.metadata(), shape)
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
) -> Potential[M, E, DT]:
    """Get the potential operator."""
    positions = _metadata.volume.fundamental_stacked_x_points(metadata)
    return Potential(basis.from_metadata(metadata), fn(positions))

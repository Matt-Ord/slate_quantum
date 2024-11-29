from __future__ import annotations

import numpy as np
from slate import basis
from slate.basis import (
    CroppedBasis,
    TruncatedBasis,
    Truncation,
    fundamental_transformed_tuple_basis_from_metadata,
)
from slate.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedVolumeMetadata,
    StackedMetadata,
)
from slate.metadata.length import SpacedLengthMetadata

from slate_quantum._util import outer_product
from slate_quantum.operator._diagonal import Potential


def _get_repeat_basis_metadata(
    metadata: SpacedVolumeMetadata, shape: tuple[int, ...]
) -> SpacedVolumeMetadata:
    return StackedMetadata(
        tuple(
            SpacedLengthMetadata(
                s * d.fundamental_size,
                spacing=LabelSpacing(delta=s * d.delta),
            )
            for (s, d) in zip(shape, metadata.children, strict=True)
        ),
        metadata.extra,
    )


def repeat_potential(
    potential: Potential[SpacedLengthMetadata, AxisDirections, np.complex128],
    shape: tuple[int, ...],
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complex128]:
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


def build_cos_potential(
    metadata: SpacedVolumeMetadata,
    height: float,
) -> Potential[SpacedLengthMetadata, AxisDirections, np.complex128]:
    """Build a cosine potential."""
    transformed_basis = fundamental_transformed_tuple_basis_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y),
    )
    n_dim = len(cropped.shape)
    data = outer_product(*(np.array([2, -1, -1]),) * n_dim)
    return Potential(
        cropped,
        0.25**n_dim * height * data * np.sqrt(transformed_basis.size),
    )

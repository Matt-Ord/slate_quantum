from __future__ import annotations

from typing import Any, Self, cast

import numpy as np
from slate.basis import Basis
from slate.basis.recast import RecastBasis
from slate.basis.stacked import fundamental_tuple_basis_from_metadata, tuple_basis
from slate.metadata.stacked import StackedMetadata, VolumeMetadata

from slate_quantum.model.operator._operator import Operator

type PotentialBasis[M: VolumeMetadata, DT: np.generic] = RecastBasis[
    StackedMetadata[M, None], M, DT
]


class Potential[M: VolumeMetadata, DT: np.generic](
    Operator[M, DT, PotentialBasis[M, DT]]
):
    def __init__(
        self: Self,
        basis: Basis[M, Any],
        raw_data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        fundamental = cast(
            Basis[M, Any], fundamental_tuple_basis_from_metadata(basis.metadata)
        )
        super().__init__(
            RecastBasis(
                tuple_basis((fundamental, fundamental.conjugate_basis())),
                fundamental,
                basis,
            ),
            raw_data,
        )

    def with_outer_basis(self, basis: Basis[M, Any]) -> Potential[M, DT]:
        """Get the Potential with the outer recast basis set to basis."""
        return Potential(
            basis, self.basis.outer_recast.__convert_vector_into__(self.raw_data, basis)
        )

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import Array, Basis, BasisMetadata, TupleBasis, TupleMetadata
from slate_core import basis as _basis

from slate_quantum._util._prod import outer_product

if TYPE_CHECKING:
    from slate_core.basis import AsUpcast


type ListMetadata[
    M0: BasisMetadata = BasisMetadata,
    M1: BasisMetadata = BasisMetadata,
] = TupleMetadata[tuple[M0, M1], Any]

type ListBasis[
    M0: BasisMetadata = BasisMetadata,
    M1: BasisMetadata = BasisMetadata,
] = Basis[ListMetadata[M0, M1]]


def with_list_basis[M1: BasisMetadata, B1: Basis, DT: np.dtype[np.generic]](
    array: Array[ListBasis[Any, M1], DT], basis: B1
) -> Array[
    AsUpcast[
        TupleBasis[tuple[B1, Basis[M1]], None],
        TupleMetadata[tuple[BasisMetadata, M1], None],
    ],
    DT,
]:
    """Get the Operator with the operator basis set to basis."""
    rhs_basis = _basis.as_tuple(array.basis).children[1]
    out_basis = TupleBasis((basis, rhs_basis)).upcast()
    return array.with_basis(out_basis)


__all__ = [
    "ListBasis",
    "ListMetadata",
    "outer_product",
    "with_list_basis",
]

from __future__ import annotations

from slate.basis import WrappedBasis
from slate.metadata import BasisMetadata

from slate_quantum.model.state.eigenstate_basis import EigenstateBasis


class BlochDiagonalBasis(WrappedBasis): ...


class BlochEigenstateBasis[M: BasisMetadata](EigenstateBasis[M]): ...

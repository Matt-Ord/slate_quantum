from __future__ import annotations

from typing import Any

from slate.basis import WrappedBasis
from slate.metadata import BasisMetadata

from slate_quantum.model.state.eigenstate_basis import EigenstateBasis


class BlochDiagonalBasis(WrappedBasis[Any, Any, Any]): ...  # noqa: D101


class BlochEigenstateBasis[M: BasisMetadata](EigenstateBasis[M]): ...  # noqa: D101

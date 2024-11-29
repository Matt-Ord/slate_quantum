from __future__ import annotations

from typing import Any

import numpy as np
from slate import FundamentalBasis
from slate.metadata import (
    DeltaMetadata,
    ExplicitLabeledMetadata,
    SpacedLabeledMetadata,
)


class TimeMetadata(DeltaMetadata[float]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(SpacedLabeledMetadata, TimeMetadata):
    """Metadata with the addition of length."""


class MomentumMetadata(DeltaMetadata[float]):
    """Metadata with the addition of momentum."""


class SpacedMomentumMetadata(SpacedLabeledMetadata, MomentumMetadata):
    """Metadata with the addition of momentum."""


class EigenvalueMetadata(ExplicitLabeledMetadata[np.complex128]):
    """Metadata with the addition of eigenvalues."""


def eigenvalue_basis(
    values: np.ndarray[Any, np.dtype[np.complex128]],
) -> FundamentalBasis[EigenvalueMetadata]:
    """Return the eigenvalue basis."""
    return FundamentalBasis(EigenvalueMetadata(values))

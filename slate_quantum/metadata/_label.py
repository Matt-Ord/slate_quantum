from __future__ import annotations

from typing import Any

import numpy as np
from slate_core import FundamentalBasis
from slate_core.metadata import (
    DeltaMetadata,
    ExplicitLabeledMetadata,
    SpacedLabeledMetadata,
)


class TimeMetadata(DeltaMetadata[np.floating]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(SpacedLabeledMetadata, TimeMetadata):
    """Metadata with the addition of length."""


class MomentumMetadata(DeltaMetadata[np.floating]):
    """Metadata with the addition of momentum."""


class SpacedMomentumMetadata(SpacedLabeledMetadata, MomentumMetadata):
    """Metadata with the addition of momentum."""


class EigenvalueMetadata(ExplicitLabeledMetadata[np.complexfloating]):
    """Metadata with the addition of eigenvalues."""


def eigenvalue_basis(
    values: np.ndarray[Any, np.dtype[np.complexfloating]],
) -> FundamentalBasis[EigenvalueMetadata]:
    """Return the eigenvalue basis."""
    return FundamentalBasis(EigenvalueMetadata(values))

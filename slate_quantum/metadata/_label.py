from __future__ import annotations

from typing import Any

import numpy as np
from slate_core import FundamentalBasis
from slate_core.metadata import (
    EvenlySpacedMetadata,
    ExplicitLabeledMetadata,
    SpacedMetadata,
)


class TimeMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(EvenlySpacedMetadata, TimeMetadata):
    """Metadata with the addition of length."""


class MomentumMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """Metadata with the addition of momentum."""


class EvenlySpacedMomentumMetadata(EvenlySpacedMetadata, MomentumMetadata):
    """Metadata with the addition of momentum."""


class EigenvalueMetadata(ExplicitLabeledMetadata[np.dtype[np.complexfloating]]):
    """Metadata with the addition of eigenvalues."""


def eigenvalue_basis(
    values: np.ndarray[Any, np.dtype[np.complexfloating]],
) -> FundamentalBasis[EigenvalueMetadata]:
    """Return the eigenvalue basis."""
    return FundamentalBasis(EigenvalueMetadata(values))

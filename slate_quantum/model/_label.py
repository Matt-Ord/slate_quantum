from __future__ import annotations

import numpy as np
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

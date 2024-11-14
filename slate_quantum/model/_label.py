from __future__ import annotations

from slate.metadata import LabeledMetadata, SpacedLabeledMetadata


class TimeMetadata(LabeledMetadata[float]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(SpacedLabeledMetadata, TimeMetadata):
    """Metadata with the addition of length."""


class MomentumMetadata(LabeledMetadata[float]):
    """Metadata with the addition of momentum."""


class SpacedMomentumMetadata(SpacedLabeledMetadata, MomentumMetadata):
    """Metadata with the addition of momentum."""

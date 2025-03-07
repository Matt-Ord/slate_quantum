from __future__ import annotations

from slate_core import TupleMetadata
from slate_core.metadata import (
    AxisDirections,
    LabelSpacing,
    SpacedLengthMetadata,
    SpacedVolumeMetadata,
)


class RepeatedLengthMetadata(SpacedLengthMetadata):
    def __init__(self, inner: SpacedLengthMetadata, n_repeats: int) -> None:
        self._inner = inner
        self.n_repeats = n_repeats
        super().__init__(
            inner.fundamental_size * n_repeats,
            spacing=LabelSpacing(
                start=inner.spacing.start, delta=n_repeats * inner.spacing.delta
            ),
        )

    @property
    def inner(self) -> SpacedLengthMetadata:
        return self._inner


type RepeatedVolumeMetadata = TupleMetadata[
    tuple[RepeatedLengthMetadata, ...], AxisDirections
]


def repeat_volume_metadata(
    metadata: SpacedVolumeMetadata, shape: tuple[int, ...]
) -> RepeatedVolumeMetadata:
    return TupleMetadata(
        tuple(
            RepeatedLengthMetadata(d, s)
            for (s, d) in zip(shape, metadata.children, strict=True)
        ),
        metadata.extra,
    )

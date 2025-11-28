from slate_core import TupleMetadata
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    EvenlySpacedVolumeMetadata,
)


class RepeatedLengthMetadata(EvenlySpacedLengthMetadata):
    def __init__(self, inner: EvenlySpacedLengthMetadata, n_repeats: int) -> None:
        self._inner = inner
        self.n_repeats = n_repeats
        super().__init__(
            inner.fundamental_size * n_repeats,
            domain=Domain(
                start=inner.domain.start, delta=n_repeats * inner.domain.delta
            ),
            interpolation=inner.interpolation,
        )

    @property
    def inner(self) -> EvenlySpacedLengthMetadata:
        return self._inner


type RepeatedVolumeMetadata = TupleMetadata[
    tuple[RepeatedLengthMetadata, ...], AxisDirections
]


def repeat_volume_metadata(
    metadata: EvenlySpacedVolumeMetadata, shape: tuple[int, ...]
) -> RepeatedVolumeMetadata:
    return TupleMetadata(
        tuple(
            RepeatedLengthMetadata(d, s)
            for (s, d) in zip(shape, metadata.children, strict=True)
        ),
        metadata.extra,
    )


def unit_cell_metadata[E: AxisDirections](
    metadata: TupleMetadata[tuple[RepeatedLengthMetadata, ...], E],
) -> TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], E]:
    """Get the fundamental cell metadata from the repeated volume metadata."""
    return TupleMetadata(
        tuple(c.inner for c in metadata.children),
        metadata.extra,
    )

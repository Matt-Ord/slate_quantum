from slate_core import TupleMetadata
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
)


class RepeatedMetadata[M: EvenlySpacedLengthMetadata = EvenlySpacedLengthMetadata](
    EvenlySpacedLengthMetadata
):
    def __init__(self, inner: M, n_repeats: int) -> None:
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
    def inner(self) -> M:
        return self._inner


type RepeatedVolumeMetadata[
    M: EvenlySpacedLengthMetadata = EvenlySpacedLengthMetadata,
    E: AxisDirections = AxisDirections,
] = TupleMetadata[tuple[RepeatedMetadata[M], ...], E]


def repeat_volume_metadata[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    shape: tuple[int, ...],
) -> RepeatedVolumeMetadata[M, E]:
    return TupleMetadata(
        tuple(
            RepeatedMetadata(d, s)
            for (s, d) in zip(shape, metadata.children, strict=True)
        ),
        metadata.extra,
    )


def unit_cell_metadata[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: RepeatedVolumeMetadata[M, E],
) -> TupleMetadata[tuple[M, ...], E]:
    """Get the fundamental cell metadata from the repeated volume metadata."""
    return TupleMetadata(tuple(c.inner for c in metadata.children), metadata.extra)

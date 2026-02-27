from typing import Any, Literal

import numpy as np
from slate_core import FundamentalBasis
from slate_core.metadata import (
    Domain,
    EvenlySpacedMetadata,
    ExplicitLabeledMetadata,
    SpacedMetadata,
)


class TimeMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """Metadata with the addition of length."""


class SpacedTimeMetadata(EvenlySpacedMetadata, TimeMetadata):
    """Metadata with the addition of length."""

    def __init__(
        self,
        fundamental_size: int,
        *,
        domain: Domain,
        # By default, time is not periodic, so we use the DST interpolation.
        interpolation: Literal["Fourier", "DST"] = "DST",
    ) -> None:
        super().__init__(fundamental_size, domain=domain, interpolation=interpolation)


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

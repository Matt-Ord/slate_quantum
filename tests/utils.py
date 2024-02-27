from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np
from scipy.stats import special_ortho_group

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
)

if TYPE_CHECKING:
    _L0Inv = TypeVar("_L0Inv", bound=int)

rng = np.random.default_rng()


def get_random_explicit_basis(
    nd: _L0Inv,
    fundamental_n: int | None = None,
    n: int | None = None,
) -> ExplicitBasis[int, int, _L0Inv]:
    fundamental_n = (
        rng.integers(2 if n is None else n, 5)  # type: ignore bad libary types
        if fundamental_n is None
        else fundamental_n
    )
    n = rng.integers(1, fundamental_n) if n is None else n  # type: ignore bad libary types
    vectors = special_ortho_group.rvs(fundamental_n)[:n]
    delta_x = np.zeros(nd)
    delta_x[0] = 1
    return ExplicitBasis(delta_x, vectors)
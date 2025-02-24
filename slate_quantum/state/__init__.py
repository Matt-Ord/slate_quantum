"""Representation of a quantum state."""

from __future__ import annotations

import slate_quantum.state._build as build
from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._build import coherent as build_coherent
from slate_quantum.state._state import (
    State,
    StateList,
    all_inner_product,
    get_all_occupations,
    get_average_occupations,
    get_occupations,
    inner_product,
    normalization,
    normalize,
    normalize_all,
)

__all__ = [
    "EigenstateBasis",
    "State",
    "StateList",
    "all_inner_product",
    "build",
    "build_coherent",
    "get_all_occupations",
    "get_average_occupations",
    "get_occupations",
    "inner_product",
    "normalization",
    "normalize",
    "normalize_all",
]

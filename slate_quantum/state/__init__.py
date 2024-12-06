"""Representation of a quantum state."""

from __future__ import annotations

from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._build import build_coherent_state
from slate_quantum.state._state import (
    State,
    StateList,
    get_all_occupations,
    get_average_occupations,
    get_occupations,
    inner_product,
    normalization,
)

__all__ = [
    "EigenstateBasis",
    "State",
    "StateList",
    "build_coherent_state",
    "get_all_occupations",
    "get_average_occupations",
    "get_occupations",
    "inner_product",
    "normalization",
]

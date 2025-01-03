"""Representation of a quantum state."""

from __future__ import annotations

from slate_quantum.state._basis import EigenstateBasis
from slate_quantum.state._build import build_coherent_state
from slate_quantum.state._state import (
    State,
    StateList,
    all_inner_product,
    get_all_occupations,
    get_average_occupations,
    get_occupations,
    inner_product,
    normalization,
    normalize_states,
)

__all__ = [
    "EigenstateBasis",
    "State",
    "StateList",
    "all_inner_product",
    "build_coherent_state",
    "get_all_occupations",
    "get_average_occupations",
    "get_occupations",
    "inner_product",
    "normalization",
    "normalize_states",
]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate import Basis, SimpleMetadata, StackedMetadata
from slate.metadata import AxisDirections, Metadata2D

from slate_quantum.bloch.build import BlochFractionMetadata
from slate_quantum.metadata._repeat import RepeatedLengthMetadata
from slate_quantum.state._state import StateList

if TYPE_CHECKING:
    from slate.basis import BasisStateMetadata
    from slate.explicit_basis import ExplicitBasis


def get_localized_basis[B: Basis[Any, Any]](
    metadata: BasisStateMetadata[B],
) -> ExplicitBasis[Any, Any]:
    """Get the Wannier basis from a list of eigenstates."""
    return states


def get_wannier_basis[B: Basis[Any, Any]](
    metadata: BasisStateMetadata[B],
) -> ExplicitBasis[Any, Any]:
    """Get the Wannier basis from a list of eigenstates."""
    return states


bloch_states = StateList[
    Metadata2D[SimpleMetadata, BlochFractionMetadata, None],
    StackedMetadata[RepeatedLengthMetadata, AxisDirections],
]


def get_bloch_wannier_basis[B: Basis[Any, Any]](
    metadata: BasisStateMetadata[B],
) -> ExplicitBasis[Any, Any, BasisStateMetadata[B]]:
    """Get the Wannier basis from a list of eigenstates."""
    return states

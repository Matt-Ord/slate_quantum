from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slate_quantum.operator._operator import Operator


def build_diagonal_bloch_hamiltonian() -> Operator[Any, Any, Any]:
    """Build the diagonalized Bloch Hamiltonian."""
    msg = "Not yet implemented"
    raise NotImplementedError(msg)

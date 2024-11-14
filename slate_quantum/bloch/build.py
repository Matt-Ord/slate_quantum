from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slate_quantum.model.operator._operator import Operator


def build_diagonal_bloch_hamiltonian() -> Operator:
    """Build the diagonalized Bloch Hamiltonian."""
    msg = "Not yet implemented"
    raise NotImplementedError(msg)

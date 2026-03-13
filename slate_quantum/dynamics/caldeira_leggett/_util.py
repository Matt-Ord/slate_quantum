import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse  # type: ignore[unknown]
from slate_core import EvenlySpacedLengthMetadata
from slate_core.metadata import AxisDirections, TupleMetadata

from slate_quantum.operator import operator_basis

try:
    import qutip  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from qutip import Qobj  # type: ignore lib
    from slate_core import Basis

    from slate_quantum.operator import Operator, OperatorBasis
    from slate_quantum.state._state import State


def operator_as_qobj[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    operator: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
) -> Qobj:  # type: ignore lib
    """Convert an operator to a qutip Qobj.

    Raises
    ------
    ImportError
        If the qutip package is not installed.
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)
    return qutip.Qobj(
        (operator)
        .with_basis(operator_basis(basis))
        .raw_data.reshape((basis.size, basis.size)),
    )


def _truncate_diagonal_to_qutip(
    diag_values: np.ndarray[tuple[int], np.dtype[np.complexfloating]],
    offset: int,
    shape: tuple[int, int],
) -> np.ndarray[tuple[int], np.dtype[np.complexfloating]]:
    """Slice a padded Scipy diagonal to the exact length QuTiP expects."""
    n_rows, n_cols = shape
    col_start = max(0, offset)
    col_end = min(n_cols, n_rows + offset)

    length = col_end - col_start
    if length <= 0:
        return np.array([], dtype=diag_values.dtype)

    # 2. Scipy stores the value for A[j-offset, j] at diag_values[j]
    # So we simply slice the buffer from the first valid column to the last.
    return diag_values[col_start:col_end]


def _full_to_qutip_diagonal(
    data: np.ndarray[tuple[int, int], np.dtype[np.complexfloating]],
    *,
    tolerance: float = 1e-10,
) -> Qobj:
    """Convert a full Qobj to a diagonal (DIA) Qobj by zeroing out small elements.

    Raises
    ------
    ImportError
        If the qutip package is not installed.
    """
    # 1. Convert to Scipy DIA format (this groups data by offsets)
    # .data contains the diagonal values, .offsets contains the i-j index
    dia = scipy.sparse.dia_matrix(data)

    clean_diags = list[np.ndarray[tuple[int], np.dtype[np.complexfloating]]]()
    clean_offsets = list[int]()

    for i, offset in enumerate(dia.offsets):  # type: ignore[unknown]
        diag_values = _truncate_diagonal_to_qutip(
            dia.data[i],  # type: ignore[unknown]
            offset.item(),
            data.shape,
        )

        # 2. Apply tolerance: if the max value in diagonal < tol, skip it
        if np.max(np.abs(diag_values)) > tolerance:
            # Clean individual jittery elements within an accepted diagonal
            diag_values[np.abs(diag_values) < tolerance] = 0

            clean_diags.append(diag_values)
            clean_offsets.append(offset.item())
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)
    return qutip.qdiags(clean_diags, clean_offsets, dims=(data.shape[0], data.shape[1]))  # type: ignore[unknown] # cspell: disable-line


def operator_as_diagonal_qobj[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    operator: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
) -> Qobj:  # type: ignore lib
    """Convert an operator to a qutip Qobj."""
    raw_data = (
        (operator)
        .with_basis(operator_basis(basis))
        .raw_data.reshape((basis.size, basis.size))
    )
    return _full_to_qutip_diagonal(raw_data)


def state_as_qobj[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[Basis[TupleMetadata[tuple[M, ...], E]]],
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
) -> Qobj:  # type: ignore lib
    """Convert a state to a qutip Qobj.

    Raises
    ------
    ImportError
        If the qutip package is not installed.
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    state_raw = state.with_basis(basis).raw_data
    if np.abs(np.linalg.norm(state_raw) - 1) > 1e-3:  # noqa: PLR2004
        msg = (
            "Warning: Initial state is not normalized when converted to the simulation basis. "
            "This is usually because the initial state has significant occupation outside the simulation basis. "
            f"This may lead to numerical instabilities. Norm: {np.linalg.norm(state_raw)}"
        )
        warnings.warn(msg, stacklevel=3, category=UserWarning)
        state_raw /= np.linalg.norm(state_raw)
    return qutip.Qobj(state_raw)

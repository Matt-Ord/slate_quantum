from typing import TYPE_CHECKING

from slate_core import EvenlySpacedLengthMetadata
from slate_core.metadata import AxisDirections, TupleMetadata

from slate_quantum.operator import operator_basis

try:
    import qutip  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    import numpy as np
    from qutip import Qobj  # type: ignore lib
    from slate_core import Basis

    from slate_quantum.operator import Operator, OperatorBasis


def operator_as_qobj[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    operator: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
) -> Qobj:  # type: ignore lib
    """Convert an operator to a qutip Qobj."""  # noqa: DOC501
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)
    return qutip.Qobj(
        (operator)
        .with_basis(operator_basis(basis))
        .raw_data.reshape((basis.size, basis.size)),
    )

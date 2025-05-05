from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import hbar  # type: ignore unknown
from slate_core import Array, BasisMetadata, SimpleMetadata, basis, linalg

from slate_quantum.noise.build import truncate_noise_operator_list
from slate_quantum.noise.diagonalize._eigenvalue import (
    get_periodic_noise_operators_diagonal_eigenvalue,
)
from slate_quantum.operator import OperatorList

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate_quantum.noise._kernel import (
        DiagonalNoiseKernel,
        NoiseOperatorList,
    )


def sample_noise_from_operators[M: BasisMetadata](
    operators: NoiseOperatorList[M], *, n_samples: int
) -> OperatorList[SimpleMetadata, M, np.complexfloating]:
    """Generate noise from a set of noise operators."""
    n_operators = operators.basis[0].size

    rng = np.random.default_rng()
    factors = (
        rng.standard_normal((n_samples, n_operators))
        + 1j * rng.standard_normal((n_samples, n_operators))
    ) / np.sqrt(2)

    scaled_factors = Array.from_array(
        cast(
            "np.ndarray[tuple[int, int], Any]",
            np.einsum(  # type: ignore lib
                "ij,j->ij",
                factors,
                np.lib.scimath.sqrt(operators.basis[0].metadata().values * hbar),
            ),
        )
    )
    data = linalg.einsum("(i j),(j k)->(i k)", scaled_factors, operators)
    return OperatorList(data.basis, data.raw_data)


def sample_noise_from_diagonal_kernel[M: BasisMetadata](
    kernel: DiagonalNoiseKernel[M, np.complexfloating],
    *,
    n_samples: int,
    truncation: Iterable[int] | None,
) -> OperatorList[SimpleMetadata, M, np.complexfloating]:
    """Generate noise for a diagonal kernel."""
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)
    operators = operators.with_list_basis(basis.as_tuple_basis(operators.basis)[0])

    truncation = range(operators.basis[0].size) if truncation is None else truncation
    truncated = truncate_noise_operator_list(operators, truncation)
    truncated = truncated.with_list_basis(basis.as_tuple_basis(truncated.basis)[0])

    return sample_noise_from_operators(truncated, n_samples=n_samples)

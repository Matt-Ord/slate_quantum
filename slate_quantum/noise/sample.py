from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.constants import hbar  # type: ignore unknown
from slate_core import Array, BasisMetadata, SimpleMetadata, basis, linalg

from slate_quantum.noise.build import truncate_noise_operator_list
from slate_quantum.noise.diagonalize._eigenvalue import (
    get_periodic_noise_operators_diagonal_eigenvalue,
)
from slate_quantum.operator._operator import (
    OperatorList,
    OperatorListWithMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from slate_quantum.metadata._label import EigenvalueMetadata
    from slate_quantum.noise._kernel import (
        DiagonalNoiseKernelWithMetadata,
        NoiseOperatorList,
    )
    from slate_quantum.operator._operator import OperatorListWithMetadata


def sample_noise_from_operators[M: BasisMetadata](
    operators: NoiseOperatorList[EigenvalueMetadata, M], *, n_samples: int
) -> OperatorListWithMetadata[SimpleMetadata, M, np.dtype[np.complexfloating]]:
    """Generate noise from a set of noise operators."""
    n_operators = operators.basis.inner.children[0].size

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
                np.lib.scimath.sqrt(
                    operators.basis.metadata().children[0].values * hbar
                ),
            ),
        )
    )
    data = linalg.einsum("(i j),(j k)->(i k)", scaled_factors, operators)
    return OperatorList(data.basis, data.raw_data)


def sample_noise_from_diagonal_kernel[M: BasisMetadata](
    kernel: DiagonalNoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
    *,
    n_samples: int,
    truncation: Iterable[int] | None,
) -> OperatorListWithMetadata[SimpleMetadata, M, np.dtype[np.complexfloating]]:
    """Generate noise for a diagonal kernel."""
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)
    operators = operators.with_list_basis(basis.as_tuple(operators.basis).children[0])

    truncation = (
        range(operators.basis.inner.children[0].size)
        if truncation is None
        else truncation
    )
    truncated = truncate_noise_operator_list(operators, truncation)
    truncated = truncated.with_list_basis(basis.as_tuple(truncated.basis).children[0])

    return sample_noise_from_operators(truncated, n_samples=n_samples)

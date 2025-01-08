from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.constants import hbar  # type:ignore lib


def sample_noise_from_diagonal_operators_split(
    operators: DiagonalNoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> DiagonalOperatorList[TupleBasis[_B0, FundamentalBasis[BasisMetadata]], _B1, _B2]:
    """Generate noise from a set of diagonal noise operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    DiagonalOperatorList[FundamentalBasis[BasisMetadata], _B1, _B2]
    """
    n_operators = operators["basis"][0].size

    rng = np.random.default_rng()
    factors = (
        rng.standard_normal((n_samples, n_operators))
        + 1j * rng.standard_normal((n_samples, n_operators))
    ) / np.sqrt(2)

    data = np.einsum(  # type: ignore lib
        "ij,j,jk->jik",
        factors,
        np.lib.scimath.sqrt(operators["eigenvalue"] * hbar),
        operators["data"].reshape(n_operators, -1),
    )
    return {
        "basis": TupleBasis(
            VariadicTupleBasis(
                (operators["basis"][0], FundamentalBasis(n_samples), None)
            ),
            operators["basis"][1],
        ),
        "data": data.ravel(),
    }


def diagonal_operator_list_from_diagonal_split(
    split: DiagonalOperatorList[TupleBasis[_B0, _B3], _B1, _B2],
) -> DiagonalOperatorList[_B3, _B1, _B2]:
    """
    Sum over the 'spit' axis.

    Parameters
    ----------
    split : DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2]

    Returns
    -------
    DiagonalOperatorList[_B3, _B1, _B2]
    """
    data = np.sum(split["data"].reshape(*split["basis"][0].shape, -1), axis=0)
    return {
        "basis": VariadicTupleBasis((split["basis"][0][1], split["basis"][1]), None),
        "data": data.ravel(),
    }


def get_diagonal_split_noise_components(
    split: DiagonalOperatorList[TupleBasis[_B0, _B3], _B1, _B2],
) -> list[DiagonalOperatorList[_B3, _B1, _B2]]:
    """Get the components of the noise.

    Parameters
    ----------
    split : DiagonalOperatorList[TupleBasis[_B3, _B0], _B1, _B2]

    Returns
    -------
    list[DiagonalOperatorList[_B3, _B1, _B2]]
    """
    data = split["data"].reshape(*split["basis"][0].shape, -1)
    return [
        {
            "basis": VariadicTupleBasis(
                (split["basis"][0][1], split["basis"][1]), None
            ),
            "data": d.ravel(),
        }
        for d in data
    ]


def sample_noise_from_diagonal_operators(
    operators: DiagonalNoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> DiagonalOperatorList[FundamentalBasis[BasisMetadata], _B1, _B2]:
    """Generate noise from a set of diagonal noise operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    DiagonalOperatorList[FundamentalBasis[BasisMetadata], _B1, _B2]
    """
    split = sample_noise_from_diagonal_operators_split(operators, n_samples=n_samples)
    return diagonal_operator_list_from_diagonal_split(split)


def sample_noise_from_operators(
    operators: NoiseOperatorList[_B0, _B1, _B2], *, n_samples: int
) -> OperatorList[FundamentalBasis[BasisMetadata], _B1, _B2]:
    """Generate noise from a set of noise operators.

    Parameters
    ----------
    operators : NoiseOperatorList[_B0, _B1, _B2]
    n_samples : int

    Returns
    -------
    OperatorList[FundamentalBasis[BasisMetadata], _B1, _B2]
    """
    n_operators = operators["basis"][0].size

    rng = np.random.default_rng()
    factors = (
        rng.standard_normal((n_samples, n_operators))
        + 1j * rng.standard_normal((n_samples, n_operators))
    ) / np.sqrt(2)

    data = np.einsum(  # type: ignore lib
        "ij,j,jk->ik",
        factors,
        np.lib.scimath.sqrt(operators["eigenvalue"] * hbar),
        operators["data"].reshape(n_operators, -1),
    )
    return {
        "basis": VariadicTupleBasis(
            (FundamentalBasis(n_samples), None), operators["basis"][1]
        ),
        "data": data.ravel(),
    }


def sample_noise_from_diagonal_kernel(
    kernel: DiagonalNoiseKernel[_B0, _B1, _B0, _B1],
    *,
    n_samples: int,
    truncation: Iterable[int] | None,
) -> OperatorList[FundamentalBasis[BasisMetadata], _B0, _B1]:
    """Generate noise for a diagonal kernel."""
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)
    truncation = range(operators["basis"][0].size) if truncation is None else truncation
    truncated = truncate_diagonal_noise_operator_list(operators, truncation)
    return sample_noise_from_diagonal_operators(truncated, n_samples=n_samples)

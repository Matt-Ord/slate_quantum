from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core import Basis, Ctype, TupleBasis, TupleMetadata, basis, linalg
from slate_core.basis import CoordinateBasis, DiagonalBasis
from slate_core.metadata import AxisDirections
from slate_core.util import slice_ignoring_axes

from slate_quantum import operator
from slate_quantum.metadata import EigenvalueMetadata
from slate_quantum.noise._kernel import (
    DiagonalKernelBasisWithMetadata,
    DiagonalKernelWithMetadata,
    IsotropicKernelBasisWithMetadata,
    IsotropicKernelWithMetadata,
    NoiseKernel,
    NoiseKernelWithMetadata,
    diagonal_kernel_basis,
    isotropic_kernel_basis,
    with_isotropic_basis,
)
from slate_quantum.noise.diagonalize import (
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
)
from slate_quantum.operator import (
    Operator,
    SuperOperatorMetadata,
    get_commutator_operator_list,
)
from slate_quantum.util._prod import outer_product

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from slate_core.metadata import (
        BasisMetadata,
    )
    from slate_core.metadata.length import LengthMetadata, SpacedLengthMetadata

    from slate_quantum.noise._kernel import AxisKernel
    from slate_quantum.operator import (
        OperatorList,
    )
    from slate_quantum.operator._operator import (
        OperatorBasis,
        OperatorBuilder,
        OperatorListBasis,
        OperatorMetadata,
        SuperOperatorBasis,
    )


def diagonal_kernel[M: BasisMetadata, CT: Ctype[Never], DT: np.dtype[np.generic]](
    outer_basis: OperatorBasis[OperatorMetadata[M], CT],
    data: np.ndarray[Any, DT],
) -> OperatorBuilder[DiagonalKernelBasisWithMetadata[OperatorMetadata[M], CT], DT]:
    kernel_basis = diagonal_kernel_basis(outer_basis)
    return Operator.build(kernel_basis, data)


def diagonal_kernel_from_operators[M_: BasisMetadata, DT_: np.dtype[np.generic]](
    operators: OperatorList[
        OperatorListBasis[EigenvalueMetadata, OperatorMetadata[OperatorMetadata[M_]]],
        DT_,
    ],
) -> DiagonalKernelWithMetadata[M_, Ctype[Never], DT_]:
    """Build a diagonal kernel from operators."""
    as_tuple = basis.as_tuple(operators.basis)
    final_basis = TupleBasis(
        (
            basis.as_index(as_tuple.children[0]),
            DiagonalBasis(basis.as_tuple(as_tuple.children[1])).upcast(),
        )
    )
    converted = operators.with_basis(final_basis.upcast()).assert_ok()

    operators_data = converted.raw_data.reshape(
        converted.basis.inner.children[0].size, -1
    )
    data = cast(
        "Any",
        np.einsum(  # type:ignore  unknown
            "a,ai,aj->ij",
            converted.basis.inner.children[0]
            .metadata()
            .values[converted.basis.inner.children[0].points],
            np.conj(operators_data),
            operators_data,  # type:ignore DT not numeric
        ),
    )
    return diagonal_kernel(converted.basis.inner.children[1], data).assert_ok()


def isotropic_kernel[M: BasisMetadata, CT: Ctype[Never], DT: np.dtype[np.generic]](
    outer_basis: Basis[M, CT],
    data: np.ndarray[Any, DT],
) -> OperatorBuilder[IsotropicKernelBasisWithMetadata[M, CT], DT]:
    kernel_basis = isotropic_kernel_basis(outer_basis)
    return Operator.build(kernel_basis, data)


def isotropic_kernel_from_function[M: LengthMetadata](
    metadata: M,
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.inner.shape)[0])

    return isotropic_kernel(
        displacements.basis.inner.children[0], correlation
    ).assert_ok()


def isotropic_kernel_from_function_stacked[
    M: SpacedLengthMetadata,
    E: AxisDirections,
](
    metadata: TupleMetadata[tuple[M, ...], E],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> IsotropicKernelWithMetadata[
    TupleMetadata[tuple[M, ...], E], Ctype[Never], np.dtype[np.complexfloating]
]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    displacements = operator.build.total_x_displacement_operator(metadata)
    correlation = fn(displacements.raw_data.reshape(displacements.basis.inner.shape)[0])

    return isotropic_kernel(
        displacements.basis.inner.children[0], correlation
    ).assert_ok()


def isotropic_kernel_from_axis[
    M: BasisMetadata,
    CT: Ctype[Never],
    DT: np.dtype[np.complexfloating],
](kernels: AxisKernel[M, CT, DT]) -> IsotropicKernelWithMetadata[M, CT, DT]:
    """Convert an axis kernel to an isotropic kernel."""
    full_basis = tuple(kernel_i.basis for kernel_i in kernels)
    full_data = tuple(kernel_i.raw_data.ravel() for kernel_i in kernels)
    # TODO: compare to old..
    return isotropic_kernel(
        TupleBasis(full_basis, None), outer_product(*full_data).ravel()
    ).assert_ok()


def axis_kernel_from_isotropic[
    M: BasisMetadata,
    CT: Ctype[Never],
    DT: np.dtype[np.complexfloating],
](
    kernel: IsotropicKernelWithMetadata[M, CT, DT],
) -> AxisKernel[M, CT, DT]:
    """Convert an isotropic kernel to an axis kernel."""
    outer_as_tuple = basis.as_tuple(kernel.basis.inner.outer_recast.outer_recast)
    converted = with_isotropic_basis(kernel, outer_as_tuple)
    n_axis = len(outer_as_tuple.shape)

    data_stacked = converted.raw_data.reshape(outer_as_tuple.shape)
    slice_without_idx = tuple(0 for _ in range(n_axis - 1))

    prefactor = converted.raw_data[0] ** ((1 - n_axis) / n_axis)
    return tuple(
        isotropic_kernel(
            axis_basis,
            prefactor * data_stacked[slice_ignoring_axes(slice_without_idx, (i,))],
        ).assert_ok()
        for i, axis_basis in enumerate(outer_as_tuple.children)
    )


def axis_kernel_from_function_stacked[M: SpacedLengthMetadata](
    metadata: TupleMetadata[tuple[M, ...], Any],
    fn: Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ],
) -> AxisKernel[M, Ctype[Never], np.dtype[np.complexfloating]]:
    """
    Get an Isotropic Kernel with a correlation beta(x-x').

    Parameters
    ----------
    basis : StackedBasisWithVolumeLike
    fn : Callable[
        [np.ndarray[Any, np.dtype[np.float64]]],
        np.ndarray[Any, np.dtype[np.complexfloating]],
    ]
        beta(x-x'), the correlation as a function of displacement

    Returns
    -------
    IsotropicNoiseKernel[
        TupleBasisWithLengthLike[*tuple[FundamentalPositionBasis, ...]],
    ]
    """
    return tuple(
        isotropic_kernel_from_function(child, fn) for child in metadata.children
    )


def kernel_from_operators[M: BasisMetadata, DT: np.dtype[np.generic]](
    operators: OperatorList[
        OperatorListBasis[EigenvalueMetadata, OperatorMetadata[M]], DT
    ],
) -> NoiseKernel[SuperOperatorBasis[M], DT]:
    as_tuple = basis.as_tuple(operators.basis)
    final_basis = TupleBasis(
        (
            basis.as_index(as_tuple.children[0]),
            basis.as_tuple(as_tuple.children[1]).upcast(),
        )
    )
    converted = operators.with_basis(final_basis.upcast()).assert_ok()

    operators_data = converted.raw_data.reshape(
        converted.basis.inner.children[0].size,
        *converted.basis.inner.children[1].inner.shape,
    )

    data = np.einsum(  # type:ignore  unknown
        "a,aji,akl->ij kl",
        converted.basis.inner.children[0].metadata().values[converted.basis.points],
        np.conj(operators_data),
        operators_data.astype(np.complex128),
    )
    return Operator.build(
        TupleBasis(
            (converted.basis.inner.children[1], converted.basis.inner.children[1])
        ).upcast(),
        data,
    ).ok()


def gaussian_correlation_fn(
    a: float, sigma: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    r"""Get a correlation function for a gaussian noise kernel.

    A gaussian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \exp{\frac{(-(x-x')^2 }{(2  \lambda^2))}}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        return (a**2 * np.exp(-(displacements**2) / (2 * sigma**2))).astype(
            np.complex128,
        )

    return fn


def lorentzian_correlation_fn(
    a: float, lambda_: float
) -> Callable[
    [np.ndarray[Any, np.dtype[np.float64]]],
    np.ndarray[Any, np.dtype[np.complexfloating]],
]:
    r"""Get a correlation function for a lorentzian noise kernel.

    A lorentzian noise kernel is isotropic, and separable into individual
    axis kernels. The kernel is given by

    .. math::
        \beta(x, x') = a^2 \frac{\lambda^2}{(x-x')^2 + \lambda^2}
    """

    def fn(
        displacements: np.ndarray[Any, np.dtype[np.float64]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
        return (a**2 * lambda_**2 / (displacements**2 + lambda_**2)).astype(
            np.complex128
        )

    return fn


def temperature_corrected_operators[M0: BasisMetadata, M1: BasisMetadata](
    hamiltonian: Operator[OperatorBasis[M1], np.dtype[np.complexfloating]],
    operators: OperatorList[
        OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
    ],
    temperature: float,
    eta: float,
) -> OperatorList[
    OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
]:
    """Get the temperature corrected operators.

    Note this returns the operators multiplied by hbar, to avoid numerical issues.
    """
    commutator = get_commutator_operator_list(hamiltonian, operators)
    thermal_energy = Boltzmann * temperature
    correction = commutator * (-1 * np.sqrt(eta / (8 * thermal_energy * hbar**2)))
    operators *= np.sqrt(2 * eta * thermal_energy / hbar**2)
    return correction + operators


def hamiltonian_shift[M1: BasisMetadata](
    hamiltonian: Operator[OperatorBasis[M1], np.dtype[np.complexfloating]],
    operators: OperatorList[
        OperatorListBasis[BasisMetadata, OperatorMetadata[M1]],
        np.dtype[np.complexfloating],
    ],
    eta: float,
) -> Operator[OperatorBasis[M1], np.dtype[np.complexfloating]]:
    """Get the temperature corrected Hamiltonian shift."""
    shift_product = linalg.einsum(
        "(i (j k)),(i (k' l))->(k' l)", operator.dagger_each(operators), operators
    )
    shift_product = Operator.build(
        shift_product.basis, shift_product.raw_data
    ).assert_ok()
    commutator = operator.commute(hamiltonian, shift_product)
    pre_factor = 1j * eta / (4 * hbar)
    return (commutator * pre_factor).as_type(np.complex64)


def truncate_noise_operator_list[M0: EigenvalueMetadata, M1: BasisMetadata](
    operators: OperatorList[
        OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
    ],
    truncation: Iterable[int],
) -> OperatorList[
    OperatorListBasis[M0, OperatorMetadata[M1]], np.dtype[np.complexfloating]
]:
    """
    Get a truncated list of diagonal operators.

    Parameters
    ----------
    operators : DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], B_0, B_1]
    truncation : Iterable[int]

    Returns
    -------
    DiagonalNoiseOperatorList[FundamentalBasis[BasisMetadata], B_0, B_1]
    """
    converted = operators.with_basis(
        basis.as_tuple(operators.basis).upcast()
    ).assert_ok()
    converted_list = converted.with_list_basis(
        basis.as_index(converted.basis.inner.children[0])
    ).assert_ok()
    eigenvalues = (
        converted_list.basis.inner.metadata()
        .children[0]
        .values[converted_list.basis.inner.children[0].points]
    )
    args = np.argsort(np.abs(eigenvalues))[::-1][np.array(list(truncation))]
    list_basis = CoordinateBasis(args, basis.as_tuple(operators.basis).children[0])
    return operators.with_list_basis(list_basis).assert_ok()


def truncate_noise_kernel[M: SuperOperatorMetadata](
    kernel: NoiseKernelWithMetadata[M, np.dtype[np.complexfloating]],
    truncation: Iterable[int],
) -> NoiseKernelWithMetadata[M, np.dtype[np.complexfloating]]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : NoiseKernelWithMetadata[B_0, B_1, B_0, B_1]
    truncation : Iterable[int]

    Returns
    -------
    NoiseKernelWithMetadata[B_0, B_1, B_0, B_1]
    """
    operators = get_periodic_noise_operators_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    return kernel_from_operators(truncated)


def truncate_diagonal_noise_kernel[
    M: BasisMetadata,
](
    kernel: DiagonalKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]],
    truncation: Iterable[int],
) -> DiagonalKernelWithMetadata[M, Ctype[Never], np.dtype[np.complexfloating]]:
    """
    Given a noise kernel, retain only the first n noise operators.

    Parameters
    ----------
    kernel : DiagonalKernelWithMetadata[B_0, B_1, B_0, B_1]
    truncation : Iterable[int]

    Returns
    -------
    DiagonalKernelWithMetadata[B_0, B_1, B_0, B_1]
    """
    operators = get_periodic_noise_operators_diagonal_eigenvalue(kernel)

    truncated = truncate_noise_operator_list(operators, truncation)
    diagonal_kernel_from_operators(truncated)
    return diagonal_kernel_from_operators(truncated)

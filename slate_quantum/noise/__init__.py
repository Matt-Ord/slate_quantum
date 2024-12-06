"""Code for generating noise in quantum simulations."""

from __future__ import annotations

from slate_quantum.noise._build import (
    build_axis_kernel_from_function_stacked,
    build_isotropic_kernel_from_function,
    build_isotropic_kernel_from_function_stacked,
    caldeira_leggett_correllation_fn,
    gaussian_correllation_fn,
    get_temperature_corrected_operators,
    truncate_diagonal_noise_kernel,
    truncate_noise_kernel,
    truncate_noise_operator_list,
)
from slate_quantum.noise._caldeira_leggett import (
    build_periodic_caldeira_leggett_axis_operators,
    build_periodic_caldeira_leggett_operators,
)
from slate_quantum.noise._kernel import (
    AxisKernel,
    DiagonalNoiseKernel,
    DiagonalNoiseOperatorList,
    IsotropicNoiseKernel,
    NoiseKernel,
    NoiseOperatorList,
    as_axis_kernel_from_isotropic,
    as_isotropic_kernel_from_axis,
    get_diagonal_noise_operators_from_axis,
)
from slate_quantum.noise.diagonalize import (
    get_linear_noise_operators_explicit_taylor_expansion,
    get_periodic_noise_operators_diagonal_eigenvalue,
    get_periodic_noise_operators_eigenvalue,
    get_periodic_noise_operators_explicit_taylor_expansion,
    get_periodic_noise_operators_isotropic_fft,
    get_periodic_noise_operators_isotropic_stacked_fft,
    get_periodic_noise_operators_real_isotropic_stacked_taylor_expansion,
    get_periodic_noise_operators_real_isotropic_taylor_expansion,
)

__all__ = [
    "AxisKernel",
    "DiagonalNoiseKernel",
    "DiagonalNoiseOperatorList",
    "IsotropicNoiseKernel",
    "NoiseKernel",
    "NoiseOperatorList",
    "as_axis_kernel_from_isotropic",
    "as_isotropic_kernel_from_axis",
    "build_axis_kernel_from_function_stacked",
    "build_isotropic_kernel_from_function",
    "build_isotropic_kernel_from_function_stacked",
    "build_periodic_caldeira_leggett_axis_operators",
    "build_periodic_caldeira_leggett_operators",
    "caldeira_leggett_correllation_fn",
    "gaussian_correllation_fn",
    "get_diagonal_noise_operators_from_axis",
    "get_linear_noise_operators_explicit_taylor_expansion",
    "get_periodic_noise_operators_diagonal_eigenvalue",
    "get_periodic_noise_operators_eigenvalue",
    "get_periodic_noise_operators_explicit_taylor_expansion",
    "get_periodic_noise_operators_isotropic_fft",
    "get_periodic_noise_operators_isotropic_stacked_fft",
    "get_periodic_noise_operators_real_isotropic_stacked_taylor_expansion",
    "get_periodic_noise_operators_real_isotropic_taylor_expansion",
    "get_temperature_corrected_operators",
    "truncate_diagonal_noise_kernel",
    "truncate_noise_kernel",
    "truncate_noise_operator_list",
]

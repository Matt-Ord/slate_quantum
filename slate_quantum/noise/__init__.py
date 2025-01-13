"""Code for generating noise in quantum simulations."""

from __future__ import annotations

from slate_quantum.noise import build
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
from slate_quantum.noise.build import (
    axis_kernel_from_function_stacked as build_axis_kernel_from_function_stacked,
)
from slate_quantum.noise.build import (
    caldeira_leggett_correlation_fn,
    gaussian_correlation_fn,
    lorentzian_correlation_fn,
    truncate_diagonal_noise_kernel,
    truncate_noise_kernel,
    truncate_noise_operator_list,
)
from slate_quantum.noise.build import (
    hamiltonain_shift as build_hamiltonian_shift,
)
from slate_quantum.noise.build import (
    isotropic_kernel_from_function as build_isotropic_kernel_from_function,
)
from slate_quantum.noise.build import (
    isotropic_kernel_from_function_stacked as build_isotropic_kernel_from_function_stacked,
)
from slate_quantum.noise.build import (
    periodic_caldeira_leggett_axis_operators as build_periodic_caldeira_leggett_axis_operators,
)
from slate_quantum.noise.build import (
    periodic_caldeira_leggett_operators as build_periodic_caldeira_leggett_operators,
)
from slate_quantum.noise.build import (
    real_periodic_caldeira_leggett_operators as build_real_periodic_caldeira_leggett_operators,
)
from slate_quantum.noise.build import (
    temperature_corrected_operators as build_temperature_corrected_operators,
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
    "DiagonalNoiseKernel",
    "DiagonalNoiseOperatorList",
    "IsotropicNoiseKernel",
    "NoiseKernel",
    "NoiseOperatorList",
    "as_axis_kernel_from_isotropic",
    "as_isotropic_kernel_from_axis",
    "build",
    "build_axis_kernel_from_function_stacked",
    "build_hamiltonian_shift",
    "build_isotropic_kernel_from_function",
    "build_isotropic_kernel_from_function_stacked",
    "build_periodic_caldeira_leggett_axis_operators",
    "build_periodic_caldeira_leggett_operators",
    "build_real_periodic_caldeira_leggett_operators",
    "build_temperature_corrected_operators",
    "caldeira_leggett_correlation_fn",
    "gaussian_correlation_fn",
    "get_diagonal_noise_operators_from_axis",
    "get_linear_noise_operators_explicit_taylor_expansion",
    "get_periodic_noise_operators_diagonal_eigenvalue",
    "get_periodic_noise_operators_eigenvalue",
    "get_periodic_noise_operators_explicit_taylor_expansion",
    "get_periodic_noise_operators_isotropic_fft",
    "get_periodic_noise_operators_isotropic_stacked_fft",
    "get_periodic_noise_operators_real_isotropic_stacked_taylor_expansion",
    "get_periodic_noise_operators_real_isotropic_taylor_expansion",
    "lorentzian_correlation_fn",
    "truncate_diagonal_noise_kernel",
    "truncate_noise_kernel",
    "truncate_noise_operator_list",
]

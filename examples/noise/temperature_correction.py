from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from slate import array, plot
from slate.metadata import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum.noise import (
    build_isotropic_kernel_from_function_stacked,
    gaussian_correlation_fn,
    get_periodic_noise_operators_isotropic_stacked_fft,
    get_temperature_corrected_operators,
)
from slate_quantum.operator import (
    build_kinetic_energy_operator,
)

# sphinx.ext.autosummary TODO: this
if __name__ == "__main__":
    # For a system interacting with a thermal environment, the noise
    # is not independent at each point in time. Instead, the noise
    # has some weak correlation in time.
    # This correlation is however very weak, and can be neglected
    # in most cases as long as we also apply a suitable temperature correction.

    # We first generate a set of noise operators, for a gaussian kernel
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )
    lambda_ = np.pi / 2
    correlation = gaussian_correlation_fn(1, lambda_)
    kernel = build_isotropic_kernel_from_function_stacked(metadata, correlation)
    operators = get_periodic_noise_operators_isotropic_stacked_fft(kernel)

    # For an ohmic environment, the correlation is given by
    # O -> \sqrt{2 \eta K T / \hbar^2} O - \sqrt{\eta / 8 K T} [H, O]
    # To see the effect of the temperature correction, we can
    # plot the operators before and after the correction.
    hamiltonian = build_kinetic_energy_operator(metadata, hbar**2)
    corrected_operators = get_temperature_corrected_operators(
        hamiltonian, operators, temperature=20 * hbar / Boltzmann, eta=1
    )

    idx = -2
    fig, ax, _ = plot.basis_against_array_2d_x(array.flatten(operators[idx, :]))
    ax.set_label("Original noise operator")
    fig.show()

    fig, ax, _ = plot.basis_against_array_2d_k(array.flatten(operators[idx, :]))
    ax.set_label("Original noise operator")
    fig.show()

    fig, ax, _ = plot.basis_against_array_2d_x(
        array.flatten(corrected_operators[idx, :])
    )
    ax.set_label("Temperature corrected noise operator")
    fig.show()

    fig, ax, _ = plot.basis_against_array_2d_k(
        array.flatten(corrected_operators[idx, :]), measure="abs"
    )
    ax.set_label("Temperature corrected noise operator")
    fig.show()
    input()

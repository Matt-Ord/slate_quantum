from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore unknown
from slate import array
from slate.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import plot_data_2d_k, plot_data_2d_x

from slate_quantum.noise import (
    build_isotropic_kernel_from_function_stacked,
    gaussian_correllation_fn,
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
    # has some weak correllation in time.
    # This correllation is however very weak, and can be neglected
    # in most cases as long as we also apply a suitable temperature correction.

    # We first generate a set of noise operators, for a gaussian kernel
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )
    lambda_ = np.pi / 2
    correlation = gaussian_correllation_fn(1, lambda_)
    kernel = build_isotropic_kernel_from_function_stacked(metadata, correlation)
    operators = get_periodic_noise_operators_isotropic_stacked_fft(kernel)

    # For an ohmic environment, the correllation is given by
    # TODO: actual formula here! # noqa: FIX002
    # To see the effect of the temperature correction, we can
    # plot the operators before and after the correction.
    #
    hamiltonian = build_kinetic_energy_operator(metadata, hbar**2)
    corrected_operators = get_temperature_corrected_operators(
        hamiltonian, operators, temperature=0.1 / Boltzmann
    )

    idx = -2
    fig, ax, _ = plot_data_2d_x(array.as_flatten_basis(operators[idx]))
    ax.set_label("Original noise operator")
    fig.show()

    fig, ax, _ = plot_data_2d_k(array.as_flatten_basis(operators[idx]))
    ax.set_label("Original noise operator")
    fig.show()

    fig, ax, _ = plot_data_2d_x(array.as_flatten_basis(corrected_operators[idx]))
    ax.set_label("Temperature corrected noise operator")
    fig.show()

    fig, ax, _ = plot_data_2d_k(array.as_flatten_basis(corrected_operators[idx]))
    ax.set_label("Temperature corrected noise operator")
    fig.show()
    input()

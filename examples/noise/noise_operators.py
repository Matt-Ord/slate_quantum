from __future__ import annotations

from statistics import correlation

import numpy as np
from slate_core import array, plot
from slate_core.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate_core.plot import get_figure

from slate_quantum.noise import (
    build,
    gaussian_correlation_fn,
    get_periodic_noise_operators_isotropic_stacked_fft,
    truncate_noise_operator_list,
)
from slate_quantum.noise._kernel import isotropic_kernel_from_operators

if __name__ == "__main__":
    # Starting with a simple 1D system, we can define the noise
    # kernel. This is a measure of the correlaltion of
    # the environment in the system.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    correlation = gaussian_correlation_fn(1, sigma=np.pi / 2)
    kernel = build.isotropic_kernel_from_function_stacked(metadata, correlation)

    # We can plot the kernel to see how the correlation decays with distance.
    # The correlation falls by $e^-1$ at a distance of \pi / 2$.
    full_data = array.as_outer_basis(array.as_outer_basis(kernel))
    fig, ax, _ = plot.array_against_axes_1d(full_data)
    ax.set_title(r"Isotropic Gaussian Kernel with $\sigma = \pi / 2$")
    fig.show()

    # To represent the noise we need some choice of noise operators.
    # There are several ways to generate these. We can for instance
    # find the eigenvectors of the kernel. Since the kernel is a
    # SuperOperator, the eigenvectors are in fact themselves operators.
    # For an isotropic kernel, the eigenvectors are all diagonal
    # operators, as below.
    operators = get_periodic_noise_operators_isotropic_stacked_fft(kernel)
    # Since the operators are diagonal, we can plot them along the diagonal
    # these correspond to the three lowest frequency fourier components
    # $e^{ikx}$.
    fig, ax = get_figure()
    for i in range(3):
        plot.array_against_axes_1d(
            array.as_outer_basis(operators[i, :]), ax=ax, measure="real"
        )
    fig.show()
    ax.set_title("The real part of the first three noise operators")

    # If we take only the first few operators, we can see that they
    # approximate the kernel.
    truncated = truncate_noise_operator_list(operators, range(3))
    restored = isotropic_kernel_from_operators(truncated)
    restored_data = array.as_outer_basis(array.as_outer_basis(restored))

    # The restored kernel is an approximation of the original kernel
    # which is good in the region which is smooth.
    # This is because the noise operators we have taken are the
    # lowest frequency components of the kernel.
    fig, ax = get_figure()
    _, _, line = plot.array_against_axes_1d(full_data, ax=ax)
    line.set_label("Original kernel")
    line.set_linestyle("--")
    _, _, line = plot.array_against_axes_1d(restored_data, ax=ax)
    line.set_label("Restored kernel")
    ax.set_title("Restored noise kernel, from the first 3 operators")
    ax.legend()
    fig.show()

    plot.wait_for_close()

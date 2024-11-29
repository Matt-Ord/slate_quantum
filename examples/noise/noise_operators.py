from __future__ import annotations

from statistics import correlation

import numpy as np
from slate import array
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import get_figure, plot_data_1d_x

from slate_quantum.noise import (
    IsotropicNoiseKernel,
    build_isotropic_kernel_from_function_stacked,
    gaussian_correllation_fn,
    get_periodic_noise_operators_real_isotropic_stacked_fft,
    truncate_noise_operator_list,
)

if __name__ == "__main__":
    # Starting with a simple 1D system, we can define the noise
    # kernel. This is a measure of the correlaltion of
    # the environment in the system.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )
    lambda_ = np.pi / 2
    correlation = gaussian_correllation_fn(1, lambda_)
    kernel = build_isotropic_kernel_from_function_stacked(metadata, correlation)

    # We can plot the kernel to see how the correllation decays with distance.
    # The correlation falls by $e^-1$ at a distance of $\lambda = \pi / 2$.
    full_data = array.as_outer_basis(array.as_outer_basis(kernel))
    fig, ax, _ = plot_data_1d_x(full_data)
    ax.set_title(r"Isotropic Gaussian Kernel with $\lambda = \pi / 2$")
    fig.show()

    # To represent the noise we need some choice of noise operators.
    # There are several ways to generate these. We can for instance
    # find the eigenvectors of the kernel. Since the kernel is a
    # SuperOperator, the eigenvectors are in fact themselves operators.
    # For an isotropic kernel, the eigenvectors are all diagonal
    # operators, as below.
    # TODO: we need a better approach in the case axis kernels fail for us...  # noqa: FIX002
    operators = get_periodic_noise_operators_real_isotropic_stacked_fft(kernel)
    # Since the operators are diagonal, we can plot them along the diagonal
    # these correspond to the three lowest frequency fourier components
    # $e^{ikx}$.
    fig, ax = get_figure()
    for i in range(3):
        plot_data_1d_x(array.as_outer_basis(operators[i]), ax=ax, measure="real")
    fig.show()
    ax.set_title("The real part of the first three noise operators")

    # If we take only the first few operators, we can see that they
    # approximate the kernel.
    truncated = truncate_noise_operator_list(operators, range(3))
    restored = IsotropicNoiseKernel.from_operators(truncated)
    restored_data = array.as_outer_basis(array.as_outer_basis(restored))

    # The restored kernel is an approximation of the original kernel
    # which is good in the region which is smooth.
    # This is because the noise operators we have taken are the
    # lowest frequency components of the kernel.
    fig, ax = get_figure()
    _, _, line = plot_data_1d_x(full_data, ax=ax)
    line.set_label("Original kernel")
    line.set_linestyle("--")
    _, _, line = plot_data_1d_x(restored_data, ax=ax)
    line.set_label("Restored kernel")
    ax.set_title("Restored noise kernel, from the first 3 operators")
    ax.legend()
    fig.show()

    input()

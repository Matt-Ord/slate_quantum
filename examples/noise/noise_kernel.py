from __future__ import annotations

from statistics import correlation

import numpy as np
from slate_core import array, plot
from slate_core.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate_core.plot import get_figure

from slate_quantum.noise import (
    build,
    caldeira_leggett_correlation_fn,
    gaussian_correlation_fn,
)

if __name__ == "__main__":
    # Starting with a simple 1D system, we can define the noise
    # kernel. This is a measure of the correlaltion of
    # the environment in the system.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    _lambda = 2 / np.pi

    correlation = gaussian_correlation_fn(1, np.sqrt(2) / _lambda)
    kernel = build.isotropic_kernel_from_function_stacked(metadata, correlation)

    # We can plot the kernel to see how the correlation decays with distance.
    # The correlation falls by $e^-1$ at a distance of $\pi / 2$.
    full_data = array.as_outer_array(array.as_outer_array(kernel))
    fig, ax, _ = plot.array_against_axes_1d(full_data)
    ax.set_title(r"Isotropic Gaussian Kernel with $\sigma = \pi / \sqrt{2}$")
    fig.show()
    # We can also plot the 'full' isotropic kernel, to see the correlation
    # between all points in the system.
    diagonal_data = array.flatten(array.as_outer_array(kernel))
    fig, ax, _ = plot.array_against_axes_2d_x(diagonal_data)
    ax.set_title(r"Isotropic Gaussian Kernel with $\sigma = \pi / \sqrt{2}$")
    fig.show()

    # If we want to represent the noise using the caldeira-leggett model
    # the kernel we are simulating will be slightly different to the 'true' kernel.
    correlation = caldeira_leggett_correlation_fn(1, _lambda)
    kernel_cl = build.isotropic_kernel_from_function_stacked(metadata, correlation)
    cl_data = array.as_outer_array(array.as_outer_array(kernel_cl))

    fig, ax = get_figure()
    _, _, line = plot.array_against_axes_1d(full_data, ax=ax)
    line.set_label("Gaussian Kernel")
    _, _, line = plot.array_against_axes_1d(cl_data, ax=ax)
    line.set_label("Caldeira Leggett Kernel")
    ax.legend()
    ax.set_title("Comaprison of CL approximation to Gaussian Kernel")
    fig.show()
    # We find that the CL kernel is a good approximation to the Gaussian kernel
    # for small displacements, but this approximation falls apart as we
    # move further from the origin.
    plot.wait_for_close()

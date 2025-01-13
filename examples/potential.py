from __future__ import annotations

import numpy as np
from slate import array, plot
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import operator

if __name__ == "__main__":
    # Metadata for a 1D volume with 100 points and a width of 14
    metadata = spaced_volume_metadata_from_stacked_delta_x((np.array([14]),), (100,))

    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 1D, height 10 with 100 points
    # Here .diagonal() is used to get the potential as an array
    # of points along the diagonal
    fig, ax, line = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    fig.show()

    potential = operator.repeat_potential(potential, (3,))
    # A repeated cos potential, with 3 repeats
    fig, ax, line = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    fig.show()

    # Metadata for a 2D volume with 100 points and a width of 14, 10
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([20, 0]), np.array([0, 10])), (100, 100)
    )
    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 2D, height 10
    fig, ax, line = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    fig.show()
    fig, ax, line = plot.basis_against_array_2d_x(array.as_outer_array(potential))
    fig.show()

    potential = operator.repeat_potential(potential, (3, 4))
    # A repeated cos potential, with (3, 4) repeats
    fig, ax, line = plot.basis_against_array_2d_x(array.as_outer_array(potential))
    fig.show()

    # For a 2D volume with fcc symmetry, the cos potential is not suitable
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (
            np.array([1, 0]),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]),
        ),
        (6, 6),
    )
    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 2D, height 10
    fig, ax, line = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    fig.show()
    fig, ax, line = plot.basis_against_array_2d_x(array.as_outer_array(potential))
    fig.show()

    potential = operator.build.fcc_potential(metadata, 1)
    fig, ax, line = plot.basis_against_array_1d_x(array.as_outer_array(potential))
    fig.show()
    fig, ax, line = plot.basis_against_array_2d_x(array.as_outer_array(potential))
    fig.show()

    plot.wait_for_close()

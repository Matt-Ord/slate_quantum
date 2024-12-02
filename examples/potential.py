from __future__ import annotations

import numpy as np
from slate import array
from slate.metadata.volume import spaced_volume_metadata_from_stacked_delta_x
from slate.plot import plot_data_1d_x, plot_data_2d_x

from slate_quantum import operator

if __name__ == "__main__":
    # Metadata for a 1D volume with 100 points and a width of 14
    metadata = spaced_volume_metadata_from_stacked_delta_x((np.array([14]),), (100,))

    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 1D, height 10 with 100 points
    # Here .diagonal() is used to get the potential as an array
    # of points along the diagonal
    fig, ax, line = plot_data_1d_x(array.as_outer_basis(potential))
    fig.show()

    potential = operator.repeat_potential(potential, (3,))
    # A repeated cos potential, with 3 repeats
    fig, ax, line = plot_data_1d_x(array.as_outer_basis(potential))
    fig.show()

    # Metadata for a 2D volume with 100 points and a width of 14, 10
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([20, 0]), np.array([0, 10])), (100, 100)
    )
    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 2D, height 10
    fig, ax, line = plot_data_1d_x(array.as_outer_basis(potential))
    fig.show()
    fig, ax, line = plot_data_2d_x(array.as_outer_basis(potential))
    fig.show()

    potential = operator.repeat_potential(potential, (3, 4))
    # A repeated cos potential, with (3, 4) repeats
    fig, ax, line = plot_data_2d_x(array.as_outer_basis(potential))
    fig.show()

    input()

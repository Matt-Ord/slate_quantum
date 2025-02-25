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
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("Cos Potential")
    ax.set_ylabel("V(x)")
    fig.show()

    potential = operator.repeat_potential(potential, (3,))
    # A repeated cos potential, with 3 repeats
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("Repeated Cos Potential")
    ax.set_ylabel("V(x)")
    fig.show()

    # A harmonic potential
    potential = operator.build.harmonic_potential(metadata, 10)
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("Harmonic Potential")
    ax.set_ylabel("V(x)")
    fig.show()

    # Metadata for a 2D volume with 100 points and a width of 14, 10
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([20, 0]), np.array([0, 10])), (100, 100)
    )
    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 2D, height 10
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("2D Cos Potential at x1=0")
    ax.set_ylabel("V(x)")
    fig.show()
    fig, ax, line = plot.array_against_axes_2d_x(array.as_outer_array(potential))
    ax.set_title("Cos Potential in 2D")
    fig.show()

    potential = operator.repeat_potential(potential, (3, 4))
    # A repeated cos potential, with (3, 4) repeats
    fig, ax, line = plot.array_against_axes_2d_x(array.as_outer_array(potential))
    ax.set_title("Repeated 2D Cos Potential")
    fig.show()

    # For a 2D volume with fcc symmetry, the cos potential is not suitable
    # We can see this by plotting - the top site PES in non-spherical
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (
            np.array([1, 0]),
            np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)]),
        ),
        (100, 100),
    )
    potential = operator.build.cos_potential(metadata, 10)
    # A single cos potential in 2D, height 10
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("2D Cos Potential with FCC symmetry at x1=0")
    ax.set_ylabel("V(x)")
    fig.show()
    fig, ax, line = plot.array_against_axes_2d_x(array.as_outer_array(potential))
    ax.set_title("Cos Potential in 2D with FCC symmetry")
    fig.show()

    potential = operator.build.fcc_potential(metadata, 1)
    fig, ax, line = plot.array_against_axes_1d(array.as_outer_array(potential))
    ax.set_title("FCC Potential at x1=0")
    ax.set_ylabel("V(x)")
    fig.show()
    fig, ax, line = plot.array_against_axes_2d_x(array.as_outer_array(potential))
    ax.set_title("FCC Potential in 2D")
    fig.show()

    plot.wait_for_close()

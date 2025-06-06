from __future__ import annotations

import numpy as np
from scipy.constants import hbar  # type: ignore stubs
from slate_core import FundamentalBasis, plot
from slate_core.metadata import (
    LabelSpacing,
    spaced_volume_metadata_from_stacked_delta_x,
)
from slate_core.plot import animate_data_over_list_1d_x

from slate_quantum import operator
from slate_quantum.dynamics import (
    solve_schrodinger_equation,
    solve_schrodinger_equation_decomposition,
)
from slate_quantum.metadata import SpacedTimeMetadata

if __name__ == "__main__":
    # Metadata for a 1D volume with 60 points in the x direction
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (60,)
    )

    potential = operator.build.cos_potential(metadata, 1)
    hamiltonian = operator.build.kinetic_hamiltonian(potential, hbar**2)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # Let's simulate the evolution of the system starting
    # in an eigenstate of the Hamiltonian
    initial_state = eigenstates[1, :]
    times = FundamentalBasis(
        SpacedTimeMetadata(60, spacing=LabelSpacing(delta=8 * np.pi * hbar))
    )
    evolution = solve_schrodinger_equation_decomposition(
        initial_state, times, hamiltonian
    )
    # The state changes in time, but only by a phase difference
    fig, ax, _anim0 = animate_data_over_list_1d_x(evolution, measure="real")
    ax.set_title("Real part of the state")
    fig.show()
    # If we plot the absolute value of the state, we see no change
    fig, ax, _anim1 = animate_data_over_list_1d_x(evolution, measure="abs")
    ax.set_title("Abs state, which does not change with time")
    fig.show()

    # We can also use Qutip to solve the same problem without using eigenvalue decomposition.
    evolution = solve_schrodinger_equation(initial_state, times, hamiltonian)
    # The state changes in time, but only by a phase difference
    fig, ax, _anim2 = animate_data_over_list_1d_x(evolution, measure="real")
    ax.set_title("Real part of the state as simulated by Qutip")
    fig.show()
    # If we plot the absolute value of the state, we see no change
    fig, ax, _anim3 = animate_data_over_list_1d_x(evolution, measure="abs")
    ax.set_title("Abs state, which does not change with time as simulated by Qutip")
    fig.show()

    plot.wait_for_close()

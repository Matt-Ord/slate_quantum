import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore stubs
from slate_core import FundamentalBasis, plot
from slate_core.metadata import (
    Domain,
    spaced_volume_metadata_from_stacked_delta_x,
)
from slate_core.plot import animate_data_over_list_1d_x

from slate_quantum import operator
from slate_quantum.dynamics import (
    select_realization,
    simulate_caldeira_leggett_realizations,
)
from slate_quantum.dynamics.caldeira_leggett import CaldeiraLeggettCondition
from slate_quantum.metadata import SpacedTimeMetadata

if __name__ == "__main__":
    # Metadata for a 1D volume with 60 points in the x direction
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (61,)
    )

    potential = operator.build.harmonic_potential(metadata, 5, offset=(np.pi,))
    mass = hbar**2
    hamiltonian = operator.build.kinetic_hamiltonian(potential, mass)
    eigenstates = operator.get_eigenstates_hermitian(hamiltonian)

    # Let's simulate the evolution of the system starting
    # in an eigenstate of the Hamiltonian
    condition = CaldeiraLeggettCondition(
        mass=mass,
        friction=0.1 * hbar,
        temperature=1 / Boltzmann,
        potential=potential,
        initial_state=eigenstates[0, :],
    )
    times = FundamentalBasis(
        SpacedTimeMetadata(60, domain=Domain(delta=1 * np.pi * hbar))
    )
    realizations = simulate_caldeira_leggett_realizations(condition, times)
    evolution = select_realization(realizations, 0)
    # The state changes in time, but only by a phase difference
    fig, ax, _anim0 = animate_data_over_list_1d_x(evolution, measure="real")
    ax.set_title("Real part of the state")
    fig.show()
    # If we plot the absolute value of the state, we see no change
    fig, ax, _anim1 = animate_data_over_list_1d_x(evolution, measure="abs")
    ax.set_title("Abs state, which would not change with time if there is no friction")
    fig.show()

    plot.wait_for_close()

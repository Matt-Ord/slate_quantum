import numpy as np
from scipy.constants import Boltzmann  # type: ignore stubs
from slate_core import FundamentalBasis
from slate_core.metadata import (
    Domain,
    spaced_volume_metadata_from_stacked_delta_x,
)
from slate_core.plot import array_against_basis

from slate_quantum import operator, state
from slate_quantum.dynamics import (
    LangevinParameters,
    caldeira_leggett,
)
from slate_quantum.metadata import SpacedTimeMetadata

metadata = spaced_volume_metadata_from_stacked_delta_x((np.array([2 * np.pi]),), (21,))

potential_single = operator.build.cos_potential(metadata, 2)
potential = operator.build.repeat_potential(potential_single, (3,))
potential_metadata = potential.basis.inner.outer_recast.metadata()


mass = 1
hamiltonian = operator.build.kinetic_hamiltonian(potential, mass, hbar=1)
initial_state = state.build_coherent(potential_metadata, (0,), (0,), sigma_0=(0.5,))


parameters = LangevinParameters(mass=mass, lambda_=0, temperature=8 / Boltzmann, hbar=1)
times = FundamentalBasis(
    SpacedTimeMetadata(60, domain=Domain(delta=2 * np.pi), interpolation="DST")
)


energies, norms = caldeira_leggett.solve_periodic_energies(
    initial_state,
    times,
    parameters,
    potential,
    method="Rouchon",
    target_delta=1e-3,
)
norm = norms[0, :]
fig, ax, line = array_against_basis(norm)
fig.savefig("temp.png")


fig, ax, line = array_against_basis(energies[0, :], measure="abs")
fig.savefig("temp1.png")

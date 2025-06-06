from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import hbar  # type: ignore lib
from slate_core import FundamentalBasis, TupleBasis, basis
from slate_core.basis import CroppedBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)
from slate_core.util import timed

from slate_quantum import operator, state
from slate_quantum.metadata import TimeMetadata
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    operator_basis,
)
from slate_quantum.state._state import StateList

try:
    import qutip  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from slate_core import Basis, Ctype, TupleMetadata

    from slate_quantum.dynamics._realization import (
        RealizationList,
        RealizationListBasis,
    )
    from slate_quantum.operator._diagonal import Potential
    from slate_quantum.state._state import StateWithMetadata


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettCondition[M: EvenlySpacedLengthMetadata, E: AxisDirections]:
    """Specifies the condition for a Caldeira-Leggett simulation."""

    mass: float
    friction: float
    temperature: float
    potential: Potential[M, E]
    initial_state: StateWithMetadata[TupleMetadata[tuple[M, ...], E]]


DT_RATIO = 10000


def _approximate_dt[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    hamiltonian: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    initial_state: StateWithMetadata[TupleMetadata[tuple[M, ...], E]],
) -> float:
    """Approximate the time step based on the metadata of the times basis."""
    norm = state.normalization(operator.apply(hamiltonian, initial_state))
    d_psi = norm / hbar
    # We want d_psi * dt to be << 1, ie dt << 1 / d_psi
    return 1 / (d_psi * DT_RATIO)


def _get_simulation_basis[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> Basis[TupleMetadata[tuple[M, ...], E]]:
    """Get the simulation basis for the Caldeira-Leggett model."""
    # For now we do nothing fancy here - in the future we could investigate automatically
    # ideally, we would like to use the split operator method, but this isn't supported by qutip yet
    # For stability, we use a hardwall basis, such that the state is forced to be zero at the boundaries.
    trigonometric = basis.trigonometric_transformed_from_metadata(
        metadata, fn="sin", ty="type 2"
    )
    return basis.with_modified_children(
        trigonometric,
        # Remove the high fequency components to avoid numerical instabilities
        lambda _, b: CroppedBasis((b.size) // 2, b).upcast(),
    ).upcast()


@timed
def simulate_caldeira_leggett_realizations[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
    MT: TimeMetadata,
](
    condition: CaldeiraLeggettCondition[M, E],
    times: Basis[MT, Ctype[np.complexfloating]],
    *,
    n_realizations: int = 1,
) -> RealizationList[RealizationListBasis[MT, TupleMetadata[tuple[M, ...], E]]]:
    """Simulate the Caldeira-Leggett model using the Stochastic Schrödinger Equation (SSE).

    To avoid numerical instabilities, the simulation uses a hard-wall boundary condition,
    centered at the origin. This is achieved by generating the initial hamiltonian and
    environment operators in a repeat of the simulation basis, which is then truncated.

    Internally, this uses the `ssesolve` function from the `QuTiP` library.

    Raises
    ------
    ImportError
        If qutip is not installed
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    times = basis.as_fundamental(times)

    simulation_basis = _get_simulation_basis(
        condition.potential.basis.metadata().children[0]
    )
    hamiltonian = operator.build.kinetic_hamiltonian(
        condition.potential, condition.mass
    )
    shift_operator = operator.build.caldeira_leggett_shift(
        simulation_basis.metadata(), friction=condition.friction
    )
    environment_operator = operator.build.caldeira_leggett_collapse(
        simulation_basis.metadata(),
        friction=condition.friction,
        temperature=condition.temperature,
        mass=condition.mass,
    )

    dt = _approximate_dt(hamiltonian, condition.initial_state)
    print(f"Will require {times.metadata().delta / dt:.2e} time steps")  # noqa: T201

    # Simulates an SSE defined as in eqn 4.76 in https://doi.org/10.1017/CBO9780511813948
    # Using the qutip.ssesolve method
    # d|psi(t)> = - i H |psi(t)> dt
    #             - (S dagger S / 2 - <S + S dagger> S / 2 + <S + S dagger>^2 / 8) |psi(t)> dt
    #             + (S - (<S + S dagger> / 2)) |psi(t)> dW
    result = qutip.ssesolve(  # type: ignore lib
        H=qutip.Qobj(
            (hamiltonian + shift_operator)
            .with_basis(operator_basis(simulation_basis))
            .raw_data.reshape((simulation_basis.size, simulation_basis.size))
            / hbar,
        ),
        psi0=qutip.Qobj(condition.initial_state.with_basis(simulation_basis).raw_data),
        tlist=times.metadata().values,
        ntraj=n_realizations,
        sc_ops=[
            qutip.Qobj(
                environment_operator.with_basis(
                    TupleBasis((simulation_basis, simulation_basis))
                ).raw_data.reshape((simulation_basis.size, simulation_basis.size))
                / hbar,
            )
        ],
        options={
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "method": "platen",
            "dt": dt,
        },
    )

    return StateList(
        TupleBasis(
            (
                TupleBasis(
                    (FundamentalBasis.from_size(n_realizations), times)
                ).upcast(),
                simulation_basis,
            )
        ).upcast(),
        np.array(
            np.asarray(
                [
                    [state.full().reshape(-1) for state in realization]  # type: ignore lib
                    for realization in result.states  # type: ignore lib
                ]
            ),
            dtype=np.complex128,
        ),
    )

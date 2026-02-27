from typing import TYPE_CHECKING, Unpack

import numpy as np
import slate_core
from slate_core import FundamentalBasis, TupleBasis, TupleMetadata, basis
from slate_core.basis import CroppedBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)
from slate_core.util import timed

from slate_quantum import operator
from slate_quantum.dynamics.caldeira_leggett._util import operator_as_diagonal_qobj
from slate_quantum.dynamics.langevin._util import (
    QutipSSEConfig,
    as_qutip_name,
    rescale_simulation_metadata,
    rescale_times,
)
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
    from slate_core import Basis, Ctype

    from slate_quantum.dynamics._realization import (
        RealizationList,
        RealizationListBasis,
    )
    from slate_quantum.dynamics.langevin._util import LangevinParameters
    from slate_quantum.state._state import StateWithMetadata


DEFAULT_TARGET_DELTA = 1e-4


def _get_periodic_basis[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> Basis[TupleMetadata[tuple[M, ...], E]]:
    """Get the simulation basis for the Caldeira-Leggett model."""
    transformed = basis.transformed_from_metadata(metadata)
    return basis.with_modified_children(
        transformed,
        # Remove the high frequency components to avoid numerical instabilities
        lambda _, b: CroppedBasis((b.size) // 2, b).upcast(),
    ).upcast()


def _build_environment_operators[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
    params: LangevinParameters,
) -> list[
    Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ]
]:
    """Build the environment operators for the Caldeira-Leggett model with periodic boundaries."""
    out_basis = operator_basis(basis).upcast()

    cos_x = (
        operator.build.cos_potential(basis.metadata(), 2.0, offset=-1.0)
        .with_basis(out_basis)
        .raw_data.reshape(basis.size, basis.size)
    )

    sin_x = (
        operator.build.sin_potential(basis.metadata(), 2.0, offset=-1.0)
        .with_basis(out_basis)
        .raw_data.reshape(basis.size, basis.size)
    )

    correction_prefactor = 1j / (params.kbt_div_hbar * params.mass)
    p_operator = (
        operator.build.p(basis.metadata(), axis=0, hbar=params.hbar)
        .with_basis(out_basis)
        .raw_data.reshape(basis.size, basis.size)
    )
    cos_x_correction = -correction_prefactor * (p_operator @ sin_x + sin_x @ p_operator)
    sin_x_correction = correction_prefactor * (p_operator @ cos_x + cos_x @ p_operator)

    dk = slate_core.metadata.volume.fundamental_stacked_dk(basis.metadata())[0][0]
    operator_prefactor = 1 / (np.sqrt(2) * dk)
    friction_const = 2 * params.kbt_div_hbar * params.mass
    return [
        Operator(
            out_basis,
            np.sqrt(friction_const) * (operator_prefactor * cos_x + cos_x_correction),
        ),
        Operator(
            out_basis,
            np.sqrt(friction_const) * (operator_prefactor * sin_x + sin_x_correction),
        ),
    ]


@timed
def solve[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
    MT: TimeMetadata,
](
    initial_state: StateWithMetadata[TupleMetadata[tuple[M, ...], E]],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    **kwargs: Unpack[QutipSSEConfig],
) -> RealizationList[RealizationListBasis[MT, TupleMetadata[tuple[M, ...], E]]]:
    """Simulate the Caldeira-Leggett model using the Periodic Approach.

    To avoid numerical instabilities, the simulation uses a set of periodic
    environment operators (sin(x) and cos(x)).
    Thestrength of the environmental interaction is carefully
    chosen to match the behavior of the "standard" approach, which
    must use hard wall or adsorbing boundary conditions to avoid instabilities.

    This function also pre-calculates a suitable set of natural units
    for the simulation based on the Hamiltonian and initial state provided in the `condition`.

    Internally, this uses the `ssesolve` function from the `QuTiP` library.

    Raises
    ------
    ImportError
        If qutip is not installed
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    times_basis = basis.as_index(times)
    normalized_params = parameters.normalized_parameters

    normalized_times = rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_parameter=parameters,
        out_parameter=normalized_params,
    )

    normalized_metadata = rescale_simulation_metadata(
        potential.basis.metadata().children[0],
        in_parameters=parameters,
        out_parameters=normalized_params,
    )
    normalized_basis = _get_periodic_basis(normalized_metadata)
    unnormalized_basis = _get_periodic_basis(potential.basis.metadata().children[0])

    hamiltonian = operator.build.kinetic_hamiltonian(
        # Convert V(x) to natural units. since hardwall basis is
        # a `mul` basis, we can just scale the operator data directly
        Operator(
            operator_basis(normalized_basis).upcast(),
            potential.with_basis(operator_basis(unnormalized_basis).upcast()).raw_data
            * (normalized_params.kbt / parameters.kbt),
        ),
        normalized_params.mass,
        hbar=normalized_params.hbar,
    )

    environment_operators = _build_environment_operators(
        normalized_basis, normalized_params
    )

    print(  # noqa: T201
        f"Starting solve, approximate {normalized_times[-1] / kwargs.get('target_delta', DEFAULT_TARGET_DELTA):.2g} time steps"
    )

    # Simulates an SSE defined as in eqn 4.76 in https://doi.org/10.1017/CBO9780511813948
    # Using the qutip.ssesolve method
    # d|psi(t)> = - i H |psi(t)> dt
    #             - (S dagger S / 2 - <S + S dagger> S / 2 + <S + S dagger>^2 / 8) |psi(t)> dt
    #             + (S - (<S + S dagger> / 2)) |psi(t)> dW
    n_trajectories = kwargs.get("n_trajectories", 1)
    result = qutip.ssesolve(  # type: ignore lib
        H=operator_as_diagonal_qobj(hamiltonian, normalized_basis)
        / normalized_params.hbar,
        psi0=qutip.Qobj(initial_state.with_basis(unnormalized_basis).raw_data),
        tlist=normalized_times,
        ntraj=n_trajectories,
        sc_ops=[
            operator_as_diagonal_qobj(op, normalized_basis)
            for op in environment_operators
        ],
        options={
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "method": as_qutip_name(kwargs.get("method", "Euler")),
            "normalize_output": False,
            "dt": kwargs.get("target_delta", DEFAULT_TARGET_DELTA),
        },
        heterodyne=True,
    )

    return StateList(
        TupleBasis(
            (
                TupleBasis(
                    (
                        FundamentalBasis.from_size(n_trajectories),
                        basis.as_fundamental(times),
                    )
                ).upcast(),
                unnormalized_basis,
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

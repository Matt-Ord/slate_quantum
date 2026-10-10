from typing import TYPE_CHECKING, Any, Unpack

import numpy as np
import slate_core
from slate_core import (
    Array,
    FundamentalBasis,
    TupleBasis,
    TupleMetadata,
    basis,
    linalg,
)
from slate_core.basis import CroppedBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)
from slate_core.util import timed

from slate_quantum import operator
from slate_quantum.dynamics.caldeira_leggett._util import (
    operator_as_diagonal_qobj,
    state_as_qobj,
    windowed_wiener,
)
from slate_quantum.dynamics.langevin._util import (
    QutipSSEConfig,
    as_qutip_name,
    rescale_alpha,
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

    correction_prefactor = (
        1j * params.hbar / (8 * np.sqrt(2) * params.kbt_div_hbar * params.mass)
    )
    k_operator = (
        operator.build.k(basis.metadata(), axis=0)
        .with_basis(out_basis)
        .raw_data.reshape(basis.size, basis.size)
    )
    cos_x_correction = -correction_prefactor * (k_operator @ sin_x + sin_x @ k_operator)
    sin_x_correction = correction_prefactor * (k_operator @ cos_x + cos_x @ k_operator)

    dk = slate_core.metadata.volume.fundamental_stacked_dk(basis.metadata())[0][0]
    operator_prefactor = 1 / (np.sqrt(2) * dk)
    friction_const = 4 * params.kbt_div_hbar * params.mass * params.lambda_
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
def solve[M: EvenlySpacedLengthMetadata, E: AxisDirections, MT: TimeMetadata](
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
    The strength of the environmental interaction is carefully
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
        # TODO: we get a different answer if we first get the hamiltonain and then  # noqa: FIX002
        # convert to the new basis - we should investigate this further to make sure it's not a bug
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
    with windowed_wiener():
        result = qutip.ssesolve(  # type: ignore lib
            H=operator_as_diagonal_qobj(hamiltonian, normalized_basis)
            / normalized_params.hbar,
            psi0=state_as_qobj(initial_state, unnormalized_basis),
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
                "map": "parallel" if n_trajectories > 1 else None,
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


@timed
def get_eigenstate_energies[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    parameters: LangevinParameters,
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
) -> Array[FundamentalBasis, np.dtype[np.floating]]:
    """Get the energies of the simulation eigenstates for the Caldeira-Leggett model with periodic boundaries."""
    normalized_params = parameters.normalized_parameters

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
    return linalg.get_eigenvalues_hermitian(hamiltonian)


@timed
def solve_energies[
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
) -> tuple[
    Array[
        TupleBasis[tuple[FundamentalBasis, Basis[MT]], None],
        np.dtype[np.floating],
    ],
    Array[
        TupleBasis[tuple[FundamentalBasis, Basis[MT]], None],
        np.dtype[np.floating],
    ],
]:
    """Simulate the Caldeira-Leggett model using the Periodic Approach.

    To avoid numerical instabilities, the simulation uses a set of periodic
    environment operators (sin(x) and cos(x)).
    The strength of the environmental interaction is carefully
    chosen to match the behavior of the "standard" approach, which
    must use hard wall or adsorbing boundary conditions to avoid instabilities.

    This function also pre-calculates a suitable set of natural units
    for the simulation based on the Hamiltonian and initial state provided in the `condition`.

    Internally, this uses the `ssesolve` function from the `QuTiP` library.

    This returns a tuple of arrays (state_energy, state_norm)
    where each array has shape (n_trajectories, n_time_points).

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
    hamiltonian_qutip = operator_as_diagonal_qobj(hamiltonian, normalized_basis)
    with windowed_wiener():
        result = qutip.ssesolve(  # type: ignore lib
            H=hamiltonian_qutip / normalized_params.hbar,
            psi0=state_as_qobj(initial_state, unnormalized_basis),
            tlist=normalized_times,
            ntraj=n_trajectories,
            sc_ops=[
                operator_as_diagonal_qobj(op, normalized_basis)
                for op in environment_operators
            ],
            e_ops=[hamiltonian_qutip, qutip.qeye(normalized_basis.size)],  # type: ignore lib # cspell: disable-line
            options={
                "progress_bar": "enhanced",
                "store_states": False,
                "keep_runs_results": True,
                "method": as_qutip_name(kwargs.get("method", "Euler")),
                "dt": kwargs.get("target_delta", DEFAULT_TARGET_DELTA),
                "map": "parallel" if n_trajectories > 1 else None,
            },
            heterodyne=True,
        )

    return (
        Array(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_trajectories),
                    basis.as_fundamental(times),
                )
            ).upcast(),
            np.array(result.runs_expect[0], dtype=np.float64),  # type: ignore lib
        ),
        Array(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_trajectories),
                    basis.as_fundamental(times),
                )
            ).upcast(),
            np.array(result.runs_expect[1], dtype=np.float64),  # type: ignore lib
        ),
    )


def _squeeze_ratio_from_expectations(
    scattering: np.ndarray[Any, np.dtype[np.complex128]],
    scattering_k: np.ndarray[Any, np.dtype[np.complex128]],
    *,
    d_k: float,
    parameters: LangevinParameters,
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    """Get the squeeze ratio of a Gaussian state from <e^(i d_k x)> and <e^(i d_k x) k>.

    For a Gaussian wavefunction psi(x) ~ exp(-A (x - <x>)^2 + i <p> x / hbar)
    |<e^(i q x)>| = exp(-q^2 <delta x^2> / 2), which is exact on a periodic
    domain for a wrapped Gaussian, so <delta x^2> = -2 log(|<e^(i q x)>|) / q^2.
    The covariance of x and p is found from
    <e^(i q x) p> / <e^(i q x)> = <p> - hbar q / 2 + i q <{x - <x>, p - <p>}> / 2.
    Re(A) = 1 / (4 <delta x^2>) and Im(A) = -2 Re(A) covariance_xp / hbar.
    The squeeze ratio is R = 2 * dimensionless_mass * lengthscale^2 * A, such that
    <delta x^2> = dimensionless_mass * lengthscale^2 / (R + R*) and
    <delta p^2> = hbar^2 |R|^2 / (dimensionless_mass * lengthscale^2 (R + R*)).
    """
    delta_x_square = -2 * np.log(np.abs(scattering)) / d_k**2
    covariance_xp = parameters.hbar * np.imag(scattering_k / scattering) / d_k
    width_parameter = (1 - 2j * covariance_xp / parameters.hbar) / (4 * delta_x_square)
    return (
        2 * parameters.dimensionless_mass * parameters.lengthscale**2 * width_parameter
    ).astype(np.complex128)


def _get_location_observables(
    result: Any,  # noqa: ANN401
    *,
    scattering_scale: float,
    d_k: float,
    parameters: LangevinParameters,
) -> tuple[
    np.ndarray[Any, np.dtype[np.complexfloating]],
    np.ndarray[Any, np.dtype[np.complex128]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """Get (alpha, squeeze_ratio, norm) from the expectation values of solve_locations.

    The expectations are of e^(i d_k x) (up to a factor scattering_scale),
    k, the identity and e^(i d_k x) k.
    """
    norm = np.array(result.runs_expect[2], dtype=np.float64)
    scattering = np.array(result.runs_expect[0], dtype=np.complex128) / (
        norm * scattering_scale
    )
    scattering_k = np.array(result.runs_expect[3], dtype=np.complex128) / (
        norm * scattering_scale
    )
    alpha = parameters.eval_alpha(
        np.unwrap(np.angle(scattering)) / d_k,
        parameters.hbar * np.array(result.runs_expect[1], dtype=np.float64) / norm,
    )
    squeeze_ratio = _squeeze_ratio_from_expectations(
        scattering, scattering_k, d_k=d_k, parameters=parameters
    )
    return alpha, squeeze_ratio, norm


@timed
def solve_locations[
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
) -> tuple[
    Array[
        TupleBasis[tuple[FundamentalBasis, Basis[MT]], None],
        np.dtype[np.complexfloating],
    ],
    Array[
        TupleBasis[tuple[FundamentalBasis, Basis[MT]], None],
        np.dtype[np.complexfloating],
    ],
    Array[
        TupleBasis[tuple[FundamentalBasis, Basis[MT]], None],
        np.dtype[np.floating],
    ],
]:
    """Simulate the Caldeira-Leggett model using the Periodic Approach.

    To avoid numerical instabilities, the simulation uses a set of periodic
    environment operators (sin(x) and cos(x)).
    The strength of the environmental interaction is carefully
    chosen to match the behavior of the "standard" approach, which
    must use hard wall or adsorbing boundary conditions to avoid instabilities.

    This function also pre-calculates a suitable set of natural units
    for the simulation based on the Hamiltonian and initial state provided in the `condition`.

    Internally, this uses the `ssesolve` function from the `QuTiP` library.

    This returns a tuple of arrays (state_alpha, squeeze_ratio, state_norm)
    where each array has shape (n_trajectories, n_time_points).

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
    scattering_qobj = operator_as_diagonal_qobj(
        operator.build.scattering_operator(normalized_metadata, n_k=(1,)),
        normalized_basis,
    )
    with windowed_wiener():
        result = qutip.ssesolve(  # type: ignore lib
            H=operator_as_diagonal_qobj(hamiltonian, normalized_basis)
            / normalized_params.hbar,
            psi0=state_as_qobj(initial_state, unnormalized_basis),
            tlist=normalized_times,
            ntraj=n_trajectories,
            sc_ops=[
                operator_as_diagonal_qobj(op, normalized_basis)
                for op in environment_operators
            ],
            e_ops=[
                scattering_qobj,
                operator_as_diagonal_qobj(
                    operator.build.k(normalized_metadata, axis=0),
                    normalized_basis,
                ),
                qutip.qeye(normalized_basis.size),  # type: ignore lib # cspell: disable-line
                scattering_qobj  # type: ignore lib
                * operator_as_diagonal_qobj(
                    operator.build.k(normalized_metadata, axis=0),
                    normalized_basis,
                ),
            ],
            options={
                "progress_bar": "enhanced",
                "store_states": False,
                "keep_runs_results": True,
                "method": as_qutip_name(kwargs.get("method", "Euler")),
                "dt": kwargs.get("target_delta", DEFAULT_TARGET_DELTA),
                "map": "parallel" if n_trajectories > 1 else None,
            },
            heterodyne=True,
        )

    alpha, squeeze_ratio, norm = _get_location_observables(
        result,
        scattering_scale=np.abs(scattering_qobj.full()).max(),  # type: ignore lib
        d_k=(2 * np.pi) / normalized_basis.metadata().children[0].delta,
        parameters=normalized_params,
    )
    return (  # pyright: ignore[reportReturnType] the basis is upcast
        Array(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_trajectories),
                    basis.as_fundamental(times),
                )
            ).upcast(),
            rescale_alpha(
                alpha,
                in_parameter=normalized_params,
                out_parameter=parameters,
            ),
        ),
        Array(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_trajectories),
                    basis.as_fundamental(times),
                )
            ).upcast(),
            squeeze_ratio.astype(np.complex128),
        ),
        Array(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_trajectories),
                    basis.as_fundamental(times),
                )
            ).upcast(),
            norm,
        ),
    )

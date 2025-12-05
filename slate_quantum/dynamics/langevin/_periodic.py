import datetime
from typing import TYPE_CHECKING, Any, Unpack, cast

import numpy as np
from slate_core import (
    Array,
    EvenlySpacedLengthMetadata,
    FundamentalBasis,
    TupleBasis,
    TupleMetadata,
    array,
    basis,
)
from slate_core.metadata.volume import AxisDirections, fundamental_stacked_dk
from slate_core.util import timed

from slate_quantum.dynamics.langevin._util import (
    LangevinParameters,
    SSEConfig,
    rescale_alpha,
    rescale_alpha_arr,
    rescale_times,
)
from slate_quantum.metadata import TimeMetadata

try:
    import sse_solver_py  # type: ignore lib
except ImportError:
    sse_solver_py = None

if TYPE_CHECKING:
    from slate_core import Basis, Ctype
    from sse_solver_py import PeriodicLangevinSystemParameters

    from slate_quantum.dynamics._realization import RealizationListIndexMetadata
    from slate_quantum.operator._operator import Operator, OperatorBasis


def get_internal_parameters[M: EvenlySpacedLengthMetadata, E: AxisDirections](  # type: ignore lib
    parameters: LangevinParameters,
    potential: tuple[float, np.ndarray[Any, np.dtype[np.complex128]]],
) -> PeriodicLangevinSystemParameters:
    if sse_solver_py is None:
        msg = "sse_solver_py is not installed"
        raise ImportError(msg)
    return sse_solver_py.PeriodicLangevinSystemParameters(  # type: ignore lib
        dimensionless_lambda=parameters.dimensionless_lambda,
        dimensionless_mass=parameters.dimensionless_mass,
        dimensionless_potential=potential[1].tolist(),
        dk_times_lengthscale=potential[0],
        kbt_div_hbar=parameters.kbt_div_hbar,
    )


def _truncate_potential(
    components: np.ndarray[Any, np.dtype[np.complex128]], tol: float = 1e-4
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    # Find the first term from the end of the array that is above the tolerance
    # and truncate the array to that length
    is_significant = np.where(np.abs(components) > tol)[0]
    if is_significant.size == 0:
        return components[:0]
    return components[: is_significant[-1] + 1]


def get_dimensionless_potential_params[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    parameters: LangevinParameters,
) -> tuple[float, np.ndarray[Any, np.dtype[np.complex128]]]:
    assert potential.basis.metadata().children[0].n_dim == 1, "Potential must be 1D"
    stacked_dk = fundamental_stacked_dk(potential.basis.metadata().children[0])
    dk = np.linalg.norm(stacked_dk[0]).item()
    lengthscale = parameters.characteristic_length

    as_diagonal = array.extract_diagonal(potential).as_array() / parameters.kbt
    real_components = np.fft.rfft(as_diagonal.real)
    return (dk * lengthscale, _truncate_potential(real_components))


@timed
def solve_periodic_langevin[
    MT: TimeMetadata,
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    initial_state: complex,
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    **kwargs: Unpack[SSEConfig],
) -> Array[
    Basis[RealizationListIndexMetadata[MT]],
    np.dtype[np.complexfloating],
]:
    r"""Solve the dynamics of a harmonic oscillator coupled to a thermal bath using the harmonic Langevin equation.

    Raises
    ------
    ImportError
        If the rust sse_solver_py is not installed
    """
    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)

    times_basis = basis.as_index(times)
    normalized_params = parameters.normalized_parameters

    normalized_times = rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_characteristic_time=parameters.characteristic_time,
        out_characteristic_time=normalized_params.characteristic_time,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)

    ts = datetime.datetime.now(tz=datetime.UTC)
    data = sse_solver_py.solve_periodic_langevin(  # type: ignore lib
        rescale_alpha(
            initial_state, in_parameter=parameters, out_parameter=normalized_params
        ),
        get_internal_parameters(
            normalized_params, get_dimensionless_potential_params(potential, parameters)
        ),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if kwargs.get("adaptive") else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),
        ),
    )
    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201
    data = np.array(cast("list[complex]", data))  # pyright: ignore[reportUnnecessaryCast]

    alpha_res = rescale_alpha_arr(
        data, out_parameter=parameters, in_parameter=normalized_params
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return Array(out_basis, alpha_res)


@timed
def solve_periodic_stable_quantum_langevin[
    MT: TimeMetadata,
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
](
    initial_state: tuple[complex, complex],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ],
    **kwargs: Unpack[SSEConfig],
) -> tuple[
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.complexfloating],
    ],
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.complexfloating],
    ],
]:
    r"""Solve the dynamics of a harmonic oscillator coupled to a thermal bath using the harmonic Langevin equation.

    Raises
    ------
    ImportError
        If the rust sse_solver_py is not installed
    """
    ts = datetime.datetime.now(tz=datetime.UTC)

    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)

    times_basis = basis.as_index(times)
    normalized_params = parameters.normalized_parameters
    normalized_times = rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_characteristic_time=parameters.characteristic_time,
        out_characteristic_time=normalized_params.characteristic_time,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    data = sse_solver_py.solve_periodic_stable_quantum_langevin(  # type: ignore lib
        (
            rescale_alpha(
                initial_state[0],
                in_parameter=parameters,
                out_parameter=normalized_params,
            ),
            initial_state[1],
        ),
        get_internal_parameters(
            normalized_params, get_dimensionless_potential_params(potential, parameters)
        ),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if kwargs.get("adaptive") else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),
        ),
    )
    data = np.array(cast("list[complex]", data)).reshape(  # pyright: ignore[reportUnnecessaryCast]
        (n_trajectories, normalized_times.size, 2)
    )

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201

    alpha_res = rescale_alpha_arr(
        data[:, :, 0], out_parameter=parameters, in_parameter=normalized_params
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return (Array(out_basis, alpha_res), Array(out_basis, data[:, :, 1]))

import datetime
from typing import TYPE_CHECKING, Unpack, cast

import numpy as np
from slate_core import (
    Array,
    FundamentalBasis,
    TupleBasis,
    basis,
)
from slate_core.metadata import SimpleMetadata
from slate_core.util import timed

from slate_quantum.dynamics.langevin._util import (
    RustSSEConfig,
    rescale_alpha,
    rescale_times,
)
from slate_quantum.metadata import TimeMetadata
from slate_quantum.state import StateList

try:
    import sse_solver_py  # type: ignore lib
except ImportError:
    sse_solver_py = None

if TYPE_CHECKING:
    from slate_core import Basis, Ctype
    from sse_solver_py import DoubleHarmonicLangevinSystemParameters  # type: ignore lib

    from slate_quantum import State
    from slate_quantum.dynamics import RealizationListBasis
    from slate_quantum.dynamics._realization import RealizationListIndexMetadata
    from slate_quantum.dynamics.langevin._util import LangevinParameters
    from slate_quantum.operator._build._potential import DoubleHarmonicParameters


def get_internal_parameters(  # type: ignore lib
    parameters: LangevinParameters,
    dimensionless_omega_barrier: float,
    left_distance_div_lengthscale: float,
    right_distance_div_lengthscale: float,
) -> DoubleHarmonicLangevinSystemParameters:
    if sse_solver_py is None:
        msg = "sse_solver_py is not installed"
        raise ImportError(msg)
    return sse_solver_py.DoubleHarmonicLangevinSystemParameters(  # type: ignore lib
        dimensionless_lambda=parameters.dimensionless_lambda,
        dimensionless_mass=parameters.dimensionless_mass,
        dimensionless_omega_barrier=dimensionless_omega_barrier,
        kbt_div_hbar=parameters.kbt_div_hbar,
        left_distance_div_lengthscale=left_distance_div_lengthscale,
        right_distance_div_lengthscale=right_distance_div_lengthscale,
    )


@timed
def solve_double_harmonic_langevin[
    MT: TimeMetadata,
](
    initial_state: complex,
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: DoubleHarmonicParameters,
    **kwargs: Unpack[RustSSEConfig],
) -> Array[
    Basis[RealizationListIndexMetadata[MT]],
    np.dtype[np.complexfloating],
]:
    r"""Solve the dynamics of a double harmonic oscillator coupled to a thermal bath using the harmonic Langevin equation.

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
        in_parameter=parameters,
        out_parameter=normalized_params,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    adaptive = kwargs.get("adaptive", False)
    if not adaptive:
        print(  # noqa: T201
            f"Simulating a total of {normalized_times[-1] / target_delta:0.2g} timesteps"
        )
    ts = datetime.datetime.now(tz=datetime.UTC)
    data = sse_solver_py.solve_double_harmonic_langevin(  # type: ignore lib
        rescale_alpha(
            initial_state, in_parameter=parameters, out_parameter=normalized_params
        ),
        get_internal_parameters(
            normalized_params,
            dimensionless_omega_barrier=potential.omega_barrier
            / parameters.kbt_div_hbar,
            left_distance_div_lengthscale=potential.left_distance
            / parameters.lengthscale,
            right_distance_div_lengthscale=potential.right_distance
            / parameters.lengthscale,
        ),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if adaptive else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),
        ),
    )
    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201
    data = np.array(cast("list[complex]", data))  # pyright: ignore[reportUnnecessaryCast]

    alpha_res = rescale_alpha(
        data, out_parameter=parameters, in_parameter=normalized_params
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return Array(out_basis, alpha_res)


@timed
def solve_double_harmonic_stable_quantum_langevin[
    MT: TimeMetadata,
](
    initial_state: tuple[complex, complex],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: DoubleHarmonicParameters,
    **kwargs: Unpack[RustSSEConfig],
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
        in_parameter=parameters,
        out_parameter=normalized_params,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    adaptive = kwargs.get("adaptive", False)
    if not adaptive:
        print(  # noqa: T201
            f"Simulating a total of {normalized_times[-1] / target_delta:0.2g} timesteps"
        )
    data = sse_solver_py.solve_double_harmonic_stable_quantum_langevin(  # type: ignore lib
        (
            rescale_alpha(
                initial_state[0],
                in_parameter=parameters,
                out_parameter=normalized_params,
            ),
            initial_state[1],
        ),
        get_internal_parameters(
            normalized_params,
            dimensionless_omega_barrier=potential.omega_barrier
            / parameters.kbt_div_hbar,
            left_distance_div_lengthscale=potential.left_distance
            / parameters.lengthscale,
            right_distance_div_lengthscale=potential.right_distance
            / parameters.lengthscale,
        ),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if adaptive else None,
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

    alpha_res = rescale_alpha(
        data[:, :, 0], out_parameter=parameters, in_parameter=normalized_params
    )
    ratio_res = data[:, :, 1]
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return (Array(out_basis, alpha_res), Array(out_basis, ratio_res))


@timed
def solve_double_harmonic_quantum_langevin[
    MT: TimeMetadata,
    MS: SimpleMetadata,
](
    initial_state: tuple[
        complex, complex, State[Basis[MS], np.dtype[np.complexfloating]]
    ],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: LangevinParameters,
    potential: DoubleHarmonicParameters,
    **kwargs: Unpack[RustSSEConfig],
) -> tuple[
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.complexfloating],
    ],
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.complexfloating],
    ],
    StateList[RealizationListBasis[MT, MS], np.dtype[np.complexfloating]],
]:
    ts = datetime.datetime.now(tz=datetime.UTC)

    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)

    times_basis = basis.as_index(times)
    normalized_params = parameters.normalized_parameters
    normalized_times = rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_parameter=parameters,
        out_parameter=normalized_params,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    adaptive = kwargs.get("adaptive", False)
    if not adaptive:
        print(  # noqa: T201
            f"Simulating a total of {normalized_times[-1] / target_delta:0.2g} timesteps"
        )
    data = sse_solver_py.solve_double_harmonic_quantum_langevin(  # type: ignore lib
        (
            rescale_alpha(
                initial_state[0],
                in_parameter=parameters,
                out_parameter=normalized_params,
            ),
            initial_state[1],
            initial_state[2].as_array().tolist(),
        ),
        get_internal_parameters(
            normalized_params,
            dimensionless_omega_barrier=potential.omega_barrier
            / parameters.kbt_div_hbar,
            left_distance_div_lengthscale=potential.left_distance
            / parameters.lengthscale,
            right_distance_div_lengthscale=potential.right_distance
            / parameters.lengthscale,
        ),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if adaptive else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),
        ),
    )
    data = np.array(cast("list[complex]", data)).reshape(  # pyright: ignore[reportUnnecessaryCast]
        (n_trajectories, normalized_times.size, -1)
    )

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201

    alpha_res = rescale_alpha(
        data[:, :, 0], out_parameter=parameters, in_parameter=normalized_params
    )
    ratio_res = data[:, :, 1]
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    out_states = StateList(
        TupleBasis((out_basis, basis.as_fundamental(initial_state[2].basis))).upcast(),
        data[:, :, 2:],
    )
    return (Array(out_basis, alpha_res), Array(out_basis, ratio_res), out_states)

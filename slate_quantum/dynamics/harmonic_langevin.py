import dataclasses
import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack, cast, overload

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore lib
from slate_core import (
    Array,
    FundamentalBasis,
    TupleBasis,
    basis,
)
from slate_core.util import timed

from slate_quantum.metadata import TimeMetadata

try:
    import sse_solver_py  # type: ignore lib
except ImportError:
    sse_solver_py = None

if TYPE_CHECKING:
    from slate_core import Basis, Ctype
    from sse_solver_py import (  # type: ignore lib
        HarmonicLangevinSystemParameters,  # type: ignore lib
    )

    from slate_quantum.dynamics._realization import RealizationListIndexMetadata


@dataclass(kw_only=True, frozen=True)
class HarmonicParameters:
    """Parameters for a harmonic system."""

    temperature: float
    omega: float
    lambda_: float
    mass: float
    hbar: float = hbar
    boltzmann: float = Boltzmann
    lengthscale: float = 1.0

    @property
    def kbt_div_hbar(self) -> float:
        """Return kB * T / hbar."""
        return self.boltzmann * self.temperature / self.hbar

    @property
    def dimensionless_omega(self) -> float:
        """Return the dimensionless omega."""
        return self.omega / self.kbt_div_hbar

    @property
    def dimensionless_lambda(self) -> float:
        """Return the dimensionless lambda_."""
        return self.lambda_ / self.kbt_div_hbar

    @property
    def dimensionless_mass(self) -> float:
        """Return the dimensionless mass."""
        return self.hbar / (2 * self.mass * self.kbt_div_hbar * self.lengthscale**2)

    @property
    def characteristic_length(self) -> float:
        """Return the characteristic length."""
        return np.sqrt(self.dimensionless_mass) * self.lengthscale

    @property
    def characteristic_time(self) -> float:
        """Return the characteristic time."""
        return 1 / self.kbt_div_hbar

    @overload
    def eval_alpha(self, x: float, p: float) -> complex: ...

    @overload
    def eval_alpha(
        self,
        x: np.ndarray[Any, np.dtype[np.floating]],
        p: np.ndarray[Any, np.dtype[np.floating]],
    ) -> np.ndarray[Any, np.dtype[np.complexfloating]]: ...

    def eval_alpha(
        self,
        x: float | np.ndarray[Any, np.dtype[np.floating]],
        p: float | np.ndarray[Any, np.dtype[np.floating]],
    ) -> complex | np.ndarray[Any, np.dtype[np.complexfloating]]:
        """Evaluate the coherent state alpha parameter for the harmonic oscillator."""
        out = (x / self.lengthscale) + 1j * (p * self.lengthscale) / self.hbar
        return out / np.sqrt(2)

    @overload
    def eval_xp(self, alpha: complex) -> tuple[float, float]: ...

    @overload
    def eval_xp(
        self,
        alpha: np.ndarray[Any, np.dtype[np.complexfloating]],
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.floating]], np.ndarray[Any, np.dtype[np.floating]]
    ]: ...

    def eval_xp(
        self, alpha: complex | np.ndarray[Any, np.dtype[np.complexfloating]]
    ) -> tuple[
        float | np.ndarray[Any, np.dtype[np.floating]],
        float | np.ndarray[Any, np.dtype[np.floating]],
    ]:
        """Evaluate the coherent state alpha parameter for the harmonic oscillator."""
        x = (alpha.real * np.sqrt(2)) * self.lengthscale
        p = (alpha.imag * np.sqrt(2)) * (self.hbar / self.lengthscale)
        return (x, p)


def _get_normalized_parameters(parameters: HarmonicParameters) -> HarmonicParameters:
    characteristic_time = 1 / parameters.kbt_div_hbar

    out = HarmonicParameters(
        temperature=parameters.temperature,
        omega=parameters.omega * characteristic_time,
        lambda_=parameters.lambda_ * characteristic_time,
        mass=parameters.mass,
        hbar=parameters.hbar * characteristic_time,
        boltzmann=parameters.boltzmann * characteristic_time**2,
        lengthscale=parameters.characteristic_length,
    )

    return dataclasses.replace(out, lengthscale=out.characteristic_length)


def _get_internal_parameters(  # type: ignore lib
    parameters: HarmonicParameters,
) -> HarmonicLangevinSystemParameters:
    if sse_solver_py is None:
        msg = "sse_solver_py is not installed"
        raise ImportError(msg)
    return sse_solver_py.HarmonicLangevinSystemParameters(  # type: ignore lib
        dimensionless_lambda=parameters.dimensionless_lambda,
        dimensionless_mass=parameters.dimensionless_mass,
        dimensionless_omega=parameters.dimensionless_omega,
        kbt_div_hbar=parameters.kbt_div_hbar,
    )


def _rescale_times(
    times: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    in_characteristic_time: float,
    out_characteristic_time: float,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Rescale the times to the characteristic time."""
    return times * (out_characteristic_time / in_characteristic_time)


def _rescale_alpha(
    alpha: complex,
    *,
    in_parameter: HarmonicParameters,
    out_parameter: HarmonicParameters,
) -> complex:
    """Get the initial alpha for the harmonic oscillator coherent state."""
    sf_len = out_parameter.lengthscale / in_parameter.lengthscale
    sf_time = out_parameter.characteristic_time / in_parameter.characteristic_time
    return alpha.real * sf_len + 1j * alpha.imag / (sf_len * sf_time)


def _rescale_alpha_arr(
    alpha: np.ndarray[Any, np.dtype[np.complexfloating]],
    *,
    in_parameter: HarmonicParameters,
    out_parameter: HarmonicParameters,
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Get the initial alpha for the harmonic oscillator coherent state."""
    sf_len = out_parameter.lengthscale / in_parameter.lengthscale
    sf_time = out_parameter.characteristic_time / in_parameter.characteristic_time
    return alpha.real * sf_len + 1j * alpha.imag / (sf_len * sf_time)


SSEMethod = Literal[
    "Euler",
    "NormalizedEuler",
    "Milsten",
    "Order2ExplicitWeak",
    "NormalizedOrder2ExplicitWeak",
    "Order2ExplicitWeakR5",
    "NormalizedOrder2ExplicitWeakR5",
]


class SSEConfig(TypedDict, total=False):
    """Configuration for the stochastic schrodinger equation solver."""

    n_trajectories: int
    method: SSEMethod
    target_delta: float
    adaptive: bool


@timed
def solve_harmonic_langevin[
    MT: TimeMetadata,
](
    initial_state: complex,
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: HarmonicParameters,
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
    ts = datetime.datetime.now(tz=datetime.UTC)

    if sse_solver_py is None:
        msg = "sse_solver_py is not installed, please install it using `pip install slate_quantum[sse_solver_py]`"
        raise ImportError(msg)

    times_basis = basis.as_index(times)
    normalized_params = _get_normalized_parameters(parameters)

    normalized_times = _rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_characteristic_time=parameters.characteristic_time,
        out_characteristic_time=normalized_params.characteristic_time,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    data = sse_solver_py.solve_harmonic_langevin(  # type: ignore lib
        _rescale_alpha(
            initial_state, in_parameter=parameters, out_parameter=normalized_params
        ),
        _get_internal_parameters(normalized_params),
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if kwargs.get("adaptive") else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),
        ),
    )
    data = np.array(cast("list[complex]", data))  # pyright: ignore[reportUnnecessaryCast]

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201

    alpha_res = _rescale_alpha_arr(
        data, out_parameter=parameters, in_parameter=normalized_params
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return Array(out_basis, alpha_res)


@timed
def solve_harmonic_quantum_langevin[
    MT: TimeMetadata,
](
    initial_state: tuple[complex, complex],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: HarmonicParameters,
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
    normalized_params = _get_normalized_parameters(parameters)
    normalized_times = _rescale_times(
        times_basis.metadata().values[times_basis.points],
        in_characteristic_time=parameters.characteristic_time,
        out_characteristic_time=normalized_params.characteristic_time,
    )

    target_delta = kwargs.get("target_delta", 1e-3)
    n_trajectories = kwargs.get("n_trajectories", 1)
    data = sse_solver_py.solve_harmonic_quantum_langevin(  # type: ignore lib
        (
            _rescale_alpha(
                initial_state[0],
                in_parameter=parameters,
                out_parameter=normalized_params,
            ),
            initial_state[1],
        ),
        _get_internal_parameters(normalized_params),
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

    alpha_res = _rescale_alpha_arr(
        data[:, :, 0], out_parameter=parameters, in_parameter=normalized_params
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return (Array(out_basis, alpha_res), Array(out_basis, data[:, :, 1]))

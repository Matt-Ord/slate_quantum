import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, Unpack

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
        SSEMethod,  # type: ignore lib
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


def _get_normalized_parameters(  # type: ignore lib
    parameters: HarmonicParameters,
    times: np.ndarray[tuple[int], np.dtype[np.floating]],
) -> tuple[
    HarmonicLangevinSystemParameters,
    tuple[float, float],
    np.ndarray[tuple[int], np.dtype[np.floating]],
]:
    characteristic_length = np.sqrt(
        parameters.hbar / (2 * parameters.mass * parameters.kbt_div_hbar)
    )

    characteristic_time = 1 / parameters.kbt_div_hbar
    if sse_solver_py is None:
        msg = "sse_solver_py is not installed"
        raise ImportError(msg)
    return (
        sse_solver_py.HarmonicLangevinSystemParameters(  # type: ignore lib
            dimensionless_lambda=parameters.dimensionless_lambda,
            dimensionless_mass=1.0,
            dimensionless_omega=parameters.dimensionless_omega,
            kbt_div_hbar=1.0,
        ),
        (characteristic_length, parameters.hbar * characteristic_time),
        times / characteristic_time,
    )


def _get_initial_alpha(
    initial_state: tuple[float, float], *, lengthscale: float, hbar: float
) -> complex:
    """Get the initial alpha for the harmonic oscillator coherent state."""
    x0, p0 = initial_state
    return (x0 / lengthscale + 1j * (lengthscale / hbar) * p0) / np.sqrt(2)


def _split_simulation_result(
    result: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
    *,
    lengthscale: float,
    hbar: float,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
    np.ndarray[tuple[int, int], np.dtype[np.floating]],
]:
    x = (np.sqrt(2) * lengthscale) * result.real
    p = (np.sqrt(2) * hbar / lengthscale) * result.imag
    return (x, p)


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
    initial_state: tuple[float, float],
    times: Basis[MT, Ctype[np.complexfloating]],
    parameters: HarmonicParameters,
    **kwargs: Unpack[SSEConfig],
) -> tuple[
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.floating],
    ],
    Array[
        Basis[RealizationListIndexMetadata[MT]],
        np.dtype[np.floating],
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
    normalized_params, (lengthscale, hbar), normalized_times = (  # type: ignore lib
        _get_normalized_parameters(
            parameters, times_basis.metadata().values[times_basis.points]
        )
    )

    target_delta = kwargs.get("target_delta", 1e-3)  # type: ignore lib
    n_trajectories = kwargs.get("n_trajectories", 1)  # type: ignore lib
    adaptive = kwargs.get("adaptive", False)  # type: ignore lib
    data = sse_solver_py.solve_harmonic_langevin(  # type: ignore lib
        _get_initial_alpha(initial_state, lengthscale=lengthscale, hbar=hbar),
        normalized_params,
        sse_solver_py.SimulationConfig(  # type: ignore lib
            times=normalized_times.tolist(),
            dt=target_delta,
            delta=(None, target_delta, None) if adaptive else None,
            n_trajectories=n_trajectories,
            n_realizations=1,
            method=kwargs.get("method", "Euler"),  # type: ignore lib
        ),
    )

    te = datetime.datetime.now(tz=datetime.UTC)
    print(f"solve_harmonic_langevin took: {(te - ts).total_seconds()} sec")  # noqa: T201

    x_res, p_res = _split_simulation_result(
        np.array(data),  # type: ignore lib
        lengthscale=lengthscale,
        hbar=hbar,
    )
    out_basis = TupleBasis(
        (FundamentalBasis.from_size(n_trajectories), times_basis)
    ).upcast()
    return (Array(out_basis, x_res), Array(out_basis, p_res))

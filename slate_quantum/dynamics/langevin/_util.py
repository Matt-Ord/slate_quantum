from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload

import numpy as np
import slate_core
from scipy.constants import Boltzmann, hbar  # type: ignore lib
from slate_core import EvenlySpacedLengthMetadata
from slate_core.metadata import AxisDirections, TupleMetadata


@dataclass(kw_only=True, frozen=True)
class LangevinParameters:
    """Parameters for a harmonic system."""

    temperature: float
    lambda_: float
    mass: float
    hbar: float = hbar
    boltzmann: float = Boltzmann
    lengthscale: float = 1.0

    @property
    def kbt(self) -> float:
        """Return K_b T."""
        return self.boltzmann * self.temperature

    @property
    def kbt_div_hbar(self) -> float:
        """Return K_b T / hbar."""
        return self.kbt / self.hbar

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

    @property
    def normalized_parameters(self: LangevinParameters) -> LangevinParameters:
        sf = self.characteristic_time

        out = LangevinParameters(
            temperature=self.temperature,
            lambda_=self.lambda_ * sf,
            mass=self.mass,
            hbar=self.hbar * sf,
            boltzmann=self.boltzmann * sf**2,
            lengthscale=self.characteristic_length,
        )
        assert np.isclose(out.characteristic_time, 1.0)
        assert np.isclose(out.dimensionless_mass, 1.0)

        return out


def rescale_times(
    times: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Rescale the times to the characteristic time."""
    sf_time = out_parameter.characteristic_time / in_parameter.characteristic_time
    return times * sf_time


@overload
def rescale_alpha(
    alpha: np.ndarray[Any, np.dtype[np.complexfloating]],
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> np.ndarray[Any, np.dtype[np.complexfloating]]: ...


@overload
def rescale_alpha(
    alpha: complex,
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> complex: ...


def rescale_alpha(
    alpha: complex | np.ndarray[Any, np.dtype[np.complexfloating]],
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> complex | np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Rescale the coherent state alpha parameter."""
    sf_len = out_parameter.lengthscale / in_parameter.lengthscale

    return alpha.real / sf_len + 1j * alpha.imag * sf_len


def rescale_simulation_metadata[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
    in_parameters: LangevinParameters,
    out_parameters: LangevinParameters,
) -> TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], AxisDirections]:
    """Create a basis with n_repeat repeats of the periodic potential."""
    vectors = slate_core.metadata.volume.fundamental_stacked_delta_x(metadata)
    vectors = tuple(
        v * (out_parameters.characteristic_length / in_parameters.characteristic_length)
        for v in vectors
    )
    return slate_core.metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        vectors,
        shape=metadata.shape,
    )


type SSEMethod = Literal[
    "Euler",
    "NormalizedEuler",
    "Milstein",
    "Order2ExplicitWeak",
    "NormalizedOrder2ExplicitWeak",
    "Order2ExplicitWeakR5",
    "NormalizedOrder2ExplicitWeakR5",
]


class SSEConfig(TypedDict, total=False):
    """Configuration for the stochastic schrodinger equation solver."""

    n_trajectories: int
    target_delta: float


type RustSSEMethod = Literal[
    "Euler",
    "NormalizedEuler",
    "Milstein",
    "Order2ExplicitWeak",
    "NormalizedOrder2ExplicitWeak",
    "Order2ExplicitWeakR5",
    "NormalizedOrder2ExplicitWeakR5",
]


class RustSSEConfig(SSEConfig, total=False):
    """Configuration for the stochastic schrodinger equation solver in rust."""

    method: RustSSEMethod
    adaptive: bool


type QutipSSEMethod = Literal[
    "Euler",
    "Platen",
    "Rouchon",
    "Order1.5Explicit",
]


class QutipSSEConfig(SSEConfig, total=False):
    """Configuration for the stochastic schrodinger equation solver in qutip."""

    method: QutipSSEMethod


def as_qutip_name(method: QutipSSEMethod) -> str:
    match method:
        case "Euler":
            return "euler"
        case "Platen":
            return "platen"
        case "Rouchon":
            return "rouchon"
        case "Order1.5Explicit":
            return "explicit1.5"
        case _:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)

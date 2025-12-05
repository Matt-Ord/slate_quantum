import dataclasses
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, overload

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore lib


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
        characteristic_time = 1 / self.kbt_div_hbar

        out = LangevinParameters(
            temperature=self.temperature,
            lambda_=self.lambda_ * characteristic_time,
            mass=self.mass,
            hbar=self.hbar * characteristic_time,
            boltzmann=self.boltzmann * characteristic_time**2,
            lengthscale=self.characteristic_length,
        )

        return dataclasses.replace(out, lengthscale=out.characteristic_length)


def rescale_times(
    times: np.ndarray[tuple[int], np.dtype[np.floating]],
    *,
    in_characteristic_time: float,
    out_characteristic_time: float,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
    """Rescale the times to the characteristic time."""
    return times * (out_characteristic_time / in_characteristic_time)


@overload
def rescale_energy(
    energy: float,
    *,
    in_characteristic_time: float,
    out_characteristic_time: float,
) -> float: ...


@overload
def rescale_energy(
    energy: np.ndarray[Any, np.dtype[np.floating]],
    *,
    in_characteristic_time: float,
    out_characteristic_time: float,
) -> np.ndarray[tuple[int], np.dtype[np.floating]]: ...


def rescale_energy(
    energy: float | np.ndarray[Any, np.dtype[np.floating]],
    *,
    in_characteristic_time: float,
    out_characteristic_time: float,
) -> np.ndarray[tuple[int], np.dtype[np.floating]] | float:
    """Rescale the times to the characteristic time."""
    return energy * (in_characteristic_time / out_characteristic_time)


def rescale_alpha(
    alpha: complex,
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> complex:
    """Get the initial alpha for the harmonic oscillator coherent state."""
    sf_len = out_parameter.lengthscale / in_parameter.lengthscale
    sf_time = out_parameter.characteristic_time / in_parameter.characteristic_time
    return alpha.real * sf_len + 1j * alpha.imag / (sf_len * sf_time)


def rescale_alpha_arr(
    alpha: np.ndarray[Any, np.dtype[np.complexfloating]],
    *,
    in_parameter: LangevinParameters,
    out_parameter: LangevinParameters,
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    """Get the initial alpha for the harmonic oscillator coherent state."""
    sf_len = out_parameter.lengthscale / in_parameter.lengthscale
    sf_time = out_parameter.characteristic_time / in_parameter.characteristic_time
    return alpha.real * sf_len + 1j * alpha.imag / (sf_len * sf_time)


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
    method: SSEMethod
    target_delta: float
    adaptive: bool

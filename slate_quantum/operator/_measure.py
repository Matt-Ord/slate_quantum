from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate import Array, BasisMetadata, array, metadata
from slate.basis import from_metadata
from slate.metadata import AxisDirections, SpacedLengthMetadata

from slate_quantum import state as _state
from slate_quantum.operator import _build
from slate_quantum.operator._build._position import scattering_operator
from slate_quantum.operator._operator import expectation, expectation_of_each

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate import StackedMetadata

    from slate_quantum.state._state import State, StateList


def potential_from_function[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[StackedMetadata[M1, E]],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> float:
    """Get the expectation of a generic potential of a state."""
    momentum = _build.potential_from_function(
        state.basis.metadata(), fn=fn, wrapped=wrapped, offset=offset
    )

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def all_potential_from_function[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> Array[M0, np.floating]:
    """Get the expectation of a generic potential of all states."""
    momentum = _build.potential_from_function(
        states.basis.metadata()[1], fn=fn, wrapped=wrapped, offset=offset
    )

    states = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, states))


def x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[StackedMetadata[M1, E]],
    *,
    ax: int,
    offset: float = 0,
    wrapped: bool = False,
) -> float:
    """Get the position of a state."""
    momentum = _build.x(state.basis.metadata(), ax=ax, offset=offset, wrapped=wrapped)

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def _get_fundamental_scatter[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](states: State[StackedMetadata[M1, E]], *, ax: int) -> complex:
    r"""Get the scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    n_dim = len(states.basis.metadata().children)
    n_k = tuple(1 if i == ax else 0 for i in range(n_dim))
    scatter = scattering_operator(states.basis.metadata(), n_k=n_k)

    states = _state.normalize(states)
    return expectation(scatter, states)


def periodic_x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](state: State[StackedMetadata[M1, E]], *, ax: int) -> float:
    r"""Get the periodic position coordinate of a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_fundamental_scatter(state, ax=ax)

    angle = np.angle(scatter)
    wrapped = np.mod(angle, (2 * np.pi))
    delta_x = metadata.volume.fundamental_stacked_delta_x(state.basis.metadata())
    return wrapped * (np.linalg.norm(delta_x[ax]).item() / (2 * np.pi))


def variance_x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[StackedMetadata[M1, E]],
    *,
    ax: int,
) -> float:
    r"""Get the width of a Gaussian wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    x = periodic_x(state, ax=ax)

    offset = tuple(-x if i == ax else 0 for i in range(state.basis.metadata().n_dim))
    return potential_from_function(
        state,
        fn=lambda pos: pos[ax].astype(np.complex128) ** 2,
        wrapped=True,
        offset=offset,
    )


def coherent_width[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[StackedMetadata[M1, E]],
    *,
    ax: int,
) -> float:
    """Get the width of a Gaussian wavepacket."""
    variance = variance_x(state, ax=ax)
    return np.sqrt(2 * variance)


def all_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.floating]:
    """Get the position of all states."""
    momentum = _build.x(states.basis.metadata()[1], ax=ax)

    states = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, states))


def _get_all_fundamental_scatter[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.complexfloating]:
    r"""Get the scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    n_dim = len(states.basis.metadata()[1].children)
    n_k = tuple(1 if i == ax else 0 for i in range(n_dim))
    scatter = scattering_operator(states.basis.metadata()[1], n_k=n_k)

    states = _state.normalize_all(states)
    return expectation_of_each(scatter, states)


def all_periodic_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.floating]:
    r"""Get the periodic position coordinate of a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_all_fundamental_scatter(states, ax=ax)

    angle = array.angle(scatter)
    wrapped = array.mod(angle, (2 * np.pi))
    delta_x = metadata.volume.fundamental_stacked_delta_x(states.basis.metadata()[1])
    return wrapped * (np.linalg.norm(delta_x[ax]).item() / (2 * np.pi))


def all_variance_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.floating]:
    r"""Get the width of a Gaussian wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    basis = from_metadata(states.basis.metadata()[0])
    data = np.array([variance_x(state, ax=ax) for state in states])
    return Array(basis, data)


def all_coherent_width[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.floating]:
    """Get the width of a Gaussian wavepacket."""
    variance = all_variance_x(states, ax=ax)
    return array.sqrt(variance * 2)


def k[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[StackedMetadata[M1, E]],
    *,
    ax: int,
) -> float:
    """Get the momentum of a state."""
    momentum = _build.k(state.basis.metadata(), idx=ax)

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def all_k[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[M0, StackedMetadata[M1, E]],
    *,
    ax: int,
) -> Array[M0, np.floating]:
    """Get the momentum of a all states."""
    momentum = _build.k(states.basis.metadata()[1], idx=ax)

    states = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, states))

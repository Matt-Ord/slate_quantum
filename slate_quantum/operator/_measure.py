from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from slate_core import Array, Basis, BasisMetadata, TupleMetadata, array, metadata
from slate_core.basis import TupleBasisLike, from_metadata
from slate_core.metadata import AxisDirections, SpacedLengthMetadata

from slate_quantum import state as _state
from slate_quantum.operator import _build
from slate_quantum.operator._build._position import scattering_operator
from slate_quantum.operator._operator import expectation, expectation_of_each

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_quantum.state._state import State, StateList


def potential_from_function[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
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
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    """Get the expectation of a generic potential of all states."""
    momentum = _build.potential_from_function(
        states.basis.metadata().children[1], fn=fn, wrapped=wrapped, offset=offset
    )

    normalized = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, normalized))  # type: ignore bad inference


def momentum_from_function[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, np.dtype[np.complex128]],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> float:
    """Get the expectation of a generic potential of a state."""
    momentum = _build.momentum_from_function(
        state.basis.metadata(), fn=fn, wrapped=wrapped, offset=offset
    )

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
    offset: float = 0,
    wrapped: bool = False,
) -> float:
    """Get the position of a state."""
    momentum = _build.x(
        state.basis.metadata(), axis=axis, offset=offset, wrapped=wrapped
    )

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def _get_fundamental_scatter[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](states: State[TupleBasisLike[tuple[M1, ...], E]], *, axis: int) -> complex:
    r"""Get the scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    n_dim = len(states.basis.metadata().children)
    n_k = tuple(1 if i == axis else 0 for i in range(n_dim))
    scatter = scattering_operator(states.basis.metadata(), n_k=n_k)

    states = _state.normalize(states)
    return expectation(scatter, states)


def periodic_x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](state: State[TupleBasisLike[tuple[M1, ...], E]], *, axis: int) -> float:
    r"""Get the periodic position coordinate of a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_fundamental_scatter(state, axis=axis)

    angle = np.angle(scatter)
    wrapped = np.mod(angle, (2 * np.pi))
    delta_x = metadata.volume.fundamental_stacked_delta_x(state.basis.metadata())
    return wrapped * (np.linalg.norm(delta_x[axis]).item() / (2 * np.pi))


def variance_x[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
) -> float:
    r"""Get the width of a Gaussian wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    x = periodic_x(state, axis=axis)

    offset = tuple(-x if i == axis else 0 for i in range(state.basis.metadata().n_dim))
    return potential_from_function(
        state,
        fn=lambda pos: pos[axis].astype(np.complex128) ** 2,
        wrapped=True,
        offset=offset,
    )


def coherent_width[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
) -> float:
    """Get the width of a Gaussian wavepacket."""
    variance = variance_x(state, axis=axis)
    return np.sqrt(2 * variance)


def all_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
    offset: float = 0,
    wrapped: bool = False,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    """Get the position of all states."""
    momentum = _build.x(
        states.basis.metadata().children[1], axis=axis, offset=offset, wrapped=wrapped
    )

    normalized = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, normalized))  # type: ignore bad inference


def _get_all_fundamental_scatter[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.complexfloating]]:
    r"""Get the scattering operator for a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    n_dim = len(states.basis.metadata().children[1].children)
    n_k = tuple(1 if i == axis else 0 for i in range(n_dim))
    scatter = scattering_operator(states.basis.metadata().children[1], n_k=n_k)

    normalized = _state.normalize_all(states)
    return expectation_of_each(scatter, normalized)  # type: ignore bad inference


def all_periodic_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    r"""Get the periodic position coordinate of a wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    scatter = _get_all_fundamental_scatter(states, axis=axis)

    angle = array.angle(scatter)
    wrapped = array.mod(angle, (2 * np.pi))
    delta_x = metadata.volume.fundamental_stacked_delta_x(
        states.basis.metadata().children[1]
    )
    np.array([]).astype(np.complex128)
    return (wrapped * (np.linalg.norm(delta_x[axis]).item() / (2 * np.pi))).as_type(
        np.float64
    )


def all_variance_x[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    r"""Get the width of a Gaussian wavepacket.

    For a gaussian wavepacket

    \ket{\psi} = A \exp{(-\frac{{(x - x_0)}^2}{2 \sigma_0} + ik_0(x-x_0))} \ket{x}

    the expectation is given by

    \braket{e^{iqx}} = e^{iq.x_0}\exp{(-\sigma_0^2q^2 / 4)}
    """
    basis = from_metadata(states.basis.metadata().children[0])
    data = np.array([variance_x(state, axis=axis) for state in states])
    return Array.build(basis, data).ok()


def all_coherent_width[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    """Get the width of a Gaussian wavepacket."""
    variance = all_variance_x(states, axis=axis)
    return array.sqrt((variance * 2).as_type(np.float64))


def k[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
) -> float:
    """Get the momentum of a state."""
    momentum = _build.k(state.basis.metadata(), axis=axis)

    state = _state.normalize(state)
    return np.real(expectation(momentum, state))


def all_k[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    """Get the momentum of all states."""
    momentum = _build.k(states.basis.metadata().children[1], axis=axis)

    normalized = _state.normalize_all(states)
    return array.real(expectation_of_each(momentum, normalized))  # type: ignore bad inference


def variance_k[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
) -> float:
    r"""Get the variance of the momentum of a state."""
    k_measured = k(state, axis=axis)

    offset = tuple(
        -k_measured if i == axis else 0 for i in range(state.basis.metadata().n_dim)
    )
    return momentum_from_function(
        state,
        fn=lambda pos: pos[axis].astype(np.complex128) ** 2,
        wrapped=True,
        offset=offset,
    )


def all_variance_k[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    r"""Get the variance of the momentum of all states.

    The variance in momentum is given by

    \braket{(K - \braket{K})^2}
    """
    basis = from_metadata(states.basis.metadata().children[0])
    data = np.array([variance_k(state, axis=axis) for state in states])
    return Array.build(basis, data).ok()


def uncertainty[
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    state: State[TupleBasisLike[tuple[M1, ...], E]],
    *,
    axis: int,
) -> float:
    r"""Get the uncertainty in position of a state."""
    x_variance = variance_x(state, axis=axis)
    k_variance = variance_k(state, axis=axis)
    return np.sqrt(x_variance * k_variance)


def all_uncertainty[
    M0: BasisMetadata,
    M1: SpacedLengthMetadata,
    E: AxisDirections,
](
    states: StateList[TupleBasisLike[tuple[M0, TupleMetadata[tuple[M1, ...], E]]]],
    *,
    axis: int,
) -> Array[Basis[M0], np.dtype[np.floating]]:
    r"""Get the uncertainty in position of all states."""
    basis = from_metadata(states.basis.metadata().children[0])
    data = np.array([uncertainty(state, axis=axis) for state in states])
    return Array.build(basis, data).ok()

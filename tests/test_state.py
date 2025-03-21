from __future__ import annotations

import numpy as np
from slate_core import basis
from slate_core.metadata.volume import spaced_volume_metadata_from_stacked_delta_x

from slate_quantum import state
from slate_quantum.state._state import State


def test_normalization_state() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    rng = np.random.default_rng()

    state_basis = basis.from_metadata(metadata)
    normalized_state = State.build(
        state_basis, np.exp(1j * 2 * np.pi * rng.random(5)) / np.sqrt(5)
    ).ok()
    np.testing.assert_allclose(state.normalization(normalized_state), 1, atol=1e-15)


def test_inner_product_state() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )
    rng = np.random.default_rng()

    state_basis = basis.from_metadata(metadata)
    normalized_state = State.build(
        state_basis, np.exp(1j * 2 * np.pi * rng.random(5)) / np.sqrt(5)
    ).ok()
    np.testing.assert_allclose(
        state.inner_product(normalized_state, normalized_state), 1, atol=1e-15
    )

    data = np.zeros(5, dtype=np.complex128)
    data[0] = +(1 - 1j) / np.sqrt(2)
    data[1] = +(1 - 1j) / np.sqrt(2)
    position_state_0 = State.build(state_basis, data).ok()

    data = np.zeros(5, dtype=np.complex128)
    data[0] = +(1 - 1j) / np.sqrt(2)
    data[1] = -(1 - 1j) / np.sqrt(2)
    position_state_1 = State.build(state_basis, data).ok()
    np.testing.assert_allclose(
        state.inner_product(position_state_1, position_state_0), 0, atol=1e-15
    )


def test_get_occupations() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (5,)
    )

    state_basis = basis.from_metadata(metadata)
    data = np.zeros(5, dtype=np.complex128)
    data[0] = (1 - 1j) / np.sqrt(2)
    data[1] = (1 - 1j) / np.sqrt(2)
    position_state = State.build(state_basis, data).ok()

    occupations = state.get_occupations(position_state)
    np.testing.assert_allclose(
        occupations.raw_data, np.array([1, 1, 0, 0, 0]), atol=1e-15
    )
    assert occupations.basis.metadata().basis == state_basis

    state_basis = basis.TransformedBasis(state_basis).resolve_ctype()
    momentum_state = State.build(state_basis, data).ok()

    occupations = state.get_occupations(momentum_state)
    np.testing.assert_allclose(
        occupations.raw_data, np.array([1, 1, 0, 0, 0]), atol=1e-15
    )
    assert occupations.basis.metadata().basis == state_basis

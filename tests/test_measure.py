from __future__ import annotations

import numpy as np
from slate.metadata import spaced_volume_metadata_from_stacked_delta_x, volume

from slate_quantum import operator, state


def test_measure_x_position_state() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (50,)
    )

    for i in range(metadata.fundamental_size):
        position_state = state.build.position(metadata, (i,))
        x = operator.measure.x(position_state, ax=0)
        assert x == i * volume.fundamental_stacked_dx(metadata)[0][0]


def test_measure_x_coherent_state() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (300,)
    )
    expected_x = 150 * volume.fundamental_stacked_dx(metadata)[0][0]
    sigma_0 = (0.1 * np.pi,)
    coherent_state = state.build.coherent(metadata, (expected_x,), (0,), sigma_0)

    x = operator.measure.x(coherent_state, ax=0)
    np.testing.assert_allclose(x, expected_x, rtol=1e-3)

    for i in range(metadata.fundamental_size):
        expected_x = i * volume.fundamental_stacked_dx(metadata)[0][0]
        coherent_state = state.build.coherent(metadata, (expected_x,), (0,), sigma_0)

        periodic_x = operator.measure.periodic_x(coherent_state, ax=0)
        difference = (periodic_x - expected_x + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(
            difference,
            0,
            atol=1e-7 * volume.fundamental_stacked_dx(metadata)[0][0],
        )
        wrapped_x = operator.measure.x(
            coherent_state, ax=0, offset=-expected_x, wrapped=True
        )
        np.testing.assert_allclose(
            wrapped_x,
            0,
            atol=1e-7 * volume.fundamental_stacked_dx(metadata)[0][0],
        )


def test_measure_width_coherent_state() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (300,)
    )

    x_0 = (150 * volume.fundamental_stacked_dx(metadata)[0][0],)

    for sigma_0 in (0.1 * np.pi, 0.01 * np.pi):
        coherent_state = state.build.coherent(metadata, x_0, (0,), (sigma_0,))

        variance = operator.measure.variance_x(coherent_state, ax=0)
        np.testing.assert_allclose(
            np.sqrt(2 * variance),
            sigma_0,
            atol=1e-7 * volume.fundamental_stacked_dx(metadata)[0][0],
        )


def test_measure_width_coherent_state_2() -> None:
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (300,)
    )
    expected_x = 150 * volume.fundamental_stacked_dx(metadata)[0][0]
    sigma_0 = (0.1 * np.pi,)
    coherent_state = state.build.coherent(metadata, (expected_x,), (0,), sigma_0)

    x = operator.measure.x(coherent_state, ax=0)
    np.testing.assert_allclose(x, expected_x, rtol=1e-3)

    x_squared = operator.measure.potential_from_function(
        coherent_state,
        fn=lambda pos: pos[0].astype(np.complex128) ** 2,
        wrapped=True,
        offset=(-expected_x,),
    )
    np.testing.assert_allclose(x_squared, sigma_0[0] ** 2 / 2, rtol=1e-3)

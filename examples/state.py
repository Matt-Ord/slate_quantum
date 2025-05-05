from __future__ import annotations

import numpy as np
import slate_core
from slate_core.metadata import spaced_volume_metadata_from_stacked_delta_x
from slate_core.plot import get_figure

from slate_quantum import state

if __name__ == "__main__":
    # In slate_quantum, a state is represented by the State class.
    # There are several methods that help to build states, such as position, momentum, coherent, and from_function.
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (600,)
    )

    fig, ax = get_figure()

    position_state = state.build.position(metadata, (200,))
    _, _, line = slate_core.plot.array_against_basis(position_state, ax=ax)
    line.set_label("Position Eigenstate")

    momentum_state = state.build.momentum(metadata, (20,))
    _, _, line = slate_core.plot.array_against_basis(momentum_state, ax=ax)
    line.set_label("Momentum Eigenstate")

    x_0 = 200 * 2 * np.pi / 500
    sigma_0 = 0.1 * np.pi
    coherent_state = state.build.coherent(metadata, (x_0,), (0,), (sigma_0,))
    _, _, line = slate_core.plot.array_against_basis(coherent_state, ax=ax)
    line.set_label("Coherent State")

    # Here we use the from_function method to build the same coherent state.
    function_state = state.build.from_function(
        metadata,
        lambda x: np.exp(-(((x[0] - x_0) / sigma_0) ** 2 / 2)),
    )
    _, _, line = slate_core.plot.array_against_basis(function_state, ax=ax)
    line.set_label("Coherent State from Function")
    line.set_linestyle("--")

    ax.set_title("Quantum States Visualization")
    ax.set_xlabel("Position /m")
    ax.set_ylabel(r"np.real($\psi(x)$)")
    ax.legend()
    fig.show()

    slate_core.plot.wait_for_close()

from __future__ import annotations

import numpy as np
from slate.metadata import spaced_volume_metadata_from_stacked_delta_x, volume

from slate_quantum import operator, state

if __name__ == "__main__":
    metadata = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (500,)
    )

    for i in range(metadata.fundamental_size):
        position_state = state.build.position(metadata, (i,))
        x = operator.measure.x(position_state, ax=0)
        assert x == i * volume.fundamental_stacked_dx(metadata)[0][0]

from __future__ import annotations

from typing import Any

import numpy as np


def outer_product(
    *arrays: np.ndarray[Any, np.dtype[np.complexfloating]],
) -> np.ndarray[Any, np.dtype[np.complexfloating]]:
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.prod(grids, axis=0)

from __future__ import annotations

from typing import Any

import numpy as np


def outer_product(
    *arrays: np.ndarray[Any, np.dtype[np.complex128]],
) -> np.ndarray[Any, np.dtype[np.complex128]]:
    grids = np.meshgrid(*arrays, indexing="ij")
    return np.prod(grids, axis=0)

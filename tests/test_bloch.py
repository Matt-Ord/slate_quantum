from __future__ import annotations

import numpy as np
from slate import Array

from slate_quantum.bloch._block_diagonal_basis import BlockDiagonalBasis


def test_block_basis() -> None:
    data = np.arange(4 * 6).reshape(4, 6)
    array = Array.from_array(data)

    block_basis = BlockDiagonalBasis(array.basis, (2, 2))

    in_block_basis = array.with_basis(block_basis)
    print(data)
    np.testing.assert_allclose(
        in_block_basis.raw_data,
        [0, 1, 6, 7, 14, 15, 20, 21],
    )
    np.testing.assert_allclose(
        in_block_basis.as_array(),
        [
            [0.0, 1.0, 0.0, 0.0],
            [6.0, 7.0, 0.0, 0.0],
            [0.0, 0.0, 14.0, 20.0],
            [0.0, 0.0, 15.0, 21.0],
        ],
    )

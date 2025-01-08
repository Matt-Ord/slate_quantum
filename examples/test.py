from __future__ import annotations

import numpy as np

A = (np.eye(2) * 2).reshape(2, 2, 1)
B = (np.eye(3) * 3).reshape(3, 3, 1)
print(np.concatenate([A, np.zeros((2, 3, 1)), np.ones((3, 2, 1)), B]))

filler = np.zeros((2, 3))

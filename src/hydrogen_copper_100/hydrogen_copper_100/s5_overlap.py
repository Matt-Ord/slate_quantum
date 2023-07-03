from __future__ import annotations

from typing import Literal

import numpy as np
from surface_potential_analysis.basis.basis import (
    FundamentalMomentumBasis3d,
)
from surface_potential_analysis.overlap.calculation import calculate_wavepacket_overlap
from surface_potential_analysis.overlap.overlap import save_overlap
from surface_potential_analysis.wavepacket.normalization import (
    calculate_normalization,
)
from surface_potential_analysis.wavepacket.wavepacket import (
    Wavepacket3dWith2dSamples,
)

from .s4_wavepacket import (
    load_normalized_copper_wavepacket_momentum,
)
from .surface_data import get_data_path

_CopperWavepacket = Wavepacket3dWith2dSamples[
    Literal[12],
    Literal[12],
    FundamentalMomentumBasis3d[Literal[24], Literal[24], Literal[250]],
]


def generate_fcc_wavepacket() -> _CopperWavepacket:
    return load_normalized_copper_wavepacket_momentum(0, (0, 0, 117), 0)


def generate_next_fcc_wavepacket() -> _CopperWavepacket:
    """
    Generate a wavepacket grid of a neighboring fcc wavefunction.

    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next fcc site
    """
    return load_normalized_copper_wavepacket_momentum(0, (24, 24, 117), 0)


def generate_hcp_wavepacket() -> _CopperWavepacket:
    return load_normalized_copper_wavepacket_momentum(1, (8, 8, 117), 0)


def generate_next_hcp_wavepacket() -> _CopperWavepacket:
    """
    Generate a wavepacket grid of a neighboring hcp wavefunction.

    This is just the original wavepacket shifted by -delta_x0,
    which we can achieve by rolling the wavepacket.

    Returns
    -------
    WavepacketGrid
        Wavepacket at the next hcp site
    """
    return load_normalized_copper_wavepacket_momentum(1, (32, 32, 117), 0)


def calculate_overlap_nickel() -> None:
    wavepacket_fcc = generate_fcc_wavepacket()
    print(calculate_normalization(wavepacket_fcc))  # noqa: T201

    wavepacket_hcp = generate_hcp_wavepacket()
    print(calculate_normalization(wavepacket_hcp))  # noqa: T201

    overlap_hcp_fcc = calculate_wavepacket_overlap(wavepacket_fcc, wavepacket_hcp)
    print(np.sum(overlap_hcp_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_fcc.npy")
    save_overlap(path, overlap_hcp_fcc)

    wavepacket_next_fcc = generate_next_fcc_wavepacket()

    overlap_fcc_fcc = calculate_wavepacket_overlap(wavepacket_fcc, wavepacket_next_fcc)
    print(np.sum(overlap_fcc_fcc["vector"]))  # noqa: T201
    path = get_data_path("overlap_fcc_fcc.npy")
    save_overlap(path, overlap_fcc_fcc)

    wavepacket_next_hcp = generate_next_hcp_wavepacket()

    overlap_hcp_hcp = calculate_wavepacket_overlap(wavepacket_hcp, wavepacket_next_hcp)
    print(np.sum(overlap_hcp_hcp["vector"]))  # noqa: T201
    path = get_data_path("overlap_hcp_hcp.npy")
    save_overlap(path, overlap_hcp_hcp)
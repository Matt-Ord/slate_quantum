from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import hamiltonian_generator
import numpy as np

from surface_potential_analysis.basis.basis import (
    ExplicitBasis,
    FundamentalBasis,
    FundamentalPositionBasis1d,
    FundamentalPositionBasis3d,
    FundamentalTransformedPositionBasis,
    TransformedPositionBasis,
)
from surface_potential_analysis.basis.stacked_basis import (
    StackedBasis,
    StackedBasisLike,
)
from surface_potential_analysis.basis.util import BasisUtil
from surface_potential_analysis.hamiltonian_builder.momentum_basis import (
    hamiltonian_from_mass_in_basis,
)
from surface_potential_analysis.operator.conversion import add_operator
from surface_potential_analysis.stacked_basis.potential_basis import (
    get_potential_basis_config_eigenstates,
)
from surface_potential_analysis.util.decorators import timed

if TYPE_CHECKING:
    from surface_potential_analysis.operator import SingleBasisOperator
    from surface_potential_analysis.potential.potential import Potential
    from surface_potential_analysis.stacked_basis.potential_basis import (
        PotentialBasisConfig,
    )
    from surface_potential_analysis.state_vector.eigenstate_collection import (
        EigenstateList,
    )

_N0Inv = TypeVar("_N0Inv", bound=int)
_N1Inv = TypeVar("_N1Inv", bound=int)
_N2Inv = TypeVar("_N2Inv", bound=int)
_NF0Inv = TypeVar("_NF0Inv", bound=int)
_NF1Inv = TypeVar("_NF1Inv", bound=int)
_NF2Inv = TypeVar("_NF2Inv", bound=int)
_B0Inv = TypeVar("_B0Inv", bound=StackedBasisLike[*tuple[Any, ...]])


class PotentialSizeError(Exception):
    """Error thrown when the potential is too small."""

    def __init__(self, axis: int, required: int, actual: int) -> None:
        super().__init__(
            f"Potential does not have enough resolution in x{axis} direction"
            f"required {required} actual {actual}"
        )


def _get_xy_hamiltonian(
    basis: _B0Inv,
    mass: float,
    bloch_fraction: np.ndarray[tuple[Literal[2]], np.dtype[np.float64]],
) -> SingleBasisOperator[_B0Inv]:
    xy_basis = StackedBasis[Any, Any](
        TransformedPositionBasis(basis[0].delta_x[:2], basis[0].n, basis[0].n),
        TransformedPositionBasis(basis[1].delta_x[:2], basis[1].n, basis[1].n),
    )

    xy_hamiltonian = hamiltonian_from_mass_in_basis(xy_basis, mass, bloch_fraction)

    xy_energies = np.diag(
        np.broadcast_to(
            np.diag(xy_hamiltonian["data"].reshape(xy_hamiltonian["basis"].shape))[
                :, np.newaxis
            ],
            (basis[0].n * basis[1].n, basis[2].n),
        ).reshape(-1)
    ).reshape(-1)

    return {"data": xy_energies, "basis": StackedBasis(basis, basis)}


class _SurfaceHamiltonianUtil(
    Generic[_N0Inv, _N1Inv, _N2Inv, _NF0Inv, _NF1Inv, _NF2Inv]
):
    _potential: Potential[
        StackedBasisLike[
            FundamentalPositionBasis3d[_NF0Inv],
            FundamentalPositionBasis3d[_NF1Inv],
            FundamentalPositionBasis3d[_NF2Inv],
        ]
    ]

    _resolution: tuple[_N0Inv, _N1Inv]
    _config: PotentialBasisConfig[FundamentalPositionBasis1d[_NF2Inv], _N2Inv]

    def __init__(
        self,
        potential: Potential[
            StackedBasisLike[
                FundamentalPositionBasis3d[_NF0Inv],
                FundamentalPositionBasis3d[_NF1Inv],
                FundamentalPositionBasis3d[_NF2Inv],
            ]
        ],
        resolution: tuple[_N0Inv, _N1Inv],
        config: PotentialBasisConfig[FundamentalPositionBasis1d[_NF2Inv], _N2Inv],
        bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    ) -> None:
        self._potential = potential
        self._resolution = resolution
        self._bloch_fraction = bloch_fraction
        self._config = config
        if 2 * (self._resolution[0] - 1) > self._potential["basis"][0].n:
            raise PotentialSizeError(
                0, 2 * (self._resolution[0] - 1), self._potential["basis"][0].n
            )

        if 2 * (self._resolution[1] - 1) > self._potential["basis"][1].n:
            raise PotentialSizeError(
                1, 2 * (self._resolution[1] - 1), self._potential["basis"][1].n
            )

    def basis(
        self,
        state_vectors_z: EigenstateList[
            FundamentalBasis[_N2Inv],
            StackedBasisLike[*tuple[FundamentalPositionBasis1d[_NF2Inv]]],
        ],
    ) -> StackedBasisLike[
        TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[3]],
        TransformedPositionBasis[_NF1Inv, _N1Inv, Literal[3]],
        ExplicitBasis[_NF2Inv, _N2Inv, Literal[3]],
    ]:
        return StackedBasis(
            TransformedPositionBasis(
                self._potential["basis"][0].delta_x,
                self._resolution[0],
                self._potential["basis"][0].n,
            ),
            TransformedPositionBasis(
                self._potential["basis"][1].delta_x,
                self._resolution[1],
                self._potential["basis"][1].n,
            ),
            ExplicitBasis(
                self._potential["basis"][2].delta_x,
                state_vectors_z["data"].reshape(state_vectors_z["basis"].shape),  # type: ignore[arg-type]
            ),
        )

    @property
    def points(
        self,
    ) -> np.ndarray[tuple[_NF0Inv, _NF1Inv, _NF2Inv], np.dtype[np.complex128]]:
        return self._potential["data"].reshape(  # type: ignore[no-any-return]
            BasisUtil(self._potential["basis"]).shape
        )

    def bloch_z_hamiltonian(
        self, bloch_fraction: float
    ) -> SingleBasisOperator[
        StackedBasisLike[
            TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[3]],
            TransformedPositionBasis[_NF1Inv, _N1Inv, Literal[3]],
            ExplicitBasis[_NF2Inv, _N2Inv, Literal[3]],
        ]
    ]:
        state_vectors_z = get_potential_basis_config_eigenstates(
            self._config, bloch_fraction=bloch_fraction
        )
        basis = self.basis(state_vectors_z)
        diagonal_energies = np.diag(
            np.broadcast_to(
                state_vectors_z["eigenvalue"].reshape(1, -1),
                (basis[0].n * basis[1].n, basis[2].n),
            ).reshape(-1)
        )
        other_energies = self._calculate_off_diagonal_energies(state_vectors_z)

        energies = diagonal_energies + other_energies

        return {"data": energies.reshape(-1), "basis": StackedBasis(basis, basis)}

    def hamiltonian(
        self,
    ) -> SingleBasisOperator[
        StackedBasisLike[
            TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[3]],
            TransformedPositionBasis[_NF1Inv, _N1Inv, Literal[3]],
            ExplicitBasis[_NF2Inv, _N2Inv, Literal[3]],
        ],
    ]:
        z_hamiltonian = self.bloch_z_hamiltonian(self._bloch_fraction.item(2))
        xy_hamiltonian = _get_xy_hamiltonian(
            z_hamiltonian["basis"][0], self._config["mass"], self._bloch_fraction[:2]
        )
        return add_operator(xy_hamiltonian, z_hamiltonian)

    def _calculate_off_diagonal_energies(
        self,
        state_vectors_z: EigenstateList[
            FundamentalBasis[_N2Inv],
            StackedBasisLike[*tuple[FundamentalPositionBasis1d[_NF2Inv]]],
        ],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
        basis = self.basis(state_vectors_z)
        return np.array(  # type: ignore[no-any-return]
            hamiltonian_generator.calculate_off_diagonal_energies2(
                self.get_ft_potential().tolist(),
                BasisUtil(basis[2]).vectors.tolist(),
                basis.shape,  # type: ignore[arg-type]
            )
        )

    @property
    def subtracted_points(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
        return np.subtract(self.points, self._config["potential"]["data"][np.newaxis, np.newaxis, :])  # type: ignore[no-any-return]

    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        return np.fft.ifft2(self.subtracted_points, axes=(0, 1))  # type: ignore[no-any-return]


@timed
def total_surface_hamiltonian(
    potential: Potential[
        StackedBasisLike[
            FundamentalPositionBasis3d[_NF0Inv],
            FundamentalPositionBasis3d[_NF1Inv],
            FundamentalPositionBasis3d[_NF2Inv],
        ]
    ],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    resolution: tuple[_N0Inv, _N1Inv],
    config: PotentialBasisConfig[FundamentalPositionBasis1d[_NF2Inv], _N2Inv],
) -> SingleBasisOperator[
    StackedBasisLike[
        TransformedPositionBasis[_NF0Inv, _N0Inv, Literal[3]],
        TransformedPositionBasis[_NF1Inv, _N1Inv, Literal[3]],
        ExplicitBasis[_NF2Inv, _N2Inv, Literal[3]],
    ]
]:
    """
    Calculate a hamiltonian using the given basis.

    Parameters
    ----------
    potential : Potential[_L0, _L1, _L2]
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    basis : StackedBasisLike[tuple[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    mass : float

    Returns
    -------
    HamiltonianWithBasis[TruncatedBasis[_L3, MomentumBasis[_L0]], TruncatedBasis[_L4, MomentumBasis[_L1]], ExplicitBasis[_L5, MomentumBasis[_L2]]]
    """
    util = _SurfaceHamiltonianUtil(potential, resolution, config, bloch_fraction)
    return util.hamiltonian()


def total_surface_hamiltonian_as_fundamental(
    potential: Potential[
        StackedBasisLike[
            FundamentalPositionBasis3d[_NF0Inv],
            FundamentalPositionBasis3d[_NF1Inv],
            FundamentalPositionBasis3d[_NF2Inv],
        ]
    ],
    bloch_fraction: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]],
    resolution: tuple[_N0Inv, _N1Inv],
    config: PotentialBasisConfig[FundamentalPositionBasis1d[_NF2Inv], _N2Inv],
) -> SingleBasisOperator[
    StackedBasisLike[
        FundamentalTransformedPositionBasis[_N0Inv, Literal[3]],
        FundamentalTransformedPositionBasis[_N1Inv, Literal[3]],
        ExplicitBasis[_NF2Inv, _N2Inv, Literal[3]],
    ]
]:
    """
    calculate the hamiltonian, and ignore the true shape in the xy direction.

    Parameters
    ----------
    potential : FundamentalPositionBasisPotential3d[_NF0Inv, _NF1Inv, _NF2Inv]
    bloch_fraction : np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    resolution : tuple[_N0Inv, _N1Inv]
    config : PotentialBasisConfig[tuple[FundamentalPositionBasis1d[_NF2Inv]], _N2Inv]

    Returns
    -------
    SingleBasisOperator[
        tuple[ MomentumBasis3d[_NF0Inv, _NF0Inv], MomentumBasis3d[_NF1Inv, _NF1Inv], ExplicitBasis3d[_NF2Inv, _N2Inv], ]
    """
    hamiltonian = total_surface_hamiltonian(
        potential, bloch_fraction, resolution, config
    )
    new_basis = StackedBasis(
        FundamentalTransformedPositionBasis(
            hamiltonian["basis"][0][0].delta_x, hamiltonian["basis"][0][0].n
        ),
        FundamentalTransformedPositionBasis(
            hamiltonian["basis"][0][1].delta_x, hamiltonian["basis"][0][1].n
        ),
        hamiltonian["basis"][0][2],
    )
    return {"basis": StackedBasis(new_basis, new_basis), "data": hamiltonian["data"]}

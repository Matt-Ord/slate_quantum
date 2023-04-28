from typing import Any, Generic, TypedDict, TypeVar, overload

import numpy as np

from surface_potential_analysis.basis import (
    Basis,
    ExplicitBasis,
    FundamentalBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis.basis import BasisUtil, as_fundamental_basis
from surface_potential_analysis.basis_config.basis_config import (
    BasisConfig,
    BasisConfigUtil,
)
from surface_potential_analysis.basis_config.conversion import convert_vector
from surface_potential_analysis.eigenstate.eigenstate import (
    Eigenstate,
    EigenstateWithBasis,
    MomentumBasisEigenstate,
    PositionBasisEigenstate,
)
from surface_potential_analysis.interpolation import pad_ft_points

_BC0Cov = TypeVar("_BC0Cov", bound=BasisConfig[Any, Any, Any], covariant=True)
_BC0Inv = TypeVar("_BC0Inv", bound=BasisConfig[Any, Any, Any])
_BC1Inv = TypeVar("_BC1Inv", bound=BasisConfig[Any, Any, Any])

_BX0Cov = TypeVar("_BX0Cov", bound=Basis[Any, Any], covariant=True)
_BX1Cov = TypeVar("_BX1Cov", bound=Basis[Any, Any], covariant=True)
_BX2Cov = TypeVar("_BX2Cov", bound=Basis[Any, Any], covariant=True)

_BX0Inv = TypeVar("_BX0Inv", bound=Basis[Any, Any])
_BX1Inv = TypeVar("_BX1Inv", bound=Basis[Any, Any])
_BX2Inv = TypeVar("_BX2Inv", bound=Basis[Any, Any])

_L0Inv = TypeVar("_L0Inv", bound=int, covariant=True)
_L1Inv = TypeVar("_L1Inv", bound=int, covariant=True)
_L2Inv = TypeVar("_L2Inv", bound=int, covariant=True)

_LF0Inv = TypeVar("_LF0Inv", bound=int)
_LF1Inv = TypeVar("_LF1Inv", bound=int)
_LF2Inv = TypeVar("_LF2Inv", bound=int)


class _StackedEigenstate(TypedDict, Generic[_BC0Cov]):
    basis: _BC0Cov
    vector: np.ndarray[tuple[int, int, int], np.dtype[np.complex_]]


StackedEigenstateWithBasis = _StackedEigenstate[BasisConfig[_BX0Cov, _BX1Cov, _BX2Cov]]


def _stack_eigenstate(state: Eigenstate[_BC0Inv]) -> _StackedEigenstate[_BC0Inv]:
    util = BasisConfigUtil[Any, Any, Any](state["basis"])
    return {"basis": state["basis"], "vector": state["vector"].reshape(util.shape)}


def _flatten_eigenstate(state: _StackedEigenstate[_BC0Inv]) -> Eigenstate[_BC0Inv]:
    return {"basis": state["basis"], "vector": state["vector"].reshape(-1)}


def convert_eigenstate_to_basis(
    eigenstate: Eigenstate[_BC0Inv], basis: _BC1Inv
) -> Eigenstate[_BC1Inv]:
    """
    Given an eigenstate, calculate the vector in the given basis.

    Parameters
    ----------
    eigenstate : Eigenstate[_BC0Inv]
    basis : _BC1Inv

    Returns
    -------
    Eigenstate[_BC1Inv]
    """
    converted = convert_vector(eigenstate["vector"], eigenstate["basis"], basis)
    return {"basis": basis, "vector": converted}


def convert_position_basis_eigenstate_to_momentum_basis(
    eigenstate: PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    """
    convert an eigenstate from position to momentum basis.

    Parameters
    ----------
    eigenstate : PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
    transformed = np.fft.fftn(
        eigenstate["vector"].reshape(util.shape),
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": (
            {
                "_type": "momentum",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "momentum",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": transformed.reshape(-1),
    }


def convert_momentum_basis_eigenstate_to_position_basis(
    eigenstate: MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
) -> PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]:
    """
    Convert an eigenstate from momentum to position basis.

    Parameters
    ----------
    eigenstate : MomentumBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]

    Returns
    -------
    PositionBasisEigenstate[_L0Inv, _L1Inv, _L2Inv]
    """
    util = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"].reshape(util.shape),
        s=util.fundamental_shape,
        axes=(0, 1, 2),
    )
    transformed = np.fft.ifftn(
        padded,
        axes=(0, 1, 2),
        s=util.fundamental_shape,
        norm="ortho",
    )
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x2,
                "n": util.fundamental_n2,  # type: ignore[typeddict-item]
            },
        ),
        "vector": transformed.reshape(-1),
    }


def _convert_momentum_basis_x01_to_position(
    eigenstate: StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | StackedEigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | StackedEigenstateWithBasis[
        TruncatedBasis[Any, MomentumBasis[_L0Inv]],
        TruncatedBasis[Any, MomentumBasis[_L1Inv]],
        _BX0Inv,
    ]
) -> StackedEigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    util = BasisConfigUtil(eigenstate["basis"])
    padded = pad_ft_points(
        eigenstate["vector"],  # type: ignore[arg-type]
        s=[util.fundamental_n0, util.fundamental_n1],
        axes=(0, 1),
    )
    transformed = np.fft.ifftn(padded, axes=(0, 1), norm="ortho")
    return {
        "basis": (
            {
                "_type": "position",
                "delta_x": util.delta_x0,
                "n": util.fundamental_n0,  # type: ignore[typeddict-item]
            },
            {
                "_type": "position",
                "delta_x": util.delta_x1,
                "n": util.fundamental_n1,  # type: ignore[typeddict-item]
            },
            eigenstate["basis"][2],
        ),
        "vector": transformed,
    }


def _convert_position_basis_x2_to_momentum(
    eigenstate: StackedEigenstateWithBasis[
        _BX0Inv,
        _BX1Inv,
        TruncatedBasis[_L0Inv, PositionBasis[_LF0Inv]],
    ]
    | StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, PositionBasis[_LF0Inv]],
) -> StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, MomentumBasis[_LF0Inv]]:
    basis = BasisUtil(eigenstate["basis"][2])
    transformed = np.fft.fftn(
        eigenstate["vector"], axes=(2,), s=(basis.fundamental_n,), norm="ortho"  # type: ignore[misc]
    )
    return {
        "basis": (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            {"_type": "momentum", "delta_x": basis.delta_x, "n": basis.fundamental_n},  # type: ignore[misc,typeddict-item]
        ),
        "vector": transformed,
    }


_PInv = TypeVar("_PInv", bound=FundamentalBasis[Any])


def _convert_explicit_basis_x2_to_position(
    eigenstate: StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, ExplicitBasis[Any, _PInv]]
) -> StackedEigenstateWithBasis[_BX0Inv, _BX1Inv, _PInv]:
    vector = np.sum(
        eigenstate["vector"][:, :, :, np.newaxis]
        * eigenstate["basis"][2]["vectors"][np.newaxis, np.newaxis, :, :],
        axis=2,
    )
    return {
        "basis": (
            eigenstate["basis"][0],
            eigenstate["basis"][1],
            eigenstate["basis"][2]["parent"],
        ),
        "vector": vector,
    }


def convert_sho_eigenstate_to_position_basis(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_LF0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        MomentumBasis[_LF0Inv],
        MomentumBasis[_LF1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
) -> PositionBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]:
    """
    Given an eigenstate in sho basis convert it into position basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]]  |  MomentumBasis[_LF0Inv], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]]  |  MomentumBasis[_LF1Inv], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ]

    Returns
    -------
    PositionBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]
    """
    stacked = _stack_eigenstate(eigenstate)
    xy_converted = _convert_momentum_basis_x01_to_position(stacked)
    converted = _convert_explicit_basis_x2_to_position(xy_converted)
    return _flatten_eigenstate(converted)


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


@overload
def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    ...


def convert_sho_eigenstate_to_fundamental_xy(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
    | EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv]:
    """
    Given a truncated basis in xy, convert to a funadamental momentum basis of lower resolution.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[Any, MomentumBasis[_L0Inv]]  |  MomentumBasis[_L0Inv], TruncatedBasis[Any, MomentumBasis[_L1Inv]]  |  MomentumBasis[_L1Inv], _BX0Inv, ]

    Returns
    -------
    EigenstateWithBasis[ TruncatedBasis[Any, MomentumBasis[_L0Inv]] | MomentumBasis[_L0Inv], TruncatedBasis[Any, MomentumBasis[_L1Inv]] | MomentumBasis[_L1Inv], _BX0Inv, ]
    """
    return {
        "basis": (
            as_fundamental_basis(eigenstate["basis"][0]),
            as_fundamental_basis(eigenstate["basis"][1]),
            eigenstate["basis"][2],
        ),
        "vector": eigenstate["vector"],
    }


def convert_sho_eigenstate_to_momentum_basis(
    eigenstate: EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]] | MomentumBasis[_L0Inv],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]] | MomentumBasis[_L1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]],
        TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
    | EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]],
    ]
) -> MomentumBasisEigenstate[_L0Inv, _L1Inv, _LF2Inv]:
    """
    Convert a sho eigenstate to momentum basis.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ TruncatedBasis[_L0Inv, MomentumBasis[_LF0Inv]]  |  MomentumBasis[_LF0Inv], TruncatedBasis[_L1Inv, MomentumBasis[_LF1Inv]]  |  MomentumBasis[_LF1Inv], ExplicitBasis[_L2Inv, PositionBasis[_LF2Inv]], ]

    Returns
    -------
    MomentumBasisEigenstate[_LF0Inv, _LF1Inv, _LF2Inv]
    """
    stacked = _stack_eigenstate(eigenstate)
    x2_position = _convert_explicit_basis_x2_to_position(stacked)
    x2_momentum = _convert_position_basis_x2_to_momentum(x2_position)
    flattened = _flatten_eigenstate(x2_momentum)
    return convert_sho_eigenstate_to_fundamental_xy(flattened)


def convert_eigenstate_01_axis_to_position_basis(
    eigenstate: EigenstateWithBasis[
        MomentumBasis[_L0Inv],
        MomentumBasis[_L1Inv],
        _BX0Inv,
    ]
) -> EigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]:
    """
    convert eigenstate to position basis in the x0 and x1 direction.

    Parameters
    ----------
    eigenstate : EigenstateWithBasis[ MomentumBasis[_L0Inv], MomentumBasis[_L1Inv], _BX0Inv, ]

    Returns
    -------
    EigenstateWithBasis[PositionBasis[_L0Inv], PositionBasis[_L1Inv], _BX0Inv]
    """
    stacked = _stack_eigenstate(eigenstate)
    converted = _convert_momentum_basis_x01_to_position(stacked)
    return _flatten_eigenstate(converted)

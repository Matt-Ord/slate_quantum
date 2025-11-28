from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, cast

import numpy as np
from slate_core import (
    AsUpcast,
    BasisMetadata,
    Ctype,
    FundamentalBasis,
    TransformedBasis,
    TupleBasis,
    TupleMetadata,
    basis,
)
from slate_core import metadata as _metadata
from slate_core.basis import (
    CroppedBasis,
    DiagonalBasis,
    TruncatedBasis,
    Truncation,
    TupleBasisLike,
)
from slate_core.metadata import (
    SIMPLE_FEATURE,
    AxisDirections,
    EvenlySpacedLengthMetadata,
    EvenlySpacedVolumeMetadata,
    LengthMetadata,
    VolumeMetadata,
)
from slate_core.util import recast_along_axes

from slate_quantum._util import outer_product
from slate_quantum.metadata import RepeatedMetadata, repeat_volume_metadata
from slate_quantum.operator._diagonal import (
    PositionOperatorBasis,
    Potential,
    position_operator_basis,
    with_outer_basis,
)
from slate_quantum.operator._operator import Operator, OperatorBasis, operator_basis

if TYPE_CHECKING:
    from collections.abc import Callable


def potential[M: BasisMetadata, E, CT: Ctype[Never], DT: np.dtype[np.generic]](
    outer_basis: TupleBasisLike[tuple[M, ...], E, CT], data: np.ndarray[Any, DT]
) -> Operator[PositionOperatorBasis[M, E], DT]:
    """Get the potential operator."""
    return Operator(position_operator_basis(outer_basis), data)


_build_potential = potential


def repeat_potential(
    potential: Potential[
        EvenlySpacedLengthMetadata,
        AxisDirections,
        Ctype[Never],
        np.dtype[np.complexfloating],
    ],
    shape: tuple[int, ...],
) -> Potential[RepeatedMetadata, AxisDirections]:
    """Create a new potential by repeating the original potential in each direction."""
    transformed_basis = basis.transformed_from_metadata(
        potential.basis.inner.outer_recast.metadata()
    )
    as_transformed = with_outer_basis(potential, transformed_basis)
    converted_basis = basis.transformed_from_metadata(
        repeat_volume_metadata(potential.basis.inner.outer_recast.metadata(), shape)
    )
    repeat_basis = basis.with_modified_children(
        converted_basis,
        lambda i, y: TruncatedBasis(
            Truncation(transformed_basis.children[i].size, shape[i], 0),
            y,
        ).upcast(),
    ).upcast()
    return _build_potential(
        repeat_basis,
        as_transformed.raw_data * np.sqrt(np.prod(shape)),
    )


def cos_potential(
    metadata: EvenlySpacedVolumeMetadata,
    height: float,
) -> Potential[EvenlySpacedLengthMetadata, AxisDirections]:
    """Build a cosine potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    data = outer_product(*(np.array([2, 1, 1], dtype=np.complex128),) * n_dim)
    return potential(
        cropped, 0.25**n_dim * height * data * np.sqrt(transformed_basis.size)
    )


def sin_potential(
    metadata: EvenlySpacedVolumeMetadata,
    height: float,
) -> Potential[EvenlySpacedLengthMetadata, AxisDirections]:
    """Build a cosine potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    data = outer_product(*(np.array([2, 1j, -1j]),) * n_dim)
    return potential(
        cropped, 0.25**n_dim * height * data * np.sqrt(transformed_basis.size)
    )


def _square_wave_points(
    n_terms: int, lanczos_factor: float
) -> np.ndarray[tuple[int], np.dtype[np.complex128]]:
    prefactor = np.sinc(2 * np.fft.fftfreq(n_terms)) ** lanczos_factor
    return prefactor * np.fft.ifft(
        np.sign(np.cos(np.linspace(0, 2 * np.pi, n_terms, endpoint=False)))
    )  # type: ignore can't infer shape of ifft


def square_potential(
    metadata: EvenlySpacedVolumeMetadata,
    height: float,
    *,
    n_terms: tuple[int, ...] | None = None,
    lanczos_factor: float = 0,
) -> Potential[EvenlySpacedLengthMetadata, AxisDirections]:
    """Build a square potential."""
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda i, y: y if n_terms is None else CroppedBasis(n_terms[i], y).upcast(),
    ).upcast()

    data = outer_product(
        *(_square_wave_points(n, lanczos_factor) for n in cropped.inner.shape)
    )
    return potential(
        cropped,
        0.5 * height * data * np.sqrt(transformed_basis.size),
    )


def fcc_potential(
    metadata: EvenlySpacedVolumeMetadata,
    height: float,
) -> Potential[EvenlySpacedLengthMetadata, AxisDirections]:
    """Generate a potential suitable for modelling an fcc surface.

    This potential contains the lowest fourier components - however for an fcc surface
    there are only six degenerate fourier components.
    """
    transformed_basis = basis.transformed_from_metadata(metadata)
    # We need only the three lowest fourier components to represent this potential
    cropped = basis.with_modified_children(
        transformed_basis,
        lambda _i, y: CroppedBasis(3, y).upcast(),
    ).upcast()
    n_dim = len(cropped.inner.shape)
    # TODO: generalize to n_dim  # noqa: FIX002
    assert n_dim == 2  # noqa: PLR2004

    data = np.array([[3, 1, 1], [1, 1, 0], [1, 0, 1]]).astype(np.complex128)
    return potential(
        cropped,
        (1 / 3) ** n_dim * data * height * np.sqrt(transformed_basis.size),
    )


def potential_from_function[
    M: TupleMetadata[tuple[LengthMetadata, ...], AxisDirections],
    DT: np.dtype[np.number],
](
    metadata: M,
    fn: Callable[
        [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
        np.ndarray[Any, DT],
    ],
    *,
    wrapped: bool = False,
    offset: tuple[float, ...] | None = None,
) -> Operator[OperatorBasis[M], DT]:
    """Get the potential operator."""
    positions = _metadata.volume.fundamental_stacked_x_points(
        metadata, offset=offset, wrapped=wrapped
    )
    points = fn(positions).ravel()
    if SIMPLE_FEATURE not in metadata.features:
        weights = metadata.basis_weights
        points /= cast("np.ndarray[Any, DT]", np.square(weights).astype(points.dtype))

    return Operator(
        DiagonalBasis(
            operator_basis(AsUpcast(basis.from_metadata(metadata), metadata))
        ).upcast(),
        cast("np.ndarray[Any, DT]", points),
    )


def harmonic_potential[M: VolumeMetadata](
    metadata: M,
    frequency: float,
    *,
    offset: tuple[float, ...] | None = None,
) -> Operator[OperatorBasis[M], np.dtype[np.complexfloating]]:
    """Build a harmonic potential.

    V(x) = 0.5 * frequency^2 * ||x||^2
    """
    return potential_from_function(
        metadata,
        lambda x: (0.5 * frequency**2 * np.linalg.norm(x, axis=0) ** 2),
        wrapped=True,
        offset=offset,
    )


@dataclass(frozen=True, kw_only=True)
class MorseParameters:
    """Parameters for the Morse potential."""

    depth: float
    height: float
    offset: float = 0


def _morse_potential_fn(
    points: np.ndarray[Any, np.dtype[np.floating]],
    params: MorseParameters,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """
    Evaluate Morse potential.

    The morse potential is given by
    .. math::
        V(x) = D_e (e^{-2(x - x_e)/a} - 2 e^{-(x - x_e)/a})

    where :math:`D_e` is the depth, :math:`x_e` is the origin, and :math:`a`
    is the height parameter.
    """
    points = np.copy(points)
    points -= params.offset
    return params.depth * (
        np.exp(-2 * (points) / params.height) - 2 * np.exp(-(points) / params.height)
    )


@dataclass(frozen=True, kw_only=True)
class CorrugatedMorseParameters(MorseParameters):
    """Parameters for the corrugated Morse potential."""

    beta: float = 0

    def with_beta(
        self, beta: float | Callable[[float], float]
    ) -> CorrugatedMorseParameters:
        """Return a new instance with the given beta."""
        return CorrugatedMorseParameters(
            depth=self.depth,
            height=self.height,
            offset=self.offset,
            beta=beta(self.beta) if callable(beta) else beta,
        )

    def with_depth(
        self, depth: float | Callable[[float], float]
    ) -> CorrugatedMorseParameters:
        """Return a new instance with the given depth."""
        return CorrugatedMorseParameters(
            depth=depth(self.depth) if callable(depth) else depth,
            height=self.height,
            offset=self.offset,
            beta=self.beta,
        )

    def with_height(
        self, height: float | Callable[[float], float]
    ) -> CorrugatedMorseParameters:
        """Return a new instance with the given height."""
        return CorrugatedMorseParameters(
            depth=self.depth,
            height=height(self.height) if callable(height) else height,
            offset=self.offset,
            beta=self.beta,
        )

    def with_offset(
        self, offset: float | Callable[[float], float]
    ) -> CorrugatedMorseParameters:
        """Return a new instance with the given offset."""
        return CorrugatedMorseParameters(
            depth=self.depth,
            height=self.height,
            offset=offset(self.offset) if callable(offset) else offset,
            beta=self.beta,
        )


def _morse_corrugation_potential_fn(
    points: np.ndarray[Any, np.dtype[np.floating]],
    params: CorrugatedMorseParameters,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    r"""
    Evaluate Morse potential.

    The morse corrugation potential is given by
    .. math::
        V(x) = - D_e \beta e^{-2(x - x_e)/a}

    where :math:`D_e` is the depth, :math:`x_e` is the origin, :math:`a`
    is the height parameter and :math:`\beta` is the corrugation factor.
    """
    points = np.copy(points)
    points -= params.offset
    return -params.depth * params.beta * np.exp(-2 * points / params.height)


def corrugated_morse_potential_function(
    metadata: VolumeMetadata,
    params: CorrugatedMorseParameters,
    *,
    axis: int = -1,
) -> Callable[
    [tuple[np.ndarray[Any, np.dtype[np.floating]], ...]],
    np.ndarray[Any, np.dtype[np.complex128]],
]:
    """Get the corrugated Morse potential operator."""
    axis %= metadata.n_dim

    def _fn(
        x: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        """Evaluate the corrugated Morse potential."""
        base = _morse_potential_fn(x[axis], params)
        corrugation = _morse_corrugation_potential_fn(x[axis], params)

        corrugation_xy = np.sum(
            [
                np.cos(2 * np.pi * x[i] / metadata.children[i].delta)
                for i in range(metadata.n_dim)
                if i != axis
            ],
            axis=0,
        )
        return base + 2 * corrugation * corrugation_xy

    return _fn


def morse_potential[M: VolumeMetadata](
    metadata: M,
    params: MorseParameters,
    *,
    axis: int = -1,
) -> Operator[OperatorBasis[M], np.dtype[np.complexfloating]]:
    """Build a Morse potential.

    The morse potential is given by
    .. math::
        V(x) = D_e (e^{-2(x - x_e)/a} - 2 e^{-(x - x_e)/a})

    where :math:`D_e` is the depth, :math:`x_e` is the origin, and :math:`a`
    is the height parameter.
    """
    axis %= metadata.n_dim
    if not all(
        SIMPLE_FEATURE in c.features
        for i, c in enumerate(metadata.children)
        if i != axis
    ):
        return cast(
            "Operator[OperatorBasis[M], np.dtype[np.complexfloating]]",
            potential_from_function(
                metadata,
                corrugated_morse_potential_function(
                    metadata,
                    CorrugatedMorseParameters(
                        depth=params.depth,
                        height=params.height,
                        offset=params.offset,
                    ),
                    axis=axis,
                ),
            ),
        )
    basis_children = TupleBasis(
        tuple(
            (
                CroppedBasis(1, TransformedBasis(FundamentalBasis(c)))
                if i != axis
                else FundamentalBasis(c)
            )
            for i, c in enumerate(metadata.children)
        ),
        metadata.extra,
    )
    out_basis = position_operator_basis(basis_children.upcast())
    out_basis = AsUpcast(out_basis, TupleMetadata((metadata, metadata)))
    data = _morse_potential_fn(metadata.children[axis].values, params) * np.sqrt(
        basis_children.fundamental_size / basis_children.children[axis].fundamental_size
    )

    if SIMPLE_FEATURE not in metadata.children[axis].features:
        data /= np.square(metadata.children[axis].basis_weights)
    return Operator(out_basis, data.astype(np.complex128))


def corrugated_morse_potential[M: VolumeMetadata](
    metadata: M,
    params: CorrugatedMorseParameters,
    *,
    axis: int = -1,
) -> Operator[OperatorBasis[M], np.dtype[np.complexfloating]]:
    """Build a Morse potential.

    The morse potential is given by
    .. math::
        V(x) = D_e (e^{-2(x - x_e)/a} - 2 e^{-(x - x_e)/a})

    where :math:`D_e` is the depth, :math:`x_e` is the origin, and :math:`a`
    is the height parameter.
    """
    axis %= metadata.n_dim
    if not all(
        SIMPLE_FEATURE in c.features
        for i, c in enumerate(metadata.children)
        if i != axis
    ):
        return cast(
            "Operator[OperatorBasis[M], np.dtype[np.complexfloating]]",
            potential_from_function(
                metadata,
                corrugated_morse_potential_function(metadata, params, axis=axis),
            ),
        )

    basis_children = TupleBasis(
        tuple(
            (
                CroppedBasis(3, TransformedBasis(FundamentalBasis(c)))
                if i != axis
                else FundamentalBasis(c)
            )
            for i, c in enumerate(metadata.children)
        ),
        metadata.extra,
    )
    out_basis = position_operator_basis(basis_children)
    out_basis = AsUpcast(out_basis, TupleMetadata((metadata, metadata)))
    data = np.zeros(basis_children.shape, dtype=np.complex128)

    index_along_axis = tuple(
        0 if i != axis else slice(None) for i in range(metadata.n_dim)
    )
    corrugation = _morse_corrugation_potential_fn(
        metadata.children[axis].values, params
    )
    for ax in range(metadata.n_dim):
        if ax == axis:
            continue
        idx_pos = tuple(
            1 if i_ax == ax else i for i_ax, i in enumerate(index_along_axis)
        )
        idx_neg = tuple(
            -1 if i_ax == ax else i for i_ax, i in enumerate(index_along_axis)
        )

        data[idx_pos] += corrugation
        data[idx_neg] += corrugation
    data[index_along_axis] += _morse_potential_fn(
        metadata.children[axis].values, params
    )
    data *= np.sqrt(
        basis_children.fundamental_size / basis_children.children[axis].fundamental_size
    )
    if SIMPLE_FEATURE not in metadata.children[axis].features:
        data /= np.square(
            metadata.children[axis].basis_weights.reshape(
                recast_along_axes(data.shape, {axis})
            )
        )
    return Operator(out_basis, data.astype(np.complex128))

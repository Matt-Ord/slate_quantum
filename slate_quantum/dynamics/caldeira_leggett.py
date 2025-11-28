from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.constants import Boltzmann, hbar  # type: ignore lib
from slate_core import FundamentalBasis, TupleBasis, TupleMetadata, basis
from slate_core.basis import CroppedBasis
from slate_core.metadata import (
    AxisDirections,
    EvenlySpacedLengthMetadata,
)
from slate_core.util import timed

from slate_quantum import operator
from slate_quantum.metadata import TimeMetadata
from slate_quantum.operator._operator import (
    Operator,
    OperatorBasis,
    operator_basis,
)
from slate_quantum.state._state import StateList

try:
    import qutip  # type: ignore lib
except ImportError:
    qutip = None

if TYPE_CHECKING:
    from slate_core import Basis, Ctype

    from slate_quantum.dynamics._realization import (
        RealizationList,
        RealizationListBasis,
    )
    from slate_quantum.state._state import StateWithMetadata


@dataclass(frozen=True, kw_only=True)
class CaldeiraLeggettCondition[M: EvenlySpacedLengthMetadata, E: AxisDirections]:
    """Specifies the condition for a Caldeira-Leggett simulation."""

    mass: float
    friction: float
    temperature: float
    potential: Operator[
        OperatorBasis[TupleMetadata[tuple[M, ...], E]], np.dtype[np.complexfloating]
    ]
    initial_state: StateWithMetadata[TupleMetadata[tuple[M, ...], E]]


DT_RATIO = 10000


def _get_hardwall_basis[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E],
) -> Basis[TupleMetadata[tuple[M, ...], E]]:
    """Get the simulation basis for the Caldeira-Leggett model."""
    # For now we do nothing fancy here - in the future we could investigate automatically
    # ideally, we would like to use the split operator method, but this isn't supported by qutip yet
    # For stability, we use a hardwall basis, such that the state is forced to be zero at the boundaries.
    trigonometric = basis.trigonometric_transformed_from_metadata(
        metadata, fn="sin", ty="type 2"
    )
    return basis.with_modified_children(
        trigonometric,
        # Remove the high frequency components to avoid numerical instabilities
        lambda _, b: CroppedBasis((b.size) // 2, b).upcast(),
    ).upcast()


@dataclass(frozen=True, kw_only=True)
class _Units:
    """Natural units for the Caldeira-Leggett simulation."""

    lengthscale: float
    timescale: float
    mass_scale: float

    @property
    def hbar(self) -> float:
        """Get the value of hbar in these natural units."""
        return hbar * (self.timescale) / (self.mass_scale * self.lengthscale**2)

    def length_to_si(self, length: float) -> float:
        """Convert a length in natural units to SI units."""
        return length * self.lengthscale

    def length_from_si(self, length: float) -> float:
        """Convert a length in SI units to natural units."""
        return length / self.lengthscale

    def time_to_si(self, time: float) -> float:
        """Convert a time in natural units to SI units."""
        return time * self.timescale

    def time_from_si(self, time: float) -> float:
        """Convert a time in SI units to natural units."""
        return time / self.timescale

    def frequency_to_si(self, frequency: float) -> float:
        """Convert a frequency in natural units to SI units."""
        return frequency / self.timescale

    def frequency_from_si(self, frequency: float) -> float:
        """Convert a frequency in SI units to natural units."""
        return frequency * self.timescale

    def mass_to_si(self, mass: float) -> float:
        """Convert a mass in natural units to SI units."""
        return mass * self.mass_scale

    def mass_from_si(self, mass: float) -> float:
        """Convert a mass in SI units to natural units."""
        return mass / self.mass_scale

    def energy_to_si(self, energy: float) -> float:
        """Convert an energy in natural units to SI units."""
        return energy * (self.mass_scale * self.lengthscale**2) / (self.timescale**2)

    def energy_from_si(self, energy: float) -> float:
        """Convert an energy in SI units to natural units."""
        return energy / ((self.mass_scale * self.lengthscale**2) / (self.timescale**2))

    @staticmethod
    def from_condition(
        unit_cell_length: float,
        temperature: float,
    ) -> _Units:
        """Create natural units from the condition parameters."""
        kb_t = Boltzmann * temperature  # Boltzmann constant times temperature
        # We rescale so that the repeat length is 1 in natural units
        lengthscale = unit_cell_length
        # The thermal energy must be 1, and d psi / dt = E / hbar
        timescale = hbar / kb_t
        mass_scale = kb_t * (timescale**2) / (lengthscale**2)
        out = _Units(
            lengthscale=lengthscale, timescale=timescale, mass_scale=mass_scale
        )

        assert out.energy_from_si(kb_t) == 1.0, (
            "Natural units energy scaling is incorrect."
        )
        return out


def _get_simulation_metadata[M: EvenlySpacedLengthMetadata, E: AxisDirections](
    metadata: TupleMetadata[tuple[M, ...], E], units: _Units
) -> TupleMetadata[tuple[M, ...], AxisDirections]:
    """Create a basis with n_repeat repeats of the periodic potential."""
    extra = AxisDirections(
        vectors=tuple(v / units.lengthscale for v in metadata.extra.vectors),
    )
    return TupleMetadata(metadata.children, extra)


@timed
def simulate_caldeira_leggett_realizations[
    M: EvenlySpacedLengthMetadata,
    E: AxisDirections,
    MT: TimeMetadata,
](
    condition: CaldeiraLeggettCondition[M, E],
    times: Basis[MT, Ctype[np.complexfloating]],
    *,
    n_realizations: int = 1,
) -> RealizationList[RealizationListBasis[MT, TupleMetadata[tuple[M, ...], E]]]:
    """Simulate the Caldeira-Leggett model using the Stochastic Schrödinger Equation (SSE).

    To avoid numerical instabilities, the simulation uses a hard-wall boundary condition,
    centered at the origin. This is achieved by generating the initial hamiltonian and
    environment operators in a repeat of the simulation basis, which is then truncated.

    This function also pre-calculates a suitable set of natural units
    for the simulation based on the Hamiltonian and initial state provided in the `condition`.

    Internally, this uses the `ssesolve` function from the `QuTiP` library.

    Raises
    ------
    ImportError
        If qutip is not installed
    """
    if qutip is None:
        msg = "The qutip package is required to use this function. Please install it with `pip install qutip`."
        raise ImportError(msg)

    units = _Units.from_condition(
        unit_cell_length=condition.potential.basis.metadata()
        .children[0]
        .extra.vectors[0][0],
        temperature=condition.temperature,
    )

    simulation_metadata = _get_simulation_metadata(
        condition.potential.basis.metadata().children[0], units
    )
    simulation_basis = _get_hardwall_basis(simulation_metadata)

    hamiltonian = operator.build.kinetic_hamiltonian(
        # Convert V(x) to natural units. since hardwall basis is
        # a `mul` basis, we can just scale the operator data directly
        Operator(
            operator_basis(simulation_basis).upcast(),
            condition.potential.with_basis(
                operator_basis(
                    _get_hardwall_basis(
                        condition.potential.basis.metadata().children[0]
                    )
                ).upcast()
            ).raw_data
            * units.energy_from_si(1.0),
        ),
        units.mass_from_si(condition.mass),
        hbar=units.hbar,
    )
    shift_operator = operator.build.caldeira_leggett_shift(
        simulation_basis.metadata(),
        friction=units.frequency_from_si(condition.friction),
    )
    environment_operator = operator.build.caldeira_leggett_collapse(
        simulation_basis.metadata(),
        friction=units.frequency_from_si(condition.friction),
        kb_t=units.energy_from_si(Boltzmann * condition.temperature),
        mass=units.mass_from_si(condition.mass),
        hbar=units.hbar,
    )

    print(  # noqa: T201
        f"Will require {units.time_from_si(times.metadata().delta) * DT_RATIO:.2e} time steps"
    )

    # Simulates an SSE defined as in eqn 4.76 in https://doi.org/10.1017/CBO9780511813948
    # Using the qutip.ssesolve method
    # d|psi(t)> = - i H |psi(t)> dt
    #             - (S dagger S / 2 - <S + S dagger> S / 2 + <S + S dagger>^2 / 8) |psi(t)> dt
    #             + (S - (<S + S dagger> / 2)) |psi(t)> dW
    result = qutip.ssesolve(  # type: ignore lib
        H=qutip.Qobj(
            (hamiltonian + shift_operator)
            .with_basis(operator_basis(simulation_basis))
            .raw_data.reshape((simulation_basis.size, simulation_basis.size))
            / units.hbar,
        ),
        psi0=qutip.Qobj(condition.initial_state.with_basis(simulation_basis).raw_data),
        tlist=times.metadata().values / units.timescale,
        ntraj=n_realizations,
        sc_ops=[
            qutip.Qobj(
                environment_operator.with_basis(
                    TupleBasis((simulation_basis, simulation_basis))
                ).raw_data.reshape((simulation_basis.size, simulation_basis.size)),
            )
        ],
        options={
            "progress_bar": "enhanced",
            "store_states": True,
            "keep_runs_results": True,
            "method": "platen",
            "dt": 1 / DT_RATIO,
        },
        heterodyne=True,
    )

    return StateList(
        TupleBasis(
            (
                TupleBasis(
                    (
                        FundamentalBasis.from_size(n_realizations),
                        basis.as_fundamental(times),
                    )
                ).upcast(),
                _get_hardwall_basis(condition.potential.basis.metadata().children[0]),
            )
        ).upcast(),
        np.array(
            np.asarray(
                [
                    [state.full().reshape(-1) for state in realization]  # type: ignore lib
                    for realization in result.states  # type: ignore lib
                ]
            ),
            dtype=np.complex128,
        ),
    )

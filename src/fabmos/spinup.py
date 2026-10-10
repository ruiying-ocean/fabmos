"""Accelerated spin-up of simulations with periodic forcing.

A simulation whose forcing repeats after a fixed number of time steps (typically one
year) approaches a state that repeats too. Stepping forward until it does can take
thousands of periods. :func:`spin_up` needs far fewer: after each period, it restarts
the simulation from the combination of previous states that is expected to drift
least (Anderson acceleration), as proposed for ocean models by Khatiwala (2023,
https://doi.org/10.1029/2022MS003447).

Usage, after ``sim.start(...)``::

    nstep = int(round(360 * 86400 / sim.timestep))  # time steps per year
    fabmos.spinup.spin_up(sim, nstep, tolerance=1e-5, max_iterations=200)
"""

from typing import Optional

import numpy as np
import pygetm

from .anderson import AndersonAccelerator
from .simulator import Simulator

# Calendars in which all years have the same number of days
FIXED_YEAR_CALENDARS = ("360_day", "noleap", "365_day", "all_leap", "366_day")


def spin_up(
    sim: Simulator,
    nstep: int,
    tolerance: float,
    max_iterations: int,
    m: int = 50,
    beta: float = 1.0,
) -> tuple[int, float]:
    """Advance a simulation until its state repeats after ``nstep`` time steps.

    The state consists of all fields that would be saved in a restart file. The
    simulation must have been started, and its forcing must have a period of
    ``nstep`` time steps (or be constant). Forcing that repeats every year only has
    such a period in a calendar in which all years are equally long, such as
    ``360_day`` or ``noleap``. Time keeps increasing: each iteration advances the
    simulation by ``nstep`` time steps, and all configured output is written as usual.

    Args:
        sim: the simulation
        nstep: number of time steps in one period of the forcing
        tolerance: stop when the drift over one period drops to this value. The drift
            is the norm of the change in state over the period, divided by the norm
            of the state itself.
        max_iterations: maximum number of periods to simulate
        m: number of previous states to combine. On the first MPI rank, memory use
            peaks at about 4 * m times the size of the state.
        beta: damping (1 for none), see :class:`fabmos.anderson.AndersonAccelerator`

    Returns:
        The number of periods simulated and the drift over the last one. If that
        still exceeds ``tolerance``, the simulation has not converged. It can then
        be continued by calling this function again.
    """
    if nstep <= 0 or nstep % sim.split_factor != 0:
        raise ValueError(
            f"A period of {nstep} time steps does not consist of a whole number of"
            f" transport steps, which take {sim.split_factor} time step(s) each."
        )
    # Created on every rank, though only the first uses it, so that invalid settings
    # are refused by all ranks alike.
    accelerator = AndersonAccelerator(m, beta)
    if sim.tiling.rank == 0 and sim.time.calendar not in FIXED_YEAR_CALENDARS:
        sim.logger.warning(
            f"In the {sim.time.calendar} calendar, years differ in length. Forcing"
            " that repeats every year then does not repeat after a fixed number of"
            " time steps, which keeps the spin-up from converging."
        )
    return _iterate(sim, nstep, tolerance, max_iterations, accelerator)


@pygetm.simulation.log_exceptions
def _iterate(
    sim: Simulator,
    nstep: int,
    tolerance: float,
    max_iterations: int,
    accelerator: AndersonAccelerator,
) -> tuple[int, float]:
    # log_exceptions stops all ranks if any of them fails. Otherwise the others
    # would wait forever in the next gather or scatter.
    comm = sim.tiling.comm
    root = comm.rank == 0
    fields = [
        field
        for _, field in sorted(sim.output_manager.fields.items())
        if field.attrs.get("_part_of_state", False)
    ]

    # Selections of unmasked points in global arrays (None: all points).
    # Like every gather and scatter below, these calls are collective: all ranks must
    # make them, although only the first rank obtains and provides values.
    unmasked = {
        2: None if sim.unmasked2d is None else sim.unmasked2d.gather(),
        3: None if sim.unmasked3d is None else sim.unmasked3d.gather(),
    }

    def select(values: np.ndarray):
        where = unmasked[values.ndim]
        return ... if where is None else where

    def get_state() -> Optional[np.ndarray]:
        all_values = [field.gather() for field in fields]
        if root:
            return np.concatenate([v[select(v)].ravel() for v in all_values])

    def set_state(x: Optional[np.ndarray]):
        offset = 0
        for field in fields:
            values = field.gather()
            if root:
                where = select(values)
                n = values[where].size
                values[where] = x[offset : offset + n].reshape(values[where].shape)
                offset += n
            field.scatter(values)
        sim.state_changed()

    x = get_state()
    for iteration in range(1, max_iterations + 1):
        for _ in range(nstep):
            sim.advance()
        gx = get_state()
        drift = None
        if root:
            norm = np.linalg.norm(gx)
            drift = float(np.linalg.norm(gx - x) / norm) if norm > 0.0 else 0.0
            sim.logger.info(f"Spin-up iteration {iteration}: drift = {drift:.3e}")
        drift = comm.bcast(drift)
        if drift <= tolerance:
            break
        if iteration < max_iterations:
            if root:
                x = accelerator.update(x, gx)
            set_state(x)
    else:
        if root:
            sim.logger.warning(
                f"Spin-up has not converged after {max_iterations} iterations:"
                f" drift {drift:.3e} exceeds the tolerance of {tolerance:.3e}."
            )
    return iteration, drift

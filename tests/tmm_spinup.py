"""Check the accelerated spin-up on a TMM simulation with a known equilibrium.

A tracer is supplied at the sea surface at a constant rate and decays everywhere.
Whatever the circulation, its global content must end up at supply / decay rate.

Usage: [mpiexec -n <N>] python tmm_spinup.py <path to TMM setup>
"""

import argparse
import logging

import cftime
import numpy as np

import fabmos
import fabmos.spinup
import fabmos.transport.tmm

# mpi4py must be imported after fabmos, as pygetm.parallel configures mpi4py.rc
from mpi4py import MPI

CALENDAR = "360_day"
YEAR = 360 * 86400.0
DECAY_RATE = 1.0 / (10.0 * YEAR)  # s-1
SURFACE_FLUX = 1.0e-5  # mol m-2 s-1
TOLERANCE = 1e-6
MAX_ITERATIONS = 40

FABM_CONFIG = dict(
    instances=dict(
        tracer=dict(model="tracer"),
        zero=dict(model="interior_constant", parameters=dict(value=0.0)),
        decay=dict(
            model="interior_relaxation",
            parameters=dict(rate=DECAY_RATE),
            coupling=dict(original="tracer/c", target="zero/data"),
        ),
        supply=dict(
            model="constant_surface_flux",
            parameters=dict(flux=SURFACE_FLUX),
            coupling=dict(target="tracer/c"),
        ),
    )
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", help="the path to the original TMM data")
args = parser.parse_args()

domain = fabmos.transport.tmm.create_domain(args.path)
sim = fabmos.transport.tmm.Simulator(
    domain, calendar=CALENDAR, fabm=FABM_CONFIG, log_level=logging.WARNING
)
comm = sim.tiling.comm

# The virtual tracer flux due to precipitation - evaporation changes tracer content.
# Switch it off, so that the equilibrium content is known exactly.
sim.use_virtual_flux = False

# A 2D field that is part of the model state, like the bottom-attached state
# variables of some biogeochemical models. Nothing changes it.
extra = sim.T.array(
    name="extra", units="1", long_name="extra state", attrs=dict(_part_of_state=True)
)
extra.fill(3.0)

wet = sim.T.mask3d.values != 0
volume = (sim.T.hn.values * sim.T.area.values)[wet]
total_area = comm.allreduce(sim.T.area.values.sum())
expected_content = SURFACE_FLUX * total_area / DECAY_RATE


def content() -> float:
    return comm.allreduce(volume @ sim["tracer_c"].values[wet])


# Two biogeochemical steps per transport step
sim.start(
    cftime.datetime(2000, 1, 1, calendar=CALENDAR),
    timestep=0.5 * domain._delta_t,
    transport_timestep=domain._delta_t,
    report=0,
    report_totals=0,
)
nstep = int(round(YEAR / sim.timestep))

# Masked points hold FABM's missing value from here on
masked_before = sim["tracer_c"].values[~wet].copy()

# A period that ends halfway a transport step must be refused
try:
    fabmos.spinup.spin_up(sim, nstep + 1, TOLERANCE, MAX_ITERATIONS)
except ValueError:
    pass
else:
    raise AssertionError("A period of an odd number of time steps was accepted")

# Too few iterations: no convergence, but the simulation remains usable
initial_error = content() / expected_content - 1.0
iterations, drift = fabmos.spinup.spin_up(sim, nstep, TOLERANCE, max_iterations=2)
assert (iterations, drift > TOLERANCE) == (2, True), "Converged within 2 iterations?"

# Continue until converged
iterations, drift = fabmos.spinup.spin_up(sim, nstep, TOLERANCE, MAX_ITERATIONS)
iterations += 2
assert drift <= TOLERANCE, f"Not converged: drift after {iterations} years is {drift}"

error = content() / expected_content - 1.0
plain_error = initial_error * np.exp(-DECAY_RATE * YEAR * iterations)
minimum = comm.allreduce(sim["tracer_c"].values[wet].min(), op=MPI.MIN)
if comm.rank == 0:
    print(f"Converged after {iterations} years: drift {drift:.3e}")
    print(f"Content differs from supply / decay rate by {error:+.3e}")
    print(f"Without acceleration, that would be {plain_error:+.3e}")
assert abs(error) < 1e-3, "Wrong equilibrium"
assert abs(error) < 0.1 * abs(plain_error), "Acceleration gained too little"
assert minimum >= 0.0, f"Negative tracer values (minimum {minimum})"

# The extra state field and masked points must be left exactly as they were
unchanged = (extra.values == 3.0).all() and np.array_equal(
    sim["tracer_c"].values[~wet], masked_before
)
assert comm.allreduce(unchanged, op=MPI.LAND), "Extra state or masked points changed"
if comm.rank == 0:
    print(f"OK: accelerated spin-up reached the equilibrium on {comm.size} rank(s)")

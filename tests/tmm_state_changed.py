"""Check that the model state can be changed from outside during a TMM simulation.

After Simulator.state_changed(), the simulation must continue from the new state as
if it had been reached by time stepping.

Usage: [mpiexec -n <N>] python tmm_state_changed.py <path to TMM setup>
"""

import argparse
import logging

import cftime
import numpy as np

import fabmos
import fabmos.transport.tmm

# mpi4py must be imported after fabmos, as pygetm.parallel configures mpi4py.rc
from mpi4py import MPI

CALENDAR = "360_day"

FABM_CONFIG = dict(
    instances=dict(
        # two identical passive tracers; b will be doubled halfway
        a=dict(model="tracer"),
        b=dict(model="tracer"),
        # will be given invalid (negative) values in the far north halfway
        negative=dict(model="tracer"),
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
wet = sim.T.mask3d.values != 0
north = np.broadcast_to(sim.T.lat.values > 0.0, wet.shape)
far_north = np.broadcast_to(sim.T.lat.values > 60.0, wet.shape)
volume = (sim.T.hn.values * sim.T.area.values)[wet]


def content(name: str) -> float:
    return comm.allreduce(volume @ sim[name].values[wet])


sim["a_c"].fill(1.0 + north)
sim["b_c"].fill(1.0 + north)

# Two biogeochemical steps per transport step, so that both kinds of step are covered
sim.start(
    cftime.datetime(2000, 1, 1, calendar=CALENDAR),
    timestep=0.5 * domain._delta_t,
    transport_timestep=domain._delta_t,
)
for _ in range(10):
    sim.advance()

sim["b_c"].values[wet] *= 2.0
sim["negative_c"].values[wet & far_north] = -1.0
content_before = content("negative_c")
assert content_before > 0.0, "This setup has too little ocean south of 60 N"
sim.state_changed()

# Invalid values have been repaired, without changing the tracer's content
minimum = comm.allreduce(sim["negative_c"].values[wet].min(), op=MPI.MIN)
assert minimum >= 0.0, f"negative_c still has negative values (minimum {minimum})"
change = content("negative_c") / content_before - 1.0
assert abs(change) < 1e-12, f"Repair changed the content of negative_c by {change:+.3e}"

# The transport's own copy of the tracers equals the FABM state
for i, tracer in enumerate(sim.tracers):
    same = np.array_equal(
        sim.local_tracers[:, i], tracer.values.ravel()[sim.wet_indices]
    )
    message = f"Transport has stale values of {tracer.name}"
    assert comm.allreduce(same, op=MPI.LAND), message

for _ in range(10):
    sim.advance()

# Transport is linear and multiplication by 2 is exact in floating point arithmetic,
# so b must be exactly twice a. If transport still used the old values of b,
# it would add the doubling as if it were a biogeochemical source.
exact = np.array_equal(sim["b_c"].values[wet], 2.0 * sim["a_c"].values[wet])
assert comm.allreduce(exact, op=MPI.LAND), "b is not exactly twice a"
if comm.rank == 0:
    print(f"OK: simulation continued from the changed state on {comm.size} rank(s)")

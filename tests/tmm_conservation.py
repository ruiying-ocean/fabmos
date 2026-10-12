"""Check that the TMM transport conserves tracers while FABM repairs invalid values.

Transport matrices are not monotone: transporting a field with sharp gradients
produces negative concentrations, which FABM's repair clips to zero. That clip must
not change the global content of any tracer. The virtual tracer flux due to
precipitation - evaporation stays on, so its rescale of the surface layer is checked
too.

Usage: [mpiexec -n <N>] python tmm_conservation.py <path to TMM setup> [--days DAYS]

With --no-repair (serial only), check instead that nothing is repaired silently:
the simulation must then stop at the first invalid value.

The tracers are saved to output.nc. To check that they do not depend on the subdomain
decomposition, compare that file between runs with different numbers of cores:
pygetm-test-scaling --nmax 2 --compare output.nc tmm_conservation.py <path to TMM setup>
"""

import argparse
import datetime
import sys

import cftime

import fabmos
import fabmos.transport.tmm

# mpi4py must be imported after fabmos, as pygetm.parallel configures mpi4py.rc
from mpi4py import MPI

# Maximum relative change in global tracer content.
# The matrices themselves conserve to about 1e-9 per simulated year.
TOLERANCE = 1e-8

CALENDAR = "360_day"

FABM_CONFIG = dict(
    instances=dict(
        # sharp edges: transport will undershoot
        box=dict(model="tracer"),
        # changed by FABM between transport steps, and develops strong gradients
        sinking=dict(model="tracer", parameters=dict(vertical_velocity=-100.0)),
        # no content at all: must stay zero rather than become NaN
        empty=dict(model="tracer"),
    )
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", help="the path to the original TMM data")
parser.add_argument("--days", type=int, default=30, help="simulation length")
parser.add_argument("--no-repair", action="store_true", help="switch off repair")
args = parser.parse_args()

domain = fabmos.transport.tmm.create_domain(args.path)
sim = fabmos.transport.tmm.Simulator(domain, calendar=CALENDAR, fabm=FABM_CONFIG)

lon = sim.T.lon.values % 360.0
lat = sim.T.lat.values
sim["box_c"].fill((lon >= 300.0) & (lon <= 330.0) & (lat >= 30.0) & (lat <= 55.0))
sim["empty_c"].fill(0.0)
sim.fabm.repair = not args.no_repair

comm = sim.tiling.comm
volume = (sim.T.hn.values * sim.T.area.values).ravel()[sim.wet_indices]


def content_and_minimum(name: str) -> tuple[float, float]:
    values = sim[name].values.ravel()[sim.wet_indices]
    return comm.allreduce(volume @ values), comm.allreduce(values.min(), op=MPI.MIN)


names = [tracer.name for tracer in sim.tracers]
initial = {name: content_and_minimum(name) for name in names}

out = sim.output_manager.add_netcdf_file(
    "output.nc", interval=datetime.timedelta(days=1), save_initial=False
)
out.request(*names)

# Two biogeochemical steps per transport step, so that both kinds of step are covered
sim.start(
    cftime.datetime(2000, 1, 1, calendar=CALENDAR),
    timestep=0.5 * domain._delta_t,
    transport_timestep=domain._delta_t,
    report=datetime.timedelta(days=args.days),
)

# Without negative values after transport, this check would not test anything
sim.gather_req.Wait()
transported = sim.Aimp @ (sim.Aexp @ sim.global_tracers)
assert (
    comm.allreduce(transported.min(), op=MPI.MIN) < 0.0
), "Transport does not produce negative values for this setup; nothing to check."

try:
    for _ in range(int(round(args.days * 86400.0 / sim.timestep))):
        sim.advance()
except Exception as e:
    if not (args.no_repair and "invalid values" in str(e)):
        raise
    print(f"OK: without repair, the simulation stops: {e}")
    sys.exit(0)
assert not args.no_repair, "Repair is off, yet negative values went unnoticed."
sim.finish()

final = {name: content_and_minimum(name) for name in names}
for name in names:
    change = final[name][0] - initial[name][0]
    if initial[name][0] != 0.0:
        change /= initial[name][0]
    if comm.rank == 0:
        print(f"{name}: change in content {change:+.3e}, minimum {final[name][1]:.3e}")
for name in names:
    assert final[name][1] >= 0.0, f"{name} has negative values"
    if initial[name][0] == 0.0:
        assert final[name] == (0.0, 0.0), f"{name} started empty but now has content"
    else:
        change = final[name][0] / initial[name][0] - 1.0
        assert abs(change) < TOLERANCE, f"{name}: content changed by {change:+.3e}"
if comm.rank == 0:
    print(f"OK: tracer content conserved over {args.days} days on {comm.size} rank(s)")

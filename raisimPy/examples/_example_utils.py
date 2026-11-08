"""Shared activation, finite-run options, and server cleanup for Python examples."""
import argparse
import atexit
from contextlib import contextmanager
from functools import partial
from itertools import count
import os
from pathlib import Path
import time


RESOURCE_DIR = Path(__file__).resolve().parents[2] / "rsc"
_running_servers = {}


def setup(raisim, parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="RaiSim Python example")
    parser.add_argument("--steps", type=int, help="stop after this many simulation steps")
    parser.add_argument("--headless", action="store_true", help="run without opening a viewer server")
    parser.add_argument("--no-sleep", action="store_true", help="run without real-time pacing")
    parser.add_argument("--port", type=int, default=8080, help="viewer TCP port (0 selects a free port)")
    parser.add_argument("--activation-key", default=os.environ.get("RAISIM_ACTIVATION_KEY"),
                        help="activation file; defaults to RAISIM_ACTIVATION_KEY or the local key")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.activation_key:
        key = Path(args.activation_key).expanduser()
        if not key.is_file():
            parser.error(f"activation file does not exist: {key}")
        raisim.World.setActivationKey(str(key.resolve()))
    else:
        for key in (RESOURCE_DIR / "activation.raisim", Path.home() / ".raisim/activation.raisim"):
            if key.is_file():
                raisim.World.setActivationKey(str(key))
                break
    return args


def simulation_steps(args, default=None):
    steps = args.steps if args.steps is not None else default
    return count() if steps is None else range(steps)


def pace(args, duration):
    if not args.headless and not args.no_sleep:
        time.sleep(duration)


def start_server(server, args):
    if not args.headless:
        server.launchServer(args.port)
        callback = partial(stop_server, server)
        _running_servers[id(server)] = callback
        atexit.register(callback)


def stop_server(server):
    callback = _running_servers.pop(id(server), None)
    if callback is not None:
        atexit.unregister(callback)
        server.killServer()


@contextmanager
def locked_server(server):
    server.lockVisualizationServerMutex()
    try:
        yield
    finally:
        server.unlockVisualizationServerMutex()


def finish_example(server, world):
    stop_server(server)
    import numpy as np
    import raisimpy as raisim
    for obj in world.getObjList():
        if isinstance(obj, raisim.ArticulatedSystem):
            state = obj.getGeneralizedCoordinate()
        elif isinstance(obj, (raisim.DeformableObject, raisim.GranularSystem)):
            state = obj.getPositions()
        else:
            state = obj.getPosition()
        if not np.isfinite(state).all():
            raise RuntimeError(f"Non-finite state in {obj.getName()}")
    print(f"Simulated {world.getWorldTime():.6f} seconds")

"""Load the forest terrain, tree/rock colliders, and saved visual scene."""
import raisimpy as raisim

from _world_utils import setup_scene, run_scene


def main():
    args = setup_scene(raisim, "forest")
    world = raisim.World(str(args.scene))
    run_scene(raisim, world, args)


if __name__ == "__main__":
    main()

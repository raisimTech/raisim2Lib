"""ANYmal C holds its posture beside the roadworks in the bundled city scene."""
import raisimpy as raisim

from _world_utils import setup_scene, add_standing_anymal, run_scene


def main():
    args = setup_scene(raisim, "city")
    world = raisim.World(str(args.scene))
    # Matches the clear robot spawn in the native rayrai_city example.
    add_standing_anymal(raisim, world, -36., -1.6, 2.6)
    run_scene(raisim, world, args)


if __name__ == "__main__":
    main()

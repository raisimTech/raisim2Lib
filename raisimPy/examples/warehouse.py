"""ANYmal C stands in the middle aisle of the bundled warehouse scene."""
import raisimpy as raisim

from _world_utils import setup_scene, add_standing_anymal, run_scene


def main():
    args = setup_scene(raisim, "warehouse")
    world = raisim.World(str(args.scene))
    # Matches the free aisle location in the native rayrai_warehouse example.
    add_standing_anymal(raisim, world, 4.4, 0.3, -0.6)
    run_scene(raisim, world, args)


if __name__ == "__main__":
    main()

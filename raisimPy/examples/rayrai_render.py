"""Render in process, capture RGBA/depth NumPy arrays, and save a screenshot."""
import argparse
from pathlib import Path
import numpy as np
import raisimpy as raisim
from _example_utils import setup, simulation_steps, pace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional PNG screenshot')
    args = setup(raisim, parser)
    rr = raisim.rayrai
    world = raisim.World()
    world.setTimeStep(.001)
    world.addGround()
    ball = world.addSphere(.3, 1.)
    ball.setPosition(0., 0., 2.)
    with rr.RayraiWindow(world, 640, 480, visible=not args.headless) as viewer:
        viewer.setRenderQualityPreset(rr.RenderQualityPreset.Balanced)
        marker = viewer.addVisualSphere('marker', .12, [1., .2, .1, 1.])
        marker.setPosition([0., 1., .6])
        material = rr.Material()
        material.roughnessFactor = .25
        marker.setMaterialOverride(material)
        viewer.setWeatherPreset(rr.WeatherPreset.Clear)
        for index in simulation_steps(args, default=1000):
            viewer.integrateWorldThreadSafe()
            if index % 16 == 0 and not viewer.render():
                break
            pace(args, world.getTimeStep())
        viewer.render()
        rgba = viewer.readRgba()
        camera = viewer.getCamera()
        viewer.renderDepthPlaneDistance(camera)
        depth = camera.readLinearDepth()
        if not np.isfinite(depth).all():
            raise RuntimeError('Non-finite captured depth')
        if args.output:
            viewer.saveScreenshot(args.output)
        print(f'RGBA: {rgba.shape} {rgba.dtype}; depth: {depth.shape} {depth.dtype}')
    print(f'Simulated {world.getWorldTime():.6f} seconds')


if __name__ == '__main__':
    main()

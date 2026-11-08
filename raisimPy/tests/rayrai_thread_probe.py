"""Isolated lifecycle/concurrency probes: a regression must time out or crash a child only."""
import gc
import sys
import threading
import weakref
import os
from pathlib import Path
import numpy as np

import raisimpy as raisim

rr = raisim.rayrai
world = raisim.World()
world.addGround()
viewer = rr.RayraiWindow(world, 32, 32, visible=False)
mode = sys.argv[1]

if mode in ('wrong_thread', 'foreign_gc', 'callback_render'):
    retained = []
    if mode == 'foreign_gc':
        retained.append(viewer.addVisualSphere('dispose-in-callback', .1))
        viewer.removeVisualObject('dispose-in-callback')
    called, failures = [], []

    def callback(_):
        if called:
            return
        called.append(True)
        if mode == 'callback_render':
            try:
                viewer.render()
            except RuntimeError as exc:
                failures.append(str(exc))
            return

        def worker():
            if mode == 'foreign_gc':
                retained.clear()
                gc.collect()
            else:
                try:
                    viewer.getCamera()
                except RuntimeError as exc:
                    failures.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    viewer.setWeatherThunderCallback(callback)
    settings = rr.RayraiWindow.defaultWeatherSettings(rr.WeatherPreset.Storm)
    settings.quality = rr.WeatherQuality.High
    settings.lightningRate = 16.
    viewer.setWeatherSettings(settings)
    for _ in range(1200):
        viewer.updateWeather(.1)
    assert called
    if mode == 'wrong_thread':
        assert failures and 'thread' in failures[0]
    if mode == 'callback_render':
        assert failures and 'callback' in failures[0]
    viewer.setWeatherThunderCallback(None)
    viewer.getCamera()  # Flush disposal on the owner with its context current.
elif mode == 'world_lock':
    locked = threading.Event()
    release = threading.Event()

    def physics():
        world.lock()
        try:
            locked.set()
            release.wait()
        finally:
            world.unlock()

    def contender():
        world.lock()
        world.unlock()

    physics_thread = threading.Thread(target=physics)
    physics_thread.start()
    locked.wait()
    timer = threading.Timer(.1, release.set)
    timer.start()
    contender_thread = threading.Thread(target=contender)
    contender_thread.start()
    assert viewer.render()
    physics_thread.join()
    contender_thread.join()
    timer.join()
elif mode == 'finalizer':
    rejected = []

    class Callback:
        def __call__(self, _):
            pass

        def __del__(self):
            try:
                viewer.render()
            except RuntimeError as exc:
                rejected.append(str(exc))

    viewer.setWeatherThunderCallback(Callback())
    viewer.close()
    assert rejected, 'The callback finalizer reentered an active renderer teardown'
elif mode == 'manual_locks':
    other=rr.RayraiWindow(world,32,32,visible=False)
    viewer.lockVisualizationServerMutex()
    try:
        try:
            other.lockVisualizationServerMutex()
        except RuntimeError:
            pass
        else:
            raise AssertionError('Duplicate World lock accepted')
        assert viewer.render()
        assert other.render()
        viewer.clearPhysicsSnapshot()
        new_world=raisim.World()
        viewer.swapWorld(new_world)
        assert viewer.render()
        viewer.close()
    finally:
        viewer.unlockVisualizationServerMutex()
    world.lock()
    world.unlock()
    other.close()
    try:
        viewer.unlockVisualizationServerMutex()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unpaired unlock accepted')
elif mode == 'callback_cycle':
    reference=weakref.ref(viewer)
    viewer.setWeatherThunderCallback(lambda _, retained=viewer: None)
    child=viewer.getCamera()
    del viewer
    gc.collect()
    assert reference() is None, 'Callback cycle retained the Python viewer'
    child.getPosition()
    del child
    gc.collect()
    viewer=rr.RayraiWindow(world,32,32,visible=False)
    class Callback:
        __slots__=()
        def __call__(self,_):
            pass
    viewer.setWeatherThunderCallback(Callback())
    viewer.setWeatherThunderCallback(None)
elif mode == 'target_lifetimes':
    assets=rr.RayraiGlobalAsset(viewer)
    standalone=rr.Visuals(viewer,raisim.ShapeType.Sphere,'standalone',assets)
    standalone.setPosition([1.,2.,3.])
    viewer.setTargetVisual(standalone)
    del standalone
    gc.collect()
    target=viewer.getTargetVisual()
    assert list(target.getPosition())==[1.,2.,3.]
    viewer.setTargetVisual(None)
    assert list(target.getPosition())==[1.,2.,3.]
    visual=viewer.addVisualSphere('retained-removed-target',.1)
    viewer.setTargetVisual(visual)
    del visual
    target=viewer.getTargetVisual()
    viewer.removeVisualObject('retained-removed-target')
    gc.collect()
    target.getPosition()
elif mode == 'contexts':
    other=rr.RayraiWindow(world,32,32,visible=False)
    resource=rr.Shader(viewer)
    other.makeCurrent()
    del resource
    gc.collect()
    # Static GPU construction uses the active viewer; disposal must preserve it.
    texture=rr.RayraiWindow.updateSingleChannelTexture(0,np.zeros((2,2),dtype=np.uint8))
    assert texture>0
    viewer.close()
    texture=rr.RayraiWindow.updateSingleChannelTexture(texture,np.ones((2,2),dtype=np.uint8))
    assert texture>0
    assert other.render(processEvents=False)
    other.close()
elif mode == 'visible_font':
    visible=rr.RayraiWindow(world,96,64,visible=True)
    fonts=list(Path('/usr/share/fonts').rglob('*.ttf'))
    if fonts:
        font=visible.loadFont(str(fonts[0]),14.)
        visible.setExternalFont(font)
    visible.addSlider('Visible slider',0.,1.,.5)
    assert visible.render()
    visible.swapWorld(raisim.World())
    if fonts:
        visible.setExternalFont(font)
    assert visible.render()
    if os.environ.get('RAISIMPY_TEST_FRAME_FAULTS')=='1':
        os.environ['RAISIMPY_RAYRAI_FAIL_FRAME']='1'
        try:
            visible.render()
        except RuntimeError as error:
            assert 'Injected' in str(error)
        else:
            raise AssertionError('Frame fault was not compiled in')
        finally:
            del os.environ['RAISIMPY_RAYRAI_FAIL_FRAME']
        assert visible.render()
    visible.close()
elif mode == 'failure_paths':
    assert os.environ.get('RAISIMPY_TEST_FRAME_FAULTS')=='1'
    def injected(key,call):
        os.environ[key]='1'
        try:
            call()
        except RuntimeError as error:
            assert 'Injected' in str(error)
        else:
            raise AssertionError('Expected test fault')
        finally:
            del os.environ[key]
    injected('RAISIMPY_RAYRAI_FAIL_LOCK',viewer.lockVisualizationServerMutex)
    world.lock()
    world.unlock()
    injected('RAISIMPY_RAYRAI_FAIL_CREATE',lambda:rr.RayraiWindow(world,32,32,visible=False))
    # A static context-dependent call proves failed construction restored viewer.
    assert rr.RayraiWindow.updateSingleChannelTexture(0,np.zeros((2,2),dtype=np.uint8))>0
    viewer.lockVisualizationServerMutex()
    injected('RAISIMPY_RAYRAI_FAIL_CLOSE',viewer.close)
    world.lock()
    world.unlock()
    viewer.unlockVisualizationServerMutex()
    assert viewer.render()
else:
    raise ValueError(mode)

viewer.close()
print(mode, 'passed', flush=True)

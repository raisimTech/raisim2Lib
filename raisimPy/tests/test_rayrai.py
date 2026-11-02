"""Public API inventory and real OpenGL integration tests for the optional backend."""
import gc
import hashlib
import ast
import builtins
import json
import os
from pathlib import Path
import threading
import tempfile
import subprocess
import sys
import unittest

import numpy as np
import raisimpy as raisim

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(raisim.hasRayrai, "Rayrai backend disabled at build time")
class RayraiApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rr = raisim.rayrai
        # Force the optional module to load without constructing a GL context.
        cls.rr.Material

    def test_inventory_has_no_unimplemented_members(self):
        inventory = json.loads((ROOT / 'raisimPy/rayrai/coverage.json').read_text())
        for entry in inventory:
            self.assertIn(entry['status'], ('generated', 'manual', 'excluded'), entry)
            if entry['status'] == 'excluded':
                self.assertTrue(entry['reason'])
                continue
            owner = self.rr
            path = entry['class'].removeprefix('raisin::').split('::')
            if entry['kind'] in ('FunctionDecl','FunctionTemplate'):
                path = path[:-1]
            for part in path:
                owner = getattr(owner, part)
            name = '__init__' if entry['kind'] == 'CXXConstructor' else entry.get('python_name', entry['name'])
            self.assertTrue(hasattr(owner, name), (entry['class'], name))

    def test_inventory_matches_sdk_headers(self):
        headers = json.loads((ROOT / 'raisimPy/rayrai/header_hashes.json').read_text())
        for path, expected in headers.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected,
                             f"SDK header changed: {path}; regenerate the bindings")

    def test_value_types_need_no_opengl_context(self):
        rr = self.rr
        material = rr.Material()
        material.roughnessFactor = .37
        material.baseColorFactor = [1., .2, .3, 1.]
        np.testing.assert_allclose(material.baseColorFactor, [1., .2, .3, 1.])
        self.assertAlmostEqual(material.roughnessFactor, .37, places=6)
        settings = rr.RayraiWindow.defaultRenderQualitySettings(rr.RenderQualityPreset.Fast)
        settings.bloomEnabled = True
        self.assertTrue(settings.bloomEnabled)
        bloom = settings.bloom()
        bloom.strength = .29
        settings.setBloom(bloom)
        self.assertAlmostEqual(settings.bloomStrength, .29, places=6)
        water = rr.defaultWaterBody(rr.WaterPreset.CalmLake)
        self.assertIsInstance(water, rr.WaterBody)
        self.assertIsInstance(rr.sampleWaterSurface(water, [0., 0.], 0.), rr.WaterSurfaceSample)
        self.assertIsInstance(rr.RayraiWindow.analyzeRgbaLuminance([0, 0, 0, 255] * 4, 2, 2),
                              rr.RayraiWindow.RgbaLuminanceMetrics)

    def test_cpu_physics_snapshot(self):
        world = raisim.World()
        world.addSphere(.2, 1.)
        capture = self.rr.PhysicsSnapshotCapture()
        frame = capture.capture(world)
        self.assertEqual(len(frame.objects), 1)
        self.assertGreater(frame.objects[0].id, 0)
        self.assertIsNotNone(frame.objects[0].description)

    def test_missing_numpy_is_an_import_error(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(raisim.__file__).parent)
        proc = subprocess.run([sys.executable, '-S', '-X', 'faulthandler', '-c',
                               'import raisimpy; import raisimpy_rayrai'],
                              env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("No module named 'numpy'", proc.stderr)
        self.assertNotIn('Aborted', proc.stderr)

    def test_optional_backend_introspection_without_numpy(self):
        env=os.environ.copy()
        env['PYTHONPATH']=str(Path(raisim.__file__).parent)
        script='''
import raisimpy as r
assert not hasattr(r, 'RayraiWindow')
assert not hasattr(r.rayrai, 'Material')
assert 'available' in dir(r.rayrai)
try:
    r.RayraiWindow
except AttributeError as error:
    assert isinstance(error.__cause__, ImportError)
else:
    raise AssertionError('Missing backend must give AttributeError')
'''
        result=subprocess.run([sys.executable,'-S','-c',script],env=env,capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_safe_metadata_and_material_bounds(self):
        rr=self.rr
        material=rr.Material()
        with self.assertRaises(ValueError):
            material.textureUvTransform(rr.Material.TextureSlot.Count)
        info=rr.Material.textureSlotInfo(rr.Material.TextureSlot.Count)
        self.assertEqual(info.name,'unknown')
        with self.assertRaises(AttributeError):
            info.name='temporary Python string'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'file.obj'
            path.write_text('v 0 0 0\n')
            source=rr.assimp.ModelSourceFile.current(str(path))
            self.assertLess(abs(source.time-path.stat().st_mtime_ns),1000000)
            timestamp=1700000000123456789
            source.time=timestamp
            self.assertEqual(source.time,timestamp)
        overrides=rr.RenderOverrides()
        self.assertIsNone(overrides.mesh)
        self.assertFalse(hasattr(rr,'ExternalFrameCache'))
        self.assertFalse(hasattr(rr.ImportVisualSceneOptions(),'report'))

    def test_stub_annotation_names(self):
        for path in (ROOT / 'raisimPy/typings/raisimpy_rayrai').glob('*.pyi'):
            tree = ast.parse(path.read_text())
            names = set(dir(builtins))
            for node in tree.body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Import):
                    names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    names.update(alias.asname or alias.name for alias in node.names)
                elif isinstance(node, ast.Assign):
                    names.update(target.id for target in node.targets if isinstance(target, ast.Name))

            def check(body, scope):
                for node in body:
                    if isinstance(node, ast.ClassDef):
                        local = scope | {child.name for child in node.body
                                         if isinstance(child, (ast.ClassDef, ast.FunctionDef))}
                        check(node.body, local)
                    elif isinstance(node, ast.FunctionDef):
                        annotations = [node.returns] + [arg.annotation for arg in
                            node.args.args + node.args.posonlyargs + node.args.kwonlyargs]
                        for annotation in filter(None, annotations):
                            for item in ast.walk(annotation):
                                if isinstance(item, ast.Name):
                                    self.assertIn(item.id, scope, f'{path.name}:{node.lineno}: {item.id}')
            check(tree.body, names)

    def test_lens_metadata(self):
        lens = raisim.CameraLensModel()
        lens.setOpenCvFisheye(100., 101., 80., 60., .01)
        self.assertTrue(lens.isFisheye())
        prop = raisim.RGBCameraProperties()
        prop.lens = lens
        self.assertEqual(prop.lens.intrinsics.fx, 100.)


@unittest.skipUnless(raisim.hasRayrai and os.environ.get('RAISIMPY_TEST_RAYRAI') == '1',
                     "Set RAISIMPY_TEST_RAYRAI=1 to run real OpenGL tests")
class RayraiRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rr = raisim.rayrai
        cls.world = raisim.World()
        cls.world.addGround()
        cls.sphere = cls.world.addSphere(.3, 1.)
        cls.sphere.setPosition(0., 0., 1.)
        cls.viewer = cls.rr.RayraiWindow(cls.world, 160, 120, visible=False,
                                       shaderBinaryCacheDirectory='/tmp/raisimpy-rayrai-test-shaders')
        cls.viewer.setRenderQualityPreset(cls.rr.RenderQualityPreset.Fast)

    @classmethod
    def tearDownClass(cls):
        cls.viewer.close()

    def test_render_and_owned_image(self):
        v = self.viewer
        self.assertTrue(v.render())
        image = v.readRgba()
        self.assertEqual(image.shape, (120, 160, 4))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(float(image[..., :3].std()), 1.)
        copy = image.copy()
        v.render()
        np.testing.assert_array_equal(image, copy)
        self.assertTrue(v.getCamera().renderTargetDiagnostics())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'render.png'
            v.saveScreenshot(path)
            import struct
            data = path.read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>II', data[16:24]), (160, 120))

    def test_visual_material_instances_points(self):
        rr, v = self.rr, self.viewer
        visual = v.addVisualSphere('python-test', .15, [1., 0., 0., 1.])
        visual.setPosition([0., .7, 1.])
        np.testing.assert_allclose(visual.getPosition(), [0., .7, 1.], atol=1e-6)
        material = rr.Material()
        material.roughnessFactor = .4
        visual.setMaterialOverride(material)
        self.assertAlmostEqual(visual.materialOverride().roughnessFactor, .4, places=6)
        bounds = visual.approximateBounds()
        self.assertEqual(len(bounds), 3)
        points = v.addPointCloud('python-points')
        points.positions = [[0., 0., 2.], [0., 1., 2.]]
        points.colors = [[1., 0., 0., 1.], [0., 1., 0., 1.]]
        points.updatePointBuffer()
        group = v.addInstancedVisuals('python-instances', raisim.ShapeType.Box,
                                     [1., 1., 1.], [1., 0., 0., 1.], [0., 1., 0., 1.])
        spec = rr.InstancedVisuals.InstanceSpec()
        spec.position = [1., 1., 1.]
        group.addInstances([spec])
        v.render()
        v.removeVisualObject('python-test')
        v.removeInstancedVisuals('python-instances')
        v.removePointCloud('python-points')
        # Shared native ownership keeps explicitly retained visuals safe.
        np.testing.assert_allclose(visual.getPosition(), [0., .7, 1.], atol=1e-6)

    def test_native_fatal_is_python_exception(self):
        with self.assertRaises(RuntimeError):
            self.viewer.getVisualObject('not-a-real-visual')
        self.assertTrue(self.viewer.render())

    def test_mesh_shader_and_async_import(self):
        rr, v = self.rr, self.viewer
        vertices = []
        for position in ([0., 0., 1.], [1., 0., 1.], [0., 1., 1.]):
            vertex = rr.Vertex()
            vertex.Position = position
            vertex.Normal = [0., 0., 1.]
            vertices.append(vertex)
        mesh = rr.OpenGLMesh(v, vertices, [0, 1, 2])
        material = mesh.material
        material.roughnessFactor = .7
        mesh.material = material
        chain = rr.RayraiGlobalAsset.MeshLodChain(v)
        chain.levels = [[mesh]]
        chain.triangleCounts = [1]
        self.assertEqual(chain.triangleCounts, [1])
        self.assertEqual(len(chain.levels[0][0].vertices), 3)
        custom = v.addVisualCustomMesh('python-mesh', [mesh])
        self.assertTrue(custom.isValid())
        shader = rr.Shader(v)
        self.assertTrue(shader.compile(
            '#version 330 core\nlayout(location=0) in vec3 p; void main(){gl_Position=vec4(p,1.);}',
            '#version 330 core\nout vec4 c; void main(){c=vec4(1.,0.,0.,1.);}'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'triangle.obj'
            path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
            pending = rr.assimp.loadModelSceneAsync(str(path))
            imported = pending.result()
            self.assertTrue(pending.done())
            meshes = rr.assimp.finishImportedModelScene(imported)
            self.assertTrue(meshes)
            self.assertEqual(len(meshes[0].vertices), 3)
            loaded, sources = rr.assimp.loadModelWithSourceFiles(str(path))
            self.assertTrue(loaded)
            self.assertTrue(sources)
            assets = rr.RayraiGlobalAsset(v)
            assets.setAsyncMeshLoadingEnabled(False)
            cached, key = assets.getMeshesWithCanonicalKey(str(path))
            self.assertTrue(cached)
            self.assertEqual(key, assets.canonicalMeshKey(str(path)))
        v.removeVisualObject('python-mesh')

    def test_weather_water_and_diagnostics(self):
        rr, v = self.rr, self.viewer
        events, errors = [], []
        def on_thunder(diagnostics):
            events.append(diagnostics)
            v.getWeatherSettings()
            try:
                v.close()
            except RuntimeError as exc:
                errors.append(str(exc))
        v.setWeatherThunderCallback(on_thunder)
        storm = rr.RayraiWindow.defaultWeatherSettings(rr.WeatherPreset.Storm)
        storm.quality = rr.WeatherQuality.High
        storm.lightningRate = 16.
        v.setWeatherSettings(storm)
        for _ in range(1200):
            v.updateWeather(.1)
        self.assertTrue(events)
        self.assertEqual(len(events), len(errors))
        self.assertIn('requestClose', errors[0])
        v.setWeatherThunderCallback(None)
        v.setWeatherPreset(rr.WeatherPreset.Rain)
        v.updateWeather(.1)
        water = rr.defaultWaterBody(rr.WaterPreset.CalmLake)
        v.setWaterBody('python-water', water)
        self.assertIsNotNone(v.findWaterBody('python-water'))
        self.assertIsInstance(v.sampleWaterSurface([0., 0.]), rr.WaterSurfaceSample)
        v.render()
        self.assertIsInstance(json.loads(v.renderDiagnosticsJson()), dict)
        self.assertEqual(len(v.printWeatherDiagnostics()), 1)
        v.removeWaterBody('python-water')
        v.setWeatherPreset(rr.WeatherPreset.Clear)

    def test_removed_physics_object_invalidates_debug_handle(self):
        world = self.world
        obj = world.addSphere(.1, 1.)
        self.viewer.render()
        handle = self.viewer.debugObjectFor(obj)
        self.assertEqual(handle.sourceObject().getId(), obj.getId())
        self.assertEqual(handle.objectId(), obj.getId())
        world.removeObject(obj)
        self.viewer.render()
        with self.assertRaises(RuntimeError):
            handle.worldPosition()

    def test_cached_hdr_texture_lifetime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.hdr'
            path.write_bytes(b'#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n' + bytes([180, 120, 80, 128]) * 8)
            texture = self.viewer.acquireSceneHdrCubemapCached(str(path), 4)
            self.assertGreater(texture.id, 0)
            again = self.viewer.acquireSceneHdrCubemapCached(str(path), 4)
            self.assertEqual(texture.id, again.id)

    def test_swap_world_and_material_copy_lifetime(self):
        rr = self.rr
        with rr.RayraiWindow(raisim.World(), 64, 48, visible=False) as viewer:
            vertices = []
            for position in ([0., 0., 0.], [1., 0., 0.], [0., 1., 0.]):
                vertex = rr.Vertex()
                vertex.Position = position
                vertices.append(vertex)
            mesh = rr.OpenGLMesh(viewer, vertices, [0, 1, 2])
            material = mesh.material
            old_camera = viewer.getCamera()
            world = raisim.World()
            world.addGround()
            viewer.swapWorld(world)
            viewer.setRenderQualityPreset(rr.RenderQualityPreset.Fast)
            self.assertIs(viewer.world, world)
            self.assertTrue(viewer.render())
            self.assertEqual(old_camera.rtWidth(), 64)
        self.assertGreaterEqual(material.roughnessFactor, 0.)

    def test_async_color_and_depth(self):
        camera = self.viewer.getCamera()
        camera.setAsyncReadbackEnabled(True, 3)
        color = depth = None
        for _ in range(12):
            self.viewer.render()
            candidate = camera.readSceneColorRgbaAsync()
            if candidate is not None:
                color = candidate
            self.viewer.renderDepthPlaneDistance(camera)
            candidate = camera.readLinearDepthAsync()
            if candidate is not None:
                depth = candidate
        self.assertIsNotNone(color)
        self.assertIsNotNone(depth)
        self.assertEqual(color.shape, (120, 160, 4))
        self.assertEqual(depth.shape, (120, 160))
        camera.setAsyncReadbackEnabled(False)

    def test_collection_on_foreign_thread(self):
        # The final C++ state deleter must defer GL cleanup to the owner thread.
        viewer = self.rr.RayraiWindow(raisim.World(), 32, 32, visible=False)
        retained = [viewer.getCamera()]
        del viewer
        gc.collect()
        self.assertEqual(retained[0].getPosition().shape, (3,))
        thread = threading.Thread(target=retained.clear)
        thread.start()
        thread.join()
        gc.collect()
        # Any owner API call drains cleanup, including a render without polling.
        self.assertTrue(self.viewer.render(processEvents=False))

    def test_concurrent_callbacks_and_finalizers(self):
        probe = Path(__file__).with_name('rayrai_thread_probe.py')
        modes=['wrong_thread', 'foreign_gc', 'world_lock', 'finalizer', 'callback_render',
               'manual_locks','callback_cycle','target_lifetimes','contexts','visible_font']
        if os.environ.get('RAISIMPY_TEST_FRAME_FAULTS')=='1':
            modes.append('failure_paths')
        for mode in modes:
            with self.subTest(mode=mode):
                result = subprocess.run([sys.executable, '-X', 'faulthandler', str(probe), mode],
                                        capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(mode + ' passed', result.stdout)

    def test_render_override_shader_ownership(self):
        rr,v=self.rr,self.viewer
        overrides=rr.RenderOverrides()
        overrides.mesh=rr.Shader(v)
        gc.collect()
        self.assertIsInstance(overrides.mesh,rr.Shader)
        overrides.mesh.compile(
            '#version 330 core\nvoid main(){gl_Position=vec4(0.);}',
            '#version 330 core\nout vec4 c;void main(){c=vec4(1.);}')
        with rr.RayraiWindow(raisim.World(),32,32,visible=False) as other:
            with self.assertRaises(ValueError):
                other.renderWithExternalCamera(other.getCamera(),overrides)
        overrides.mesh=None

    def test_thread_rejection(self):
        errors = []
        def wrong_thread():
            try:
                self.viewer.render()
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=wrong_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn('thread', errors[0])

    def test_external_camera_color_and_depth(self):
        rr, v = self.rr, self.viewer
        properties = raisim.RGBCameraProperties()
        properties.width, properties.height = 80, 60
        properties.hFOV = 1.2
        properties.clipNear, properties.clipFar = .1, 50.
        camera = rr.Camera(v, properties)
        camera.position = [4., 3., 3.]
        camera.yaw = -135.
        camera.pitch = -25.
        camera.update(False)
        overrides = rr.RayraiWindow.RenderOverrides()
        v.renderWithExternalCamera(camera, overrides)
        self.assertEqual(camera.readRgba().shape, (60, 80, 4))
        v.renderDepthPlaneDistance(camera)
        depth = camera.readLinearDepth()
        self.assertEqual(depth.shape, (60, 80))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.isfinite(depth).all())

    def test_sync_rgb_and_depth_sensor_pose(self):
        world=raisim.World()
        robot=world.addArticulatedSystem(str(ROOT/'rsc/anymal_c/urdf/anymal_sensored.urdf'))
        sensors=robot.getSensorSet('depth_camera_front_camera_parent')
        with self.rr.RayraiWindow(world,32,32,visible=False) as viewer:
            camera=viewer.getCamera()
            for name in ('color','depth'):
                sensor=sensors.getSensor(name)
                self.rr.syncRaisimCameraPose(sensor,camera)
                np.testing.assert_allclose(camera.position,sensor.getPosition(),atol=1e-6)
                np.testing.assert_allclose(camera.front,sensor.getOrientation()[:,0],atol=1e-6)

    def test_multiple_contexts_and_close_invalidation(self):
        rr, first = self.rr, self.viewer
        with rr.RayraiWindow(self.world, 96, 64, visible=False,
                             shaderBinaryCacheDirectory='/tmp/raisimpy-rayrai-test-shaders') as second:
            second.setRenderQualityPreset(rr.RenderQualityPreset.Fast)
            camera = second.getCamera()
            for _ in range(2):
                with second.synchronousMeshLoading():
                    pass
            visual = second.addVisualBox('owned', .1, .1, .1)
            second.render()
            self.assertEqual(second.readRgba().shape, (64, 96, 4))
            first.render()
            self.assertEqual(first.readRgba().shape, (120, 160, 4))
            with self.assertRaises(ValueError):
                first.setTargetVisual(visual)
        for call in (camera.getPosition, visual.getPosition, second.render):
            with self.assertRaises(RuntimeError):
                call()
        second.close()
        self.assertTrue(first.render())

    def test_slider_and_cache_control(self):
        v = self.viewer
        slider = v.addSlider('test-slider', 0., 1., .3)
        self.assertAlmostEqual(v.getSlider(slider), .3, places=6)
        v.setSliderValue('test-slider', .6)
        self.assertAlmostEqual(v.getSliderValue('test-slider'), .6, places=6)
        v.invalidateExternalFrameCache(False)
        self.assertTrue(v.render())


if __name__ == '__main__':
    unittest.main()

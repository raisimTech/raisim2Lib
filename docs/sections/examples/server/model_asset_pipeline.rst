####################################
Server Example: Model Asset Pipeline
####################################

Overview
========
Demonstrates the mesh import and export pipeline. The example is grouped with
the server examples but does not start a server or open a window.

Use this example when preparing mesh assets for simulation. The RaiSim asset
APIs do the mesh processing, so application code only calls
``preprocessMesh``, ``addMesh``, and the export helpers.

Target
======
CMake target: ``model_asset_pipeline``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/model_asset_pipeline

On Windows, run ``model_asset_pipeline.exe`` instead.
This example is non-visual. It prints the preprocessed mesh path, whether the
preprocessing cache was hit, and the exported file paths.

Details
=======
- Runs mesh preprocessing before adding the mesh to the world.
- Adds the generated mesh asset through ``World::addMesh``.
- Exports mesh assets from the world to OBJ files.
- Keeps the workflow in ``addMesh`` and the asset APIs instead of duplicating
  mesh processing in application code.

Generated files
===============
The example writes its output to ``raisim_model_asset_pipeline_example`` in the
system temporary directory, for example:

.. code-block:: text

   /tmp/raisim_model_asset_pipeline_example

The output includes:

- a small source OBJ created by the example (``source_tetra.obj``),
- a preprocessed mesh in the ``cache`` directory,
- OBJ files exported from the RaiSim world (``exported_meshes``), and
- an XML world file that references the resulting scene (``scene.xml``).

API pattern
===========

.. code-block:: cpp

   raisim::Mesh::PreprocessOptions options;
   options.cacheDirectory = cacheDirectory;
   auto result = raisim::Mesh::preprocessMesh(inputObj, options);
   auto* mesh = world.addMesh(result.outputPath, mass, scale);
   auto exported = world.exportMeshAssetsToObj(outputDirectory);

The preprocessing result includes the output path, a content hash of the
source file, and whether an existing cached OBJ was reused (``cacheHit``), so a
toolchain can skip repeated work when the source mesh and options have not
changed.


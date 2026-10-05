############################
Rayrai Example: Nested Glass
############################

Overview
========
Traces refraction through a water-filled glass with an enclosed air bubble, an
empty hollow cup, a concave amber torus, and two overlapping tinted glass
solids. The media have explicit indices of refraction and dielectric
priorities, so overlapping interfaces resolve in a defined order. The default
backend uses Vulkan hardware ray queries when available and otherwise selects
the portable OpenGL tracer. It starts with 10 paths per frame and at most 10
bounces.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai_nested_glass_showcase.png
   :alt: Nested water, glass, and air with overlapping transparent solids
   :width: 100%

Target
======
CMake target: ``rayrai_nested_glass``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_nested_glass

On Windows, run ``rayrai_nested_glass.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.
Drag to look, use WASD and Space to move, and click an object to orbit it. The
**Geometry refraction** panel controls the backend, samples per frame, and
bounce count.

For a finite PNG capture or a stacked-media stress scene:

.. code-block:: bash

   ./build-examples/examples/rayrai_nested_glass --backend=portable --out=nested_glass.png --frames=24 --warmup=0
   ./build-examples/examples/rayrai_nested_glass --layers=4

Options
=======
* ``--backend=auto|portable|vulkan`` selects the tracing backend; ``vulkan``
  requires the Vulkan ray-query path.
* ``--samples=N`` (1 to 64) and ``--bounces=N`` (1 to 128) set the paths per
  frame and the bounce limit.
* ``--layers=N`` builds 1 to 16 concentric glass/water boxes; ``--layers=0``
  returns to the default composition.
* ``--out=FILE`` renders ``--frames`` frames (after ``--warmup`` frames) in a
  hidden window, saves a PNG, and exits. ``--benchmark`` runs a finite timing
  pass instead; ``--width``, ``--height``, and ``--msaa`` set the render size
  and MSAA level.

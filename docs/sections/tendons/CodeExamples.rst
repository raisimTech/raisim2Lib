#############################
Tendon code examples
#############################

These examples show the C++ needed for individual tendon behaviors. Every C++
block below is included directly from the runnable ``tendon_recipes.cpp`` source.
The program also checks the resulting motion, forces, and exported state, so
these are executable examples with physical acceptance checks.

Download :download:`tendon_recipes.cpp <../../code/tendons/tendon_recipes.cpp>`
and its :download:`CMakeLists.txt <../../code/tendons/CMakeLists.txt>` into the
same directory, or use ``docs/code/tendons`` in the distribution. For the larger
interactive and TCP scenes, see :doc:`Examples`. The recipes themselves run
headlessly and print measurements.

Build and run the recipes
=========================

From the ``raisim2Lib`` root on Linux:

.. code-block:: bash

   cmake -S docs/code/tendons -B /tmp/raisim-tendon-recipes \
     -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-20 \
     -DCMAKE_PREFIX_PATH="$PWD/raisim" \
     -DRAISIM_RECIPE_ACTIVATION_KEY=/path/to/activation.raisim
   cmake --build /tmp/raisim-tendon-recipes --parallel 12
   /tmp/raisim-tendon-recipes/tendon_recipes limit /path/to/activation.raisim
   ctest --test-dir /tmp/raisim-tendon-recipes -j 12 --output-on-failure

Set the prefix to the installed RaiSim package directory if its location differs.
On macOS and Windows, use the platform's compiler setup and build directory;
omit the Linux compiler override. Multi-configuration generators also need
``--config Release`` when building and ``-C Release`` for CTest. Their executable
normally resides in a configuration subdirectory such as ``Release``. On Windows,
make the RaiSim package's runtime DLLs discoverable as described in
:doc:`../Installation`.

The command syntax is ``tendon_recipes CASE [ACTIVATION_KEY|-] [EXPORT_XML]``.
Omit the key argument or use ``-`` for normal license discovery. The CMake key
setting is optional and affects CTest. The ``export`` case also requires an
absolute output filename in an existing directory, for example:

.. code-block:: bash

   /tmp/raisim-tendon-recipes/tendon_recipes export - /tmp/tendon-recipe.xml

Available cases are ``limit``, ``lock``, ``spring``, ``force``, ``servo``,
``wrap``, ``fixed``, ``coupling``, ``export``, and ``straight``. Each has a CTest
entry; the ``straight`` case runs the three straight-tendon examples below. Export
writes the supplied file, so choose an output path you intend to replace.

The fragments below are function bodies. To use one in your own program,
include ``<raisim/World.hpp>`` and ``<iostream>``, initialize your license before
creating the world, and provide this shared alias and stepping helper:

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [common]
   :end-before: // END [common]
   :dedent: 0

Limit the maximum separation
============================

Run ``tendon_recipes limit``. A 1 kg load starts at 1 m with an outward velocity
of 2 m/s. It stops at the 1.2 m upper bound. No spring or actuator is needed for
a hard length limit.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [limit]
   :end-before: // END [limit]
   :dedent: 2

For a minimum separation, set ``lowerLimit`` instead. To allow only an interval,
set both bounds, for example ``lowerLimit = 0.8`` and ``upperLimit = 1.2``.
``limitMargin`` moves the unilateral activation boundaries inward.

Lock the current length
=======================

Run ``tendon_recipes lock``. Refresh geometry and assign the current length to
both limits. The load can still swing: only its distance from the anchor is
constrained. This is the tendon equivalent of a bilateral straight wire.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [lock]
   :end-before: // END [lock]
   :dedent: 2

Equal bounds select a bilateral row; ``limitMargin`` is ignored for this lock.
``positionCorrection`` controls error correction, so setting a target different
from the current length does not teleport the body. To soften the lock, use
positive ``limitCompliance``; see :doc:`Physics` for its discrete meaning.

To release the lock later, reset only its limits. Include ``<limits>``:

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [release]
   :end-before: // END [release]
   :dedent: 2

Springs, drives, and other configured effects remain active. The ``lock`` case
also checks that the load can move beyond the former bound after release.

Suspend a load with a slack spring
==================================

Run ``tendon_recipes spring``. A vertical 1 kg load settles near
``1 + 9.81 / 200 = 1.04905`` m. Dry friction permits a small equilibrium band
around that value. Damping dissipates oscillation, while armature adds inertia
along the tendon coordinate.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [spring]
   :end-before: // END [spring]
   :dedent: 2

For a spring that also resists compression, set both spring endpoints to the
same rest length. The spring interval, hard limit, and actuator target are
independent. A slack spring can still have damping or friction force.

Apply a bounded tension command
===============================

Run ``tendon_recipes force``. The requested 5 N tension is limited to 4 N. In
zero gravity, the 1 kg load reaches -0.4 m/s after 0.1 s. Negative signed actuator
force pulls toward shorter length.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [force]
   :end-before: // END [force]
   :dedent: 2

A fresh ``Drive{}`` clears previous gains, targets, and activation filtering.
Calling ``setTension()`` alone preserves those settings. The actuator bounds do
not clamp passive springs, friction, or limit/coupling reactions.

Control the length with a servo
===============================

Run ``tendon_recipes servo``. A pull-only winch lifts the suspended load from a
1.2 m cable length toward 0.8 m. Feedforward compensates gravity in this vertical
configuration, while position/velocity feedback controls the transient.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [servo]
   :end-before: // END [servo]
   :dedent: 2

For a moving target, copy ``getDrive()``, update ``targetLength`` and optionally
``targetVelocity``, call ``setDrive()``, and then integrate each step.
``activationTime`` filters the feedforward force; it does not delay the target
or filter the feedback gains. The gravity-compensation value depends on the
actual routing and load and must be recomputed for other mechanisms.

Route over a cylinder and add a pulley branch
=============================================

Run ``tendon_recipes wrap``. The two main loads receive 4 N upward, and the
independent divisor-2 output branch receives 2 N. Their first-step vertical
velocities are approximately 0.004 and 0.002 m/s, respectively. Add ``<cmath>``
when copying this fragment.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [wrap]
   :end-before: // END [wrap]
   :dedent: 2

``{wheel}`` locates the routing guide at that body's origin. The cylinder axis
is local +Z; the body's orientation rotates it into world Y. The top side site
retains the overhead route. ``pulley(2)`` begins a separate branch rather than
creating a segment from ``right`` to the output anchor.

To wrap a sphere, use ``Path::sphere(centerSite, radius)`` with suitably placed
endpoints. The wrapping primitive is separate from collision geometry: the
static cylinder above exists to show the wheel. ``updateGeometry(true)`` exposes
the route for inspection; normal Rayrai rendering draws it automatically.

Drive a weighted joint coordinate
=================================

Run ``tendon_recipes fixed``. This small model supplies two independent
prismatic joints with unit moving masses. It needs no external URDF file.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [fixed_model]
   :end-before: // END [fixed_model]

The tendon drives ``x + 0.5*y`` toward 0.5. For these masses, zero initial state,
and no other forces, the resulting joint coordinates approach 0.4 and 0.2.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [fixed]
   :end-before: // END [fixed]
   :dedent: 2

Replace the model helper with your robot and use its revolute/prismatic joint
names. The tendon constrains or drives the weighted sum, leaving other motion
possible. Coefficients on revolute joints carry the chosen transmission units.
A fixed tendon has no spatial cable line; :doc:`Examples` includes a visible
mechanism combining a fixed actuator with spatial cables.

Couple two visible cable lengths
================================

Run ``tendon_recipes coupling``. Tendon A extends by 0.2 m and tendon B shortens
by 0.13 m. Creation poses establish the references, so initialize the bodies
before adding the tendons.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [coupling]
   :end-before: // END [coupling]
   :dedent: 2

Argument order matters: the first tendon is the dependent coordinate. This
example passes ``b`` first and ``a`` second. The coefficient array is constant
term first; ``{0., 2., 0.1, 0., 0.}`` instead means
``deltaB = 2*deltaA + 0.1*deltaA*deltaA``. Passing ``nullptr`` as the second tendon
locks the first relative to its creation reference plus the constant term.

Inspect, style, export, and reload
==================================

Run the ``export`` case with an absolute output filename. The file preserves
attachment names, appearance, limits, and the current force command. Add
``<stdexcept>`` when copying the fragment and supply ``exportPath`` as a string.

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [export]
   :end-before: // END [export]
   :dedent: 2

No custom drawing loop is required in Rayrai. Geometry refresh with ``true`` is
used here only to inspect segment coordinates. Alpha zero hides the cable while
retaining its physics; ``setEnabled(false)`` disables both its physics and its
drawing. Export preserves the current command, not a future C++ control schedule.
See :doc:`Reference` for XML attributes and embedded-URDF sidecars.

Straight connections through the tendon API
===========================================

Run ``tendon_recipes straight`` to execute these three independent worlds.
Each connection is a two-site spatial tendon. Here a static sphere supplies the
anchor; a null-object site can instead attach directly to the world frame.

Equal limits lock separation in both directions and stop the initial outward
velocity:

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [straight_lock]
   :end-before: // END [straight_lock]
   :dedent: 4

``getLength()`` is the current coordinate; ``getProperties()`` holds the bounds.
A pull-only spring is slack below its rest length. The implicit update with
0.1 m initial stretch and 100 N/m stiffness gives approximately 10 N of tension:

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [straight_spring]
   :end-before: // END [straight_spring]
   :dedent: 4

The same tendon type supports a commanded tension without a spring or limit:

.. literalinclude:: ../../code/tendons/tendon_recipes.cpp
   :language: cpp
   :start-after: // BEGIN [straight_force]
   :end-before: // END [straight_force]
   :dedent: 4

For a compression-only spring, set ``springLower = L`` and
``springUpper = std::numeric_limits<double>::infinity()``. For a bilateral
spring, set both endpoints to L. See :doc:`../Constraints` for the complete
mapping from the legacy wire API.

Python uses the same model
==========================

``raisimpy.Tendon`` exposes the same properties, routes, drives, and scalar
inspection methods; point and joint gradients are available only in C++.
Properties and drive getters return copies; apply edits with the corresponding
setter so validation and wake-up happen. The world owns tendons
and couplings; do not use their Python handles after explicit removal or after
removing an attached object.

.. code-block:: python

   import raisimpy as raisim

   # Configure the RaiSim activation key before creating the world if needed.
   world = raisim.World()
   world.setTimeStep(0.001)
   load = world.addSphere(0.05, 1.0)
   load.setPosition(0.0, 0.0, 1.0)
   Site = raisim.Tendon.Site
   Path = raisim.Tendon.PathElement
   properties = raisim.Tendon.Properties()
   properties.upperLimit = 1.0
   properties.width = 0.01
   properties.color = [0.1, 0.9, 0.6, 1.0]
   cable = world.addSpatialTendon("cable", [
       Path.via(Site(position=[0.0, 0.0, 2.0])),
       Path.via(Site(load)),
   ], properties)
   cable.setTension(2.0)
   for _ in range(1000):
       world.integrate()
   cable.updateGeometry()
   print(cable.getLength(), cable.getTension())
   properties = cable.getProperties()
   properties.lowerLimit = properties.upperLimit = cable.getLength()
   cable.setProperties(properties)
   world.removeTendon(cable)

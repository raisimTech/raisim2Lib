CustomLengthConstraint
==========================

``CustomLengthConstraint`` allows users to apply custom constraints.
Users set the tension between two points using ``setTension()``.

An example is shown below:

.. code-block:: cpp

  auto wire7 = world.addCustomWire(pin7, 0, {0,0,0}, anymalB, 0, {0., 0, 0}, 2.0);
  wire7->setTension(400);

The following code results in:

.. image:: ../image/customWire.gif

You can find a runnable example `here <https://github.com/raisimTech/raisim2Lib/blob/master/examples/src/server/length_constraints_newtons_cradle.cpp>`__.


API
----------------

.. doxygenclass:: raisim::CustomLengthConstraint
   :members:

#############################
Joint Damping and Friction
#############################

Every revolute and prismatic joint can have viscous damping and Coulomb friction, and a spherical
joint can have viscous damping. They model bearings, seals and gears. They are passive: they only
remove energy, they are not part of the generalized force you set, and they act whether or not the
joint is controlled.

Set them with the standard URDF ``<dynamics>`` attributes. Both default to zero:

.. code-block:: xml

    <joint name="knee" type="revolute">
        <parent link="thigh"/>
        <child link="shank"/>
        <axis xyz="0 1 0"/>
        <dynamics damping="0.5" friction="2.0"/>
    </joint>

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Joint type
     - ``damping`` :math:`b`
     - ``friction`` :math:`\tau_c`
   * - revolute
     - :math:`Nm\,s/rad`
     - :math:`Nm`
   * - prismatic
     - :math:`N\,s/m`
     - :math:`N`
   * - spherical
     - :math:`Nm\,s/rad`, on each of the three angular velocity components
     - not supported (ignored)

The MJCF loader reads the joint ``damping`` attribute but not ``frictionloss``.

Both are measured at the joint, i.e., after any gear. If the joint is driven by an actuator, the
actuator's ``output_damping`` and ``output_friction`` add to them (see :doc:`../Actuators`).

Damping
=============================
Damping is the torque :math:`\tau_d = -b\,u` on the joint velocity :math:`u` (relative to the
parent body, see :ref:`articulated_systems`). RaiSim integrates it implicitly with the
trapezoidal rule, i.e., at the average velocity of the time step :math:`\Delta t`:

.. math::

  \tau_d = -b\,\frac{u_t + u_{t+1}}{2}.

For example, a free joint with inertia :math:`I` under a constant torque :math:`\tau` and no gravity
accelerates as

.. math::

  I\,\frac{u_{t+1} - u_t}{\Delta t} = \tau - b\,\frac{u_t + u_{t+1}}{2}

and converges to the velocity :math:`\tau / b`. The implicit integration keeps a joint with a large
damping stable at any time step. The d gain of the PD controller is integrated the same way. The
difference is that damping pulls the velocity to zero, while the d gain pulls it to the velocity
target.

To change the damping at runtime, call ``setJointDamping()`` with one coefficient per degree of
freedom (``getDOF()`` entries). The new values act from the next step. Leave the six entries of a
floating base at zero: they are not joints, and their damping would not be integrated implicitly.

The URDF ``effort`` limit bounds only the commanded torque (PD plus feedforward). The damping torque
is passive and is never clipped. Actuator torques (see :doc:`../Actuators`) are not bounded by the
effort limit either: they are limited by the operating regions of their motors and are added after
the effort clamp.

Friction
=============================
Friction is Coulomb friction with stiction. It opposes the motion with the constant torque
:math:`\tau_c`, and it holds a joint at rest as long as the other torques on it (actuation,
gravity, contacts and the motion of the other bodies) stay below :math:`\tau_c`:

.. math::

  \tau_f = -\tau_c\,\mathrm{sgn}(u_{t+1}) \ \text{if} \ u_{t+1} \neq 0,
  \qquad |\tau_f| \le \tau_c \ \text{if} \ u_{t+1} = 0.

Once the other torques exceed :math:`\tau_c`, the joint accelerates with their sum minus
:math:`\tau_c`. For the free joint above, that is
:math:`I\,(u_{t+1} - u_t)/\Delta t = \tau - \tau_c\,\mathrm{sgn}(u_{t+1})`.

RaiSim solves friction as a joint impulse bounded by :math:`\tau_c\,\Delta t` in the contact
solver, together with the contacts and the joint limits. A joint therefore stops exactly instead
of chattering around zero velocity, and the friction torque is consistent with the contact forces
on the robot. A robot with friction adds one row to the contact
solver; robots without friction do not pay for it.

The joint's own friction is set in the model file; there is no C++ setter for it. The
``output_friction`` of an actuator can be changed at runtime by editing the definitions from
``getActuators()`` and passing them to ``setActuators()``.

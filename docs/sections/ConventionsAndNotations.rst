#############################
Conventions and Notations
#############################


Kinematics
===============

1. A quaternion is written as :math:`[w, x, y, z]^T`, with the scalar part first. For a rotation by the angle :math:`\theta` about the unit axis :math:`\boldsymbol{a}`, :math:`w = \cos(\theta/2)` and :math:`[x, y, z]^T = \sin(\theta/2)\,\boldsymbol{a}`.
2. Unless explicitly stated otherwise, all quantities in RaiSim are expressed in the world frame.
3. The world frame is an inertial frame: it neither moves nor accelerates.
4. For articulated systems, the base angular velocity is defined in the world frame (in contrast to many other simulators which utilize the base frame). For further details, please refer to :ref:`articulated_systems`.
5. Quantities use SI units (m, kg, s, rad). The default gravity is :math:`[0, 0, -9.81]^T` m/s², so the world :math:`z` axis points up.


A position vector in a Cartesian coordinate system is represented as :math:`{}^W\boldsymbol{r}`, where :math:`^W` denotes the frame in which the vector is expressed.
In this document, :math:`W` represents the **world frame**.

A position vector can be expressed in any reference frame. To transform the vector representation between frames, a linear transformation is employed as follows:

.. math::

  {}^B\boldsymbol{r} &= {}^{BW\!}\boldsymbol{R}\, {}^W\boldsymbol{r},\\
  {}^W\boldsymbol{r} &= {}^{WB\!}\boldsymbol{R}\, {}^B\boldsymbol{r}.

In this document, the symbol :math:`^B` denotes the **body frame**, which is assigned to every rigid body within RaiSim.
The term :math:`{}^{WB}\boldsymbol{R}` refers to the **body rotation matrix**.
**Note that the body rotation matrix maps a vector expressed in the body frame to its corresponding vector in the world frame.**


Dynamics
===============

The pages on dynamics share one notation.

* Bold symbols are vectors and matrices; italic symbols are scalars, including the entries of a
  vector: :math:`u_j` is the entry of :math:`\boldsymbol{u}` for joint :math:`j`.
* :math:`\boldsymbol{q}` and :math:`\boldsymbol{u}` are the generalized coordinate and the
  generalized velocity of an articulated system, with :math:`|\boldsymbol{u}|` entries (see
  :doc:`articulated_system/StateAndKinematics`). Joint velocities are entries of
  :math:`\boldsymbol{u}`.
* :math:`\boldsymbol{\tau}` is a generalized force, :math:`\boldsymbol{M}` the mass matrix and
  :math:`\boldsymbol{h}` the nonlinear term (see :doc:`articulated_system/DynamicsAndControl`).
* :math:`\boldsymbol{J}` is a Jacobian, :math:`\boldsymbol{J}\boldsymbol{u} = \boldsymbol{v}` for
  the velocity :math:`\boldsymbol{v}` of a point.
* :math:`\Delta t` is the time step, and the subscripts :math:`t` and :math:`t+1` mark the beginning
  and the end of a step.
* :math:`\boldsymbol{\lambda}` is an impulse, which acts as the generalized impulse
  :math:`\boldsymbol{J}^T\boldsymbol{\lambda}`. :math:`\boldsymbol{\lambda}_c` are the impulses of
  the contact solver (contacts, joint limits and joint friction).
* For a joint, :math:`b` is the damping and :math:`\tau_c` the friction (see
  :doc:`articulated_system/JointDampingAndFriction`); for an actuator, :math:`G` is the gear ratio
  (see :doc:`Actuators`).

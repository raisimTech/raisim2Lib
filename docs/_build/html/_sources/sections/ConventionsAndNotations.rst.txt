#############################
Conventions and Notations
#############################


Kinematics
===============

1. A quaternion is defined as :math:`\mathbf{q} = [w, x, y, z]^T`, where :math:`[x, y, z]^T` denotes the scaled axis of rotation and :math:`w` represents the cosine of the half-angle.
2. Unless explicitly stated otherwise, all quantities in RaiSim are expressed in the world frame.
3. The world frame is defined as an inertial frame with zero acceleration and zero velocity.
4. For articulated systems, the base angular velocity is defined in the world frame (in contrast to many other simulators which utilize the base frame). For further details, please refer to :ref:`articulated_systems`.


A position vector in a Cartesian coordinate system is represented as :math:`^Wr`, where :math:`^W` denotes the frame in which the vector is expressed.
In this document, :math:`W` represents the **world frame**, an inertial frame (zero acceleration).
Unless otherwise explicitly stated, all vector quantities in RaiSim are expressed in the world frame.

A position vector can be expressed in any reference frame. To transform the vector representation between frames, a linear transformation is employed as follows:

.. math::

  \begin{eqnarray}
    ^Br&= {}^{BW\!}R ^Wr,\\
    ^Wr&= {}^{WB\!}R ^Br.
  \end{eqnarray}

In this document, the symbol :math:`^B` denotes the **body frame**, which is assigned to every rigid body within RaiSim.
The term :math:`^{WB}R` refers to the **body rotation matrix**.
**Note that the body rotation matrix maps a vector expressed in the body frame to its corresponding vector in the world frame.**

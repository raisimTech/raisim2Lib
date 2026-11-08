# Conventions and Notations

## Kinematics

1. A quaternion is written as $[w, x, y, z]^T$, with the scalar part first. For a rotation by the angle $\theta$ about the unit axis $\boldsymbol{a}$, $w = \cos(\theta/2)$ and $[x, y, z]^T = \sin(\theta/2)\,\boldsymbol{a}$.
2. Unless explicitly stated otherwise, all quantities in RaiSim are expressed in the world frame.
3. The world frame is an inertial frame: it neither moves nor accelerates.
4. For articulated systems, the base angular velocity is defined in the world frame (in contrast to many other simulators which utilize the base frame). For further details, please refer to [Joints](./articulated_system/StateAndKinematics.md#joints).
5. Quantities use SI units (m, kg, s, rad). The default gravity is $[0, 0, -9.81]^T$ m/s², so the world $z$ axis points up.

A position vector in a Cartesian coordinate system is represented as ${}^W\boldsymbol{r}$, where $^W$ denotes the frame in which the vector is expressed.
In this document, $W$ represents the **world frame**.

A position vector can be expressed in any reference frame. To transform the vector representation between frames, a linear transformation is employed as follows:

$$
\begin{aligned}
{}^B\boldsymbol{r} &= {}^{BW\!}\boldsymbol{R}\, {}^W\boldsymbol{r},\\
{}^W\boldsymbol{r} &= {}^{WB\!}\boldsymbol{R}\, {}^B\boldsymbol{r}.
\end{aligned}
$$

In this document, the symbol $^B$ denotes the **body frame**, which is assigned to every rigid body within RaiSim.
The term ${}^{WB}\boldsymbol{R}$ refers to the **body rotation matrix**.
**Note that the body rotation matrix maps a vector expressed in the body frame to its corresponding vector in the world frame.**

## Dynamics

The pages on dynamics share one notation.

- Bold symbols are vectors and matrices; italic symbols are scalars, including the entries of a
  vector: $u_j$ is the entry of $\boldsymbol{u}$ for joint $j$.
- $\boldsymbol{q}$ and $\boldsymbol{u}$ are the generalized coordinate and the
  generalized velocity of an articulated system, with $|\boldsymbol{u}|$ entries (see
  [State and Kinematics](./articulated_system/StateAndKinematics.md)). Joint velocities are entries of
  $\boldsymbol{u}$.
- $\boldsymbol{\tau}$ is a generalized force, $\boldsymbol{M}$ the mass matrix and
  $\boldsymbol{h}$ the nonlinear term (see [Dynamics and Control](./articulated_system/DynamicsAndControl.md)).
- $\boldsymbol{J}$ is a Jacobian, $\boldsymbol{J}\boldsymbol{u} = \boldsymbol{v}$ for
  the velocity $\boldsymbol{v}$ of a point.
- $\Delta t$ is the time step, and the subscripts $t$ and $t+1$ mark the beginning
  and the end of a step.
- $\boldsymbol{\lambda}$ is an impulse, which acts as the generalized impulse
  $\boldsymbol{J}^T\boldsymbol{\lambda}$. $\boldsymbol{\lambda}_c$ are the impulses of
  the contact solver (contacts, joint limits and joint friction).
- For a joint, $b$ is the damping and $\tau_c$ the friction (see
  [Joint Damping and Friction](./articulated_system/JointDampingAndFriction.md)); for an actuator, $G$ is the gear ratio
  (see [Actuators](./Actuators.md)).

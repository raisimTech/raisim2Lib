# Dynamics and Control

## Dynamics

All force and torque acting on the system can be represented as a single vector in the generalized velocity space.
This representation is called **generalized force** $\boldsymbol{\tau}$.
Just like in a Cartesian coordinate (i.e., x, y, z axes), the power exerted by an articulated system is computed as a dot product of generalized force and generalized velocity (i.e., $\boldsymbol{u}\cdot\boldsymbol{\tau}$).

We can also combine the mass and inertia of the whole articulated system and represent them in a single matrix.
This matrix is called **mass matrix** or **inertia matrix** and denoted by $\boldsymbol{M}$.
A mass matrix represents how much the articulated system resists change in generalized velocities.
Naively speaking, a large mass matrix means that the articulated system experiences a low velocity change for a given generalized force.

The total kinetic energy of the system is computed as $\frac{1}{2}\boldsymbol{u}^T\boldsymbol{M}\boldsymbol{u}$.
This quantity can be obtained by `getKineticEnergy()`.

The total potential energy due to the gravity is a sum of $m\,g\,z$ (mass, gravitational acceleration and height) for all bodies.
This quantity can be obtained by `getPotentialEnergy(gravity)`.
The gravity vector must be passed because only the world stores it (e.g., `world.getGravity()`).

The equation of motion of an articulated system is shown below:

$$
\begin{equation}
   \boldsymbol{\tau} = \boldsymbol{M}(\boldsymbol{q})\dot{\boldsymbol{u}} + \boldsymbol{h}(\boldsymbol{q}, \boldsymbol{u}).
\end{equation}
$$

Here $\boldsymbol{h}$ is called a **non-linear term**.
There are three sources of force that contribute to the non-linear term: gravity, Coriolis, and centrifugal force.
It is rarely useful to compute the gravity contribution to the nonlinear term alone.
If it is needed, evaluate the nonlinear term at zero generalized velocity, e.g., on a copy of the robot in another world: then the Coriolis and centrifugal contributions are zero.

The following methods are used to obtain dynamic quantities

- `getMassMatrix()` (it includes the rotor inertia on its diagonal)
- `getNonlinearities(gravity)`
- `getInverseMassMatrix()` (call `getMassMatrix()` first)

## Inverse Dynamics

RaiSim can compute inverse dynamics using the recursive Newton-Euler algorithm.
It is the only option for computing the force and torque acting at joints.
Joint force/torque are the sum of the constraint joint force/torque and actuation force/torque.
For example, a revolute joint constrains motions in 5 degrees of freedom, which means that there are 5-dimensional constraint forces/torque and 1-dimensional joint actuation torque acting at a revolute joint.

In minimal coordinate simulation (such as RaiSim), these constraint forces/torques are not computed in the simulation loop.
These forces/torques can be computed after a simulation loop using the inverse dynamics pipeline.

To enable inverse dynamics, call `raisim::ArticulatedSystem::setComputeInverseDynamics(true)`.
**This flag is set automatically if the robot has an IMU sensor**.
Note that the inverse dynamics pipeline will slow down the simulation by about 10%.

After a simulation loop, you can call `raisim::ArticulatedSystem::getForceAtJointInWorldFrame()` and `raisim::ArticulatedSystem::getTorqueAtJointInWorldFrame()` to get forces and torques acting at the specified joint.

Assuming that there are no joint position/velocity limit forces acting at the joint, you can compute the joint actuation as a dot product of the joint axis and the joint torque.
The `inverse_dynamics` example ([Server Example: Inverse Dynamics](../examples/server/inverse_dynamics.md)) does this for ANYmal and compares the result with the applied generalized force.

## PD Controller

When naively implemented, a PD controller can often make a robot unstable.
However, this is often less problematic for robotics since this instability is also present in real systems (discrete-time control systems).

For other applications like animation and graphics, it is often desirable to have a PD controller that stays stable with a large time step.
Therefore, the built-in PD controller is integrated implicitly and stays stable with much larger time steps and gains than a naive implementation.

The built-in PD torque and the feedforward generalized force are clamped together to the joint
actuation limits (URDF `effort` or `setActuationLimits()`). Joint damping and actuator torques
are added after this clamp and are not clipped by it.
Because PD is integrated implicitly, the clamp bounds only its explicit part, and
`getGeneralizedForce()` reports the applied PD torque only approximately (see
[Joint torque of a simulation step](#joint-torque-of-a-simulation-step)). The built-in PD torque is not an actuator torque, so the
motor operating regions of actuators do not clip it (see [Actuators](#actuators) below).

To use this PD controller, set the desired control gains first:

```cpp
Eigen::VectorXd pGain(robot->getDOF()), dGain(robot->getDOF());
pGain<< ...; // set your proportional gain values here
dGain<< ...; // set your differential gain values here
robot->setPdGains(pGain, dGain);
```

Note that **the dimension of the pGain vector is the same as that of the generalized velocity NOT that of the coordinate**.
For a floating base, the first six gains are forced to zero.
`setPdGains()` also switches the control mode to `ControlMode::PD_PLUS_FEEDFORWARD_TORQUE`.

Finally, the target position and the velocity can be set as follows:

```cpp
Eigen::VectorXd pTarget(robot->getGeneralizedCoordinateDim()), vTarget(robot->getDOF());
pTarget<< ...; // set your position target
vTarget<< ...; // set your velocity target
robot->setPdTarget(pTarget, vTarget);
```

Here, **the dimension of the pTarget vector is the same as that of the generalized coordinate NOT that of the velocity**.
This can be confusing and may seem inconsistent.
However, this is a valid convention.
The only reason the two dimensions differ is the quaternion representation.
The quaternion target is represented by a quaternion whereas the virtual spring stiffness between the two orientations can be represented by a 3D vector, which is composed of motions in each angular velocity components.

A feedforward force term can be added by `setGeneralizedForce()` if desired.
This term is set to zero by default.
Note that this value is stored in the class instance and does not change unless the user specifies it so.
If this feedforward force should be applied for a single time step, set it to zero in the subsequent control loop (after the `integrate()` call of the world).

The theory of the implemented PD controller can be found in chapter 1.2 of this [article](https://www.overleaf.com/read/dbqbgcnhzykq).
This document is intended for advanced users and is not required to use RaiSim.

## Actuators

The URDF `effort` limit is a constant bound per joint on the feedforward generalized force plus
the built-in PD torque; joint damping is passive and is not clipped. Actuator torques set with
`setActuatorTorque()` or `setActuatorTorques()` are clipped separately, to the motor operating
regions (MOR) of their motors, mapped to the joints (with their couplings), and added after the
effort clamp. Effort limits do not clip actuator torques, and MOR does not clip
`setGeneralizedForce()` or the built-in PD controller. The combined generalized force is not
clamped again. To keep a PD controller inside the motor operating regions, compute its torques
explicitly and send them with `setActuatorTorques()`. See [Actuators](../Actuators.md).

## Apply External Forces/Torques

The following two methods are used to apply external force and torque respectively

- `setExternalForce`
- `setExternalTorque`

You will find the above methods in the [ArticulatedSystem API](./API.md).

<a id="joint-torque-of-a-step"></a>

## Joint torque of a simulation step

This section gives exactly what a simulation step of length $\Delta t$ computes for an
articulated system. Bold symbols are vectors with one entry per generalized velocity (one per
revolute or prismatic joint, three per spherical joint and six for a floating base), and a product
of two such vectors is taken entry by entry. $\boldsymbol{q}_t$ and $\boldsymbol{u}_t$
are the generalized coordinate and velocity at the beginning of the step, and
$\boldsymbol{u}_{t+1}$ is the velocity at its end. For a spherical joint,
$\boldsymbol{q}_{ref} - \boldsymbol{q}_t$ stands for the rotation vector of
$\boldsymbol{q}_t^{-1} \boldsymbol{q}_{ref}$, the shorter rotation (an angle of at most
$\pi$): a quaternion and its negative are the same orientation, as the joint state or as the
target.

**1. The command, clamped to the actuation limits.**

$$
\begin{aligned}
\boldsymbol{\tau}_{PD}  &= \boldsymbol{k}_p\,(\boldsymbol{q}_{ref} - \boldsymbol{q}_t)
                           + \boldsymbol{k}_d\,(\boldsymbol{u}_{ref} - \boldsymbol{u}_t) \\
\boldsymbol{\tau}_{cmd} &= \mathrm{clamp}\bigl(\boldsymbol{\tau}_{ff} + \boldsymbol{\tau}_{PD},\;
                           \boldsymbol{\tau}_{lower},\; \boldsymbol{\tau}_{upper}\bigr)
\end{aligned}
$$

$\boldsymbol{\tau}_{ff}$ is the force set with `setGeneralizedForce()`,
$\boldsymbol{k}_p$ and $\boldsymbol{k}_d$ are the gains of `setPdGains()`, and
$\boldsymbol{q}_{ref}$ and $\boldsymbol{u}_{ref}$ are the targets of `setPdTarget()`.
In the `FORCE_AND_TORQUE` control mode, $\boldsymbol{k}_p = \boldsymbol{k}_d = 0$ here and
in step 4. The bounds are $\mp$ the URDF `effort` or those of `setActuationLimits()`; a
joint without them is not clamped.

**2. The actuator torques.** Take an actuator with the gear ratio $G$, the driven joint
$j$, the couplings $(c, r_c)$ and the torque $\tau_a$ set with
`setActuatorTorques()` (see [Actuators](../Actuators.md)). Its motor has the stall torque
$\tau_{stall} = K_t V_{bus} / R$, the back-EMF damping $b = K_t / (R K_v)$ and the
peak torque $\tau_{peak}$. With $u_{t,j}$ the entry of $\boldsymbol{u}_t$ for
joint $j$:

$$
\begin{aligned}
\omega_m     &= G\,u_{t,j} + \textstyle\sum_c r_c\,u_{t,c} \\
\tau_{m,min} &= \mathrm{clamp}\bigl(-\tau_{stall} - b\,\omega_m,\; -\tau_{peak},\; \tau_{peak}\bigr) \\
\tau_{m,max} &= \mathrm{clamp}\bigl(\phantom{-}\tau_{stall} - b\,\omega_m,\; -\tau_{peak},\; \tau_{peak}\bigr) \\
\tau_m       &= \mathrm{clamp}\bigl(\tau_a / G,\; \tau_{m,min},\; \tau_{m,max}\bigr)
\end{aligned}
$$

The actuator adds $G\,\tau_m$ to joint $j$ and $r_c\,\tau_m$ to each coupled
joint $c$, and $\boldsymbol{\tau}_{act}$ is the sum over all actuators. With
`setMotorOperatingRegionEnforced(false)`, $\tau_m = \tau_a / G$. The actuation limits do
not clip actuator torques.

**3. The explicit torque.** Everything evaluated at the beginning of the step is summed, and the
sum is not clamped:

$$
\boldsymbol{\tau}_{exp} = \boldsymbol{\tau}_{cmd} - (\boldsymbol{b}_{joint} + \boldsymbol{b}_{out})\,\boldsymbol{u}_t
                          + \boldsymbol{\tau}_{act} + \boldsymbol{k}_s\,(\boldsymbol{q}_s - \boldsymbol{q}_t)
                          + \boldsymbol{J}^T \boldsymbol{F}_{ext}
$$

$\boldsymbol{b}_{joint}$ is the joint damping (URDF `damping` or `setJointDamping()`, see
[Joint Damping and Friction](./JointDampingAndFriction.md)) and $\boldsymbol{b}_{out}$ the output damping of the joint's
actuator. $\boldsymbol{k}_s$ and $\boldsymbol{q}_s$ are the stiffness and the rest
position of the joint springs (URDF `<dynamics stiffness spring_mount>` or `addSpring()`, see
`getSprings()`). $\boldsymbol{J}^T \boldsymbol{F}_{ext}$ are the forces and torques of
`setExternalForce()`, `setExternalTorque()` and `setConstraintForce()`, mapped through their
Jacobians.

**4. The velocity.** The PD gains, the damping and the springs are integrated implicitly: the
diagonal $\boldsymbol{w}$ is added to the mass matrix $\boldsymbol{M}$
(`getMassMatrix()`, which includes the rotor inertia). With the nonlinear term
$\boldsymbol{h}$ (gravity, Coriolis and centrifugal forces, `getNonlinearities()`), the
articulated-body algorithm computes the velocity without contacts, $\boldsymbol{u}^*$, and the
contact solver adds the impulses $\boldsymbol{\lambda}_c$ of the contacts, the joint limits
and the joint friction through their Jacobian $\boldsymbol{J}_c$ and the same matrix:

$$
\begin{aligned}
\boldsymbol{w}         &= \frac{\Delta t}{2}\,(\boldsymbol{b}_{joint} + \boldsymbol{b}_{out} + \boldsymbol{k}_d)
                          + \frac{\Delta t^2}{4}\,(\boldsymbol{k}_p + \boldsymbol{k}_s) \\
\tilde{\boldsymbol{M}} &= \boldsymbol{M} + \mathrm{diag}(\boldsymbol{w}) \\
\boldsymbol{u}^*       &= \boldsymbol{u}_t + \Delta t\;\tilde{\boldsymbol{M}}^{-1}\,(\boldsymbol{\tau}_{exp} - \boldsymbol{h}) \\
\boldsymbol{u}_{t+1}   &= \boldsymbol{u}^* + \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}_c^T \boldsymbol{\lambda}_c
\end{aligned}
$$

A position-limit impulse acts on a joint outside its range. A velocity-limit impulse acts on a
joint whose entry of $\boldsymbol{u}^*$ exceeds its velocity limit, unless a position-limit
impulse acts on that joint. A joint-friction impulse acts on every revolute, prismatic or spherical
joint with friction (see [Joint Damping and Friction](./JointDampingAndFriction.md)).

**5. The applied joint torque.** With $\Delta\boldsymbol{u} = \boldsymbol{u}_{t+1} - \boldsymbol{u}_t$,
moving the implicit part to the right-hand side gives the equation of motion of the step:

$$
\boldsymbol{M}\,\frac{\Delta\boldsymbol{u}}{\Delta t} + \boldsymbol{h}
  = \boldsymbol{\tau}_{exp} - \boldsymbol{w}\,\frac{\Delta\boldsymbol{u}}{\Delta t}
    + \frac{\boldsymbol{J}_c^T \boldsymbol{\lambda}_c}{\Delta t}
$$

The torque on the joints over the step is therefore, term by term:

| Source | Torque over the step |
| --- | --- |
| feedforward and PD | $\boldsymbol{\tau}_{cmd} - \bigl(\tfrac{1}{2} \boldsymbol{k}_d + \tfrac{\Delta t}{4} \boldsymbol{k}_p\bigr)\,\Delta\boldsymbol{u}$ |
| damping | $-\tfrac{1}{2}\,(\boldsymbol{b}_{joint} + \boldsymbol{b}_{out})\,(\boldsymbol{u}_t + \boldsymbol{u}_{t+1})$ (trapezoidal rule) |
| actuators | $\boldsymbol{\tau}_{act}$, constant over the step |
| springs | $\boldsymbol{k}_s\,(\boldsymbol{q}_s - \boldsymbol{q}_t) - \tfrac{\Delta t}{4}\,\boldsymbol{k}_s\,\Delta\boldsymbol{u}$ |
| external forces | $\boldsymbol{J}^T \boldsymbol{F}_{ext}$ |
| joint friction | $\boldsymbol{\lambda}_f / \Delta t$, with $\lvert\lambda_{f,j}\rvert \le (\tau_{c,joint,j} + \tau_{c,out,j})\,\Delta t$ for a revolute or prismatic joint $j$ and $\lVert\boldsymbol{\lambda}_{f,j}\rVert \le \tau_{c,j}\,\Delta t$ for the three entries of a spherical joint $j$ |
| contacts and joint limits | the rest of $\boldsymbol{J}_c^T \boldsymbol{\lambda}_c / \Delta t$ |

The friction impulse $\boldsymbol{\lambda}_f$, the joints' share of
$\boldsymbol{J}_c^T \boldsymbol{\lambda}_c$, stops a joint if the bound allows it, and
otherwise opposes the motion with the bound. $\boldsymbol{\tau}_{c,joint}$ is the joint
friction (see [Joint Damping and Friction](./JointDampingAndFriction.md)) and $\boldsymbol{\tau}_{c,out}$ the output
friction of the joint's actuator. For a spherical joint, the bound applies to the magnitude of the
joint's three entries, so its friction torque is the same in every direction.

Only the explicit part of the feedforward and PD torque is clamped. The implicit part is not, so
the applied PD torque can exceed the actuation limits by
$\bigl(\tfrac{1}{2} \boldsymbol{k}_d + \tfrac{\Delta t}{4} \boldsymbol{k}_p\bigr)\,|\Delta\boldsymbol{u}|$,
e.g., when an impact changes the velocity within one step. Without the clamp, the feedforward and
PD torque is

$$
\boldsymbol{\tau}_{ff} + \boldsymbol{k}_p \left(\boldsymbol{q}_{ref} - \boldsymbol{q}_t - \frac{\Delta t}{4}\,\Delta\boldsymbol{u}\right)
          + \boldsymbol{k}_d \left(\boldsymbol{u}_{ref} - \frac{1}{2}\,(\boldsymbol{u}_t + \boldsymbol{u}_{t+1})\right)
$$

**6. The position.** The position is integrated with the average velocity $\bar{\boldsymbol{u}}$
of the scheme set with `setIntegrationScheme()` (a quaternion with the same angular velocity):

$$
\boldsymbol{q}_{t+1} = \boldsymbol{q}_t + \Delta t\,\bar{\boldsymbol{u}}
$$

| Integration scheme | $\bar{\boldsymbol{u}}$ |
| --- | --- |
| `SEMI_IMPLICIT` (default) | $\boldsymbol{u}_{t+1}$ |
| `TRAPEZOID` | $\tfrac{1}{2}(\boldsymbol{u}_t + \boldsymbol{u}_{t+1})$ |
| `EULER` | $\boldsymbol{u}_t$ |

The torque is the same for the three. `SEMI_IMPLICIT` moves the robot with the velocity the contact
solver leaves, so a foot that an impact stops stays at the surface. `TRAPEZOID` follows a body
under a constant force exactly (a thrown ball), but under forces that change with the state (gravity on a joint, springs, the PD
torque, the gyroscopic force) it gains energy: an undamped pendulum's swing grows by about 9% in 20 s
at $\Delta t$ = 1 ms, where `SEMI_IMPLICIT` keeps it within 0.1%. On an impact, `TRAPEZOID` also
goes into the ground by up to half the step's travel. `RUNGE_KUTTA_4` applies the velocity change of the contact
solver at the beginning of the step and then evaluates steps 1-4 without contacts at each of its
four stages, at the state of the stage; the actuator bounds use the motor speeds of the stage, and
`getActuatorStates()` describes the last stage. In the PD control mode, `SEMI_IMPLICIT` replaces
`EULER` and `RUNGE_KUTTA_4` once the robot has been in contact (from the step after its first
contact), until its state is set again with `setState()`, `setGeneralizedCoordinate()` or an
applied `solveIK()` solution. `SEMI_IMPLICIT` also replaces `RUNGE_KUTTA_4` while the robot has
loop or mimic constraints, which are eliminated from the velocity of one step (see [Closed-Loop Systems](./ClosedLoopSystems.md)).

`getGeneralizedForce()` returns $\boldsymbol{\tau}_{cmd} + \boldsymbol{\tau}_{act}$ at the
current state: the PD term with the position error of the last step and the current velocity, and
the actuator bounds at the current motor speeds. It does not include the damping, the springs, the
external forces, the implicit terms or the solver impulses.

## Kinematic loops

An articulated system is simulated in the generalized coordinates of a kinematic tree. A *loop
constraint* closes a loop of the tree again: it ties an anchor point on `body1` to the
coincident point on `body2`. A **pin** holds the two points together in all three directions.
An **equality constraint** holds them together only along one or two axes fixed in `body1`;
along the remaining directions the points slide freely. A **mimic constraint** couples two joint
coordinates linearly. The three are handled in the same way. This section derives the dynamics
with the equality constraint as the main case; [Closed-Loop Systems](./ClosedLoopSystems.md) describes how to model
loops and how they behave, and [Mimic Joints](./MimicJoints.md) the mimic constraints. The symbols are those of
[Joint torque of a simulation step](#joint-torque-of-a-simulation-step) above.

The constraints are not iterated by the contact solver. In every step, each articulated system
eliminates them exactly, before the contacts are solved:

1. it takes the velocity $\boldsymbol{u}^*$ that the system would reach without them (step 4
   above),
2. applies the one impulse that makes that velocity satisfy them, and
3. replaces the response of everything else that acts on the system (contacts, joint limits,
   joint friction, tendons) by its response on the constrained mechanism.

The contact solver then works on a smaller problem whose every solution already satisfies the
constraints.

### Notation

| Symbol | Meaning |
| --- | --- |
| $\boldsymbol{q},\ \boldsymbol{u}$ | generalized coordinate and velocity of the tree ($\|\boldsymbol{u}\|$ entries; see [State and Kinematics](./StateAndKinematics.md)) |
| $\boldsymbol{M},\ \tilde{\boldsymbol{M}}$ | the mass matrix, and the mass matrix with the implicit diagonal of step 4 |
| $\boldsymbol{p}_1,\ \boldsymbol{p}_2$ | world positions of the anchor on `body1` and on `body2` |
| $\boldsymbol{a}_i$ | constrained unit axes of an equality constraint in the world frame. They are fixed in `body1`'s frame $B_1$, where they are ${}^{B_1}\boldsymbol{a}_i$. |
| $\boldsymbol{v}_b(\boldsymbol{x}),\ \boldsymbol{\omega}_b$ | velocity of the material point of body $b$ at the world point $\boldsymbol{x}$, and the angular velocity of body $b$ |
| $\boldsymbol{J}_b(\boldsymbol{x})$ | Jacobian of that point, $\boldsymbol{J}_b(\boldsymbol{x})\,\boldsymbol{u} = \boldsymbol{v}_b(\boldsymbol{x})$ (see [State and Kinematics](./StateAndKinematics.md)) |
| $\boldsymbol{J},\ \boldsymbol{\lambda}$ | all constraint rows stacked ($\|\boldsymbol{\lambda}\| \times \|\boldsymbol{u}\|$), and their impulses |
| $\boldsymbol{J}_c,\ \boldsymbol{\lambda}_c$ | the rows and impulses of the contacts, the joint limits and the joint friction (step 4) |
| $\boldsymbol{e}$ | the position errors of the rows |
| $\boldsymbol{D} = \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}^T$ | the coupling of the rows (their inverse effective mass, the Delassus matrix of the constraints), $\|\boldsymbol{\lambda}\| \times \|\boldsymbol{\lambda}\|$ |
| $\Delta t,\ \beta$ | the time step and the error-correction rate, $\beta = \mathrm{clamp}(0.2\,\mathrm{erp}, 0, 0.3)/\Delta t$ (see [Closed-Loop Systems](./ClosedLoopSystems.md)) |

### The constraint and its rate

For each of its axes, the equality constraint requires

$$
e_i(\boldsymbol{q}) = \boldsymbol{a}_i \cdot (\boldsymbol{p}_1 - \boldsymbol{p}_2) = 0 .
$$

A pin uses three fixed world directions as its axes $\boldsymbol{a}_i$. A mimic constraint
uses $e = q_\mathrm{follower} - m\, q_\mathrm{leader} - c$ for the multiplier $m$ and
the offset $c$ (see [Mimic Joints](./MimicJoints.md)).

**The exact rate of an equality row.** The step is solved for velocities, so each row needs the
time derivative of its error, linear in the generalized velocity $\boldsymbol{u}$. Take one
axis $\boldsymbol{a}$, with $e = \boldsymbol{a}\cdot(\boldsymbol{p}_1 - \boldsymbol{p}_2)$.
Three things move:

- the axis: it is fixed in `body1`, so it turns with `body1`'s angular velocity,
  $\dot{\boldsymbol{a}} = \boldsymbol{\omega}_1\times \boldsymbol{a}$;
- `body1`'s anchor: $\boldsymbol{p}_1$ is a material point of `body1`, so
  $\dot{\boldsymbol{p}}_1 = \boldsymbol{v}_1(\boldsymbol{p}_1)$;
- `body2`'s anchor: $\boldsymbol{p}_2$ is a material point of `body2`, so
  $\dot{\boldsymbol{p}}_2 = \boldsymbol{v}_2(\boldsymbol{p}_2)$.

One more fact is needed, the velocity field of a rigid body. The material point of `body1` that
currently sits at a world point $\boldsymbol{x}$ moves with
$\boldsymbol{v}_1(\boldsymbol{x}) = \boldsymbol{v}_1(\boldsymbol{y}) + \boldsymbol{\omega}_1\times(\boldsymbol{x} - \boldsymbol{y})$
for any other point $\boldsymbol{y}$: two points of the same body differ in velocity only by
the rotation.

1. The product rule gives
   $\dot e = \dot{\boldsymbol{a}}\cdot(\boldsymbol{p}_1 - \boldsymbol{p}_2) + \boldsymbol{a}\cdot(\dot{\boldsymbol{p}}_1 - \dot{\boldsymbol{p}}_2)$.

2. Inserting the three rates, the first term is the axis turning and the second the anchors
   moving:

   $$
   \dot e = \underbrace{(\boldsymbol{\omega}_1\times \boldsymbol{a})\cdot(\boldsymbol{p}_1 - \boldsymbol{p}_2)}_{\text{A: the axis turns}}
          + \underbrace{\boldsymbol{a}\cdot\big(\boldsymbol{v}_1(\boldsymbol{p}_1) - \boldsymbol{v}_2(\boldsymbol{p}_2)\big)}_{\text{B: the anchors move}} .
   $$

3. Move `body1`'s velocity from $\boldsymbol{p}_1$ to $\boldsymbol{p}_2$. By the
   velocity field,
   $\boldsymbol{v}_1(\boldsymbol{p}_1) = \boldsymbol{v}_1(\boldsymbol{p}_2) + \boldsymbol{\omega}_1\times(\boldsymbol{p}_1 - \boldsymbol{p}_2)$,
   so term B splits into

   $$
   \text{B} = \boldsymbol{a}\cdot\big(\boldsymbol{v}_1(\boldsymbol{p}_2) - \boldsymbol{v}_2(\boldsymbol{p}_2)\big)
            + \boldsymbol{a}\cdot\big(\boldsymbol{\omega}_1\times(\boldsymbol{p}_1 - \boldsymbol{p}_2)\big) .
   $$

4. The scalar triple product is invariant under a cyclic permutation and changes sign when two
   factors are swapped, so the last term is
   $-(\boldsymbol{\omega}_1\times \boldsymbol{a})\cdot(\boldsymbol{p}_1 - \boldsymbol{p}_2)$:
   term A with the opposite sign.

5. Term A and the rest of term B cancel:

   $$
   \dot e = \boldsymbol{a}\cdot\big(\boldsymbol{v}_1(\boldsymbol{p}_2) - \boldsymbol{v}_2(\boldsymbol{p}_2)\big)
          = \underbrace{\boldsymbol{a}^T\big(\boldsymbol{J}_1(\boldsymbol{p}_2) - \boldsymbol{J}_2(\boldsymbol{p}_2)\big)}_{\text{a row of } \boldsymbol{J}}\,\boldsymbol{u} .
   $$

Nothing was approximated: the rate holds at every instant, however far the anchors have slid
apart. An equality constraint with two axes has one such row per axis.

**The same result, seen from body1.** In `body1`'s frame, ${}^{B_1}\boldsymbol{a}$ and
${}^{B_1}\boldsymbol{p}_1$ are constant, so $e$ changes only through the motion of
`body2`'s anchor relative to `body1`. That relative velocity is `body2`'s velocity at the
anchor minus the velocity of the point of `body1` that coincides with it,
$\boldsymbol{v}_2(\boldsymbol{p}_2) - \boldsymbol{v}_1(\boldsymbol{p}_2)$, which gives
$\dot e = \boldsymbol{a}\cdot\big(\boldsymbol{v}_1(\boldsymbol{p}_2) - \boldsymbol{v}_2(\boldsymbol{p}_2)\big)$
again. An equality constraint says that `body2`'s anchor stays on a plane (one axis) or a line
(two axes) fixed in `body1`; what matters is how that anchor moves relative to `body1`,
measured where the anchor is.

**Body1's own anchor would give a wrong row.** Evaluating `body1` at its own anchor keeps only
term B,
$\boldsymbol{a}^T\big(\boldsymbol{J}_1(\boldsymbol{p}_1) - \boldsymbol{J}_2(\boldsymbol{p}_2)\big)\boldsymbol{u} = \dot e - (\boldsymbol{\omega}_1\times \boldsymbol{a})\cdot(\boldsymbol{p}_1 - \boldsymbol{p}_2)$.
That row misses the turning of the axis, an error that vanishes only when the anchors coincide or
when $\boldsymbol{\omega}_1\times \boldsymbol{a}$ happens to be perpendicular to
$\boldsymbol{p}_1 - \boldsymbol{p}_2$. An equality constraint lets the anchors slide apart
along its free directions by design, so the error would grow with the slide and with `body1`'s
rotation: a 5 cm slide at 2 rad/s gives 0.1 m/s along the constrained axis. RaiSim uses the exact
row above.

**A pin.** Its axes are fixed in the world, so $\dot{\boldsymbol{a}} = 0$ and there is no
term A, and it holds the anchors together in every direction, so
$\boldsymbol{p}_1 - \boldsymbol{p}_2$ stays at zero. Its rows
$\boldsymbol{a}_i^T\big(\boldsymbol{J}_1(\boldsymbol{p}_1) - \boldsymbol{J}_2(\boldsymbol{p}_2)\big)$
are exact as they are.

**Where the force acts.** By virtual work, a row
$\boldsymbol{a}^T\big(\boldsymbol{J}_1(\boldsymbol{p}_2) - \boldsymbol{J}_2(\boldsymbol{p}_2)\big)$
with the impulse $\lambda$ applies the generalized impulse

$$
\big(\boldsymbol{J}_1(\boldsymbol{p}_2) - \boldsymbol{J}_2(\boldsymbol{p}_2)\big)^T \boldsymbol{a}\,\lambda
  = \boldsymbol{J}_1(\boldsymbol{p}_2)^T(\lambda \boldsymbol{a}) - \boldsymbol{J}_2(\boldsymbol{p}_2)^T(\lambda \boldsymbol{a}) :
$$

the point impulse $\lambda \boldsymbol{a}$ on `body1` at $\boldsymbol{p}_2$, and
$-\lambda \boldsymbol{a}$ on `body2` at the same point. The two are equal, opposite and
collinear, so they apply no net force or torque to the system. They have no component along the
free directions, and their power $\lambda\,\dot e$ is zero while the constraint holds: the
constraint does no work while the points slide.

### Equations of motion of a step

With the constraint impulses $\boldsymbol{\lambda}$ added to step 4, one step of the
velocity-level integration reads

$$
\tilde{\boldsymbol{M}}\,(\boldsymbol{u}_{t+1} - \boldsymbol{u}_t)
  = \Delta t\,(\boldsymbol{\tau}_{exp} - \boldsymbol{h}) + \boldsymbol{J}_c^T \boldsymbol{\lambda}_c
    + \boldsymbol{J}^T \boldsymbol{\lambda} .
$$

The constraints are imposed on the velocity at the end of the step, with the position error fed
back:

<div id="equation-loop-target"></div>

$$
\boldsymbol{J}\,\boldsymbol{u}_{t+1} = -\beta\, \boldsymbol{e} .
\tag{1}
$$

As one linear system in $\boldsymbol{u}_{t+1}$ and $\boldsymbol{\lambda}$ (for given
$\boldsymbol{\lambda}_c$), this is the saddle-point (KKT) problem

$$
\begin{bmatrix} \tilde{\boldsymbol{M}} & -\boldsymbol{J}^T \\ \boldsymbol{J} & \boldsymbol{0} \end{bmatrix}
\begin{bmatrix} \boldsymbol{u}_{t+1} \\ \boldsymbol{\lambda} \end{bmatrix}
=
\begin{bmatrix} \tilde{\boldsymbol{M}} \boldsymbol{u}_t + \Delta t(\boldsymbol{\tau}_{exp} - \boldsymbol{h})
                + \boldsymbol{J}_c^T \boldsymbol{\lambda}_c \\ -\beta \boldsymbol{e} \end{bmatrix} .
$$

The impulses $\boldsymbol{\lambda}_c$ are unknown as well, and subject to the Coulomb friction
cones and to one-sided limits; that is what the contact solver iterates on. The elimination below
removes $\boldsymbol{\lambda}$ from this problem exactly, so the solver iterates only on
$\boldsymbol{\lambda}_c$.

### Eliminating the constraints

**1. The free velocity.** Without any impulse, the velocity at the end of the step is
$\boldsymbol{u}^* = \boldsymbol{u}_t + \Delta t\, \tilde{\boldsymbol{M}}^{-1}(\boldsymbol{\tau}_{exp} - \boldsymbol{h})$
(step 4).

**2. Solve the first block row for** $\boldsymbol{u}_{t+1}$:

$$
\boldsymbol{u}_{t+1} = \boldsymbol{u}^* + \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}_c^T\boldsymbol{\lambda}_c
                       + \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T\boldsymbol{\lambda} .
$$

**3. Substitute it into the constraints** [(1)](#equation-loop-target). This leaves a system for
$\boldsymbol{\lambda}$ alone, whose matrix $\boldsymbol{D}$ is the Schur complement of
$\tilde{\boldsymbol{M}}$ in the KKT matrix:

$$
\boldsymbol{D}\,\boldsymbol{\lambda} = -\beta \boldsymbol{e} - \boldsymbol{J} \boldsymbol{u}^*
  - \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}_c^T \boldsymbol{\lambda}_c .
$$

Its solution splits into a part that depends only on the free motion and a part induced by the
other impulses:

<div id="equation-loop-lambda"></div>

$$
\boldsymbol{\lambda} = \underbrace{\boldsymbol{D}^{-1}\big(-\beta \boldsymbol{e} - \boldsymbol{J} \boldsymbol{u}^*\big)}_{\boldsymbol{\lambda}_0}
          - \boldsymbol{D}^{-1} \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}_c^T \boldsymbol{\lambda}_c .
\tag{2}
$$

**4. Substitute** $\boldsymbol{\lambda}$ **back:**

$$
\boldsymbol{u}_{t+1} = \underbrace{\boldsymbol{u}^* + \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T\boldsymbol{\lambda}_0}_{\boldsymbol{u}_0}
        + \underbrace{\Big(\tilde{\boldsymbol{M}}^{-1} - \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T \boldsymbol{D}^{-1} \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1}\Big)}_{\boldsymbol{M}_c^{-1}}\,
          \boldsymbol{J}_c^T \boldsymbol{\lambda}_c .
$$

These are the reduced dynamics, in two parts:

- $\boldsymbol{u}_0$ is the free velocity moved onto the constraints; by construction,
  $\boldsymbol{J} \boldsymbol{u}_0 = -\beta \boldsymbol{e}$. Each step applies the impulse
  $\boldsymbol{J}^T\boldsymbol{\lambda}_0$ before the contacts are solved.

- Every other impulse acts through the inverse mass of the constrained mechanism,

  $$
  \boldsymbol{M}_c^{-1} = \tilde{\boldsymbol{M}}^{-1}
    - \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}^T \boldsymbol{D}^{-1} \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} ,
  $$

  whose response to an impulse is the unconstrained response minus its constrained part:
  $\boldsymbol{M}_c^{-1}\boldsymbol{J}_c^T = \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}_c^T - \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T\,\boldsymbol{D}^{-1}\big(\boldsymbol{J} \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}_c^T\big)$.

Three properties make this exact:

- **No impulse can open a constraint.** $\boldsymbol{J} \boldsymbol{M}_c^{-1} = \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} - \boldsymbol{D}\,\boldsymbol{D}^{-1}\boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} = \boldsymbol{0}$,
  so $\boldsymbol{J} \boldsymbol{u}_{t+1} = \boldsymbol{J} \boldsymbol{u}_0 = -\beta \boldsymbol{e}$
  for any $\boldsymbol{\lambda}_c$. The constraints hold whatever the contact solver does and
  however many iterations it runs.
- **The contact problem keeps its form.** $\boldsymbol{M}_c^{-1}$ is symmetric and positive
  semidefinite. The contacts' apparent inverse inertia becomes
  $\boldsymbol{J}_c \boldsymbol{M}_c^{-1} \boldsymbol{J}_c^T$: the same kind of quantity as
  without constraints, only smaller, because the closed mechanism is stiffer.
- **The constraint impulse is still known.** Equation [(2)](#equation-loop-lambda) gives
  $\boldsymbol{\lambda}$ once the solver has found $\boldsymbol{\lambda}_c$; this is the
  impulse reported for each constraint (see
  [Reading constraint forces](./ClosedLoopSystems.md#reading-constraint-forces)).

In short, RaiSim solves the KKT system by block elimination: it eliminates
$\boldsymbol{\lambda}$ through $\boldsymbol{D}$, solves the contacts on the constrained
mechanism with the inverse mass $\boldsymbol{M}_c^{-1}$, and recovers
$\boldsymbol{\lambda}$ from the solved $\boldsymbol{\lambda}_c$.

If the rows of $\boldsymbol{J}$ are dependent (redundant constraints, or a loop at a singular
pose), $\boldsymbol{D}$ is singular. RaiSim then keeps an independent subset of the rows; the
motion is the same, and the load goes to the rows that are kept (see
[Redundant constraints and singular poses](./ClosedLoopSystems.md#redundant-constraints-and-singular-poses)).

### Why these are the reduced dynamics

Constrained dynamics can equivalently be written in coordinates of the allowed motion. Let the
columns of $\boldsymbol{N}$ span the null space of $\boldsymbol{J}$, the velocities that
keep the constraints. Restricting the equations of motion to that space (d'Alembert's principle: the
constraint forces do no work on the allowed motions) gives the reduced mass
$\boldsymbol{N}^T \tilde{\boldsymbol{M}} \boldsymbol{N}$, and the constrained mechanism
responds to a generalized impulse through
$\boldsymbol{N} (\boldsymbol{N}^T \tilde{\boldsymbol{M}} \boldsymbol{N})^{-1} \boldsymbol{N}^T$.
For $\boldsymbol{J}$ of full row rank, this is the same matrix:

$$
\boldsymbol{N} (\boldsymbol{N}^T \tilde{\boldsymbol{M}} \boldsymbol{N})^{-1} \boldsymbol{N}^T
  = \tilde{\boldsymbol{M}}^{-1} - \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}^T
    (\boldsymbol{J} \tilde{\boldsymbol{M}}^{-1} \boldsymbol{J}^T)^{-1} \boldsymbol{J} \tilde{\boldsymbol{M}}^{-1}
  = \boldsymbol{M}_c^{-1} .
$$

Both sides map a generalized impulse along the constraints, $\boldsymbol{J}^T \boldsymbol{y}$,
to zero, and both map $\tilde{\boldsymbol{M}} \boldsymbol{N} \boldsymbol{z}$ to
$\boldsymbol{N} \boldsymbol{z}$. The two subspaces $\{\boldsymbol{J}^T \boldsymbol{y}\}$
and $\{\tilde{\boldsymbol{M}} \boldsymbol{N} \boldsymbol{z}\}$ together span the whole space:
$\tilde{\boldsymbol{M}} \boldsymbol{N} \boldsymbol{z} = \boldsymbol{J}^T \boldsymbol{y}$
implies $\boldsymbol{N}^T \tilde{\boldsymbol{M}} \boldsymbol{N} \boldsymbol{z} = \boldsymbol{0}$,
so $\boldsymbol{z} = \boldsymbol{0}$.

The right-hand form needs only $\boldsymbol{J}$,
$\tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T$ and a system with one equation per constraint
row. No basis of the null space is built, the generalized coordinates stay those of the tree, and
the number of constraint rows $|\boldsymbol{\lambda}|$ is small compared with
$|\boldsymbol{u}|$.

$\boldsymbol{P} = \boldsymbol{I} - \tilde{\boldsymbol{M}}^{-1}\boldsymbol{J}^T \boldsymbol{D}^{-1} \boldsymbol{J}$
projects onto the allowed velocities, orthogonally in the kinetic-energy metric
$\tilde{\boldsymbol{M}}$: $\boldsymbol{P} \boldsymbol{u}$ is the allowed velocity
closest to $\boldsymbol{u}$ in $\|\cdot\|_{\tilde{\boldsymbol{M}}}$. Moving
$\boldsymbol{u}^*$ to $\boldsymbol{u}_0$ in step 4 is this projection, shifted so that
$\boldsymbol{J} \boldsymbol{u}_0 = -\beta \boldsymbol{e}$ instead of zero. Physically, it is
the impulse that a perfectly rigid and lossless constraint applies: of all velocities that satisfy
the constraints, $\boldsymbol{u}_0$ is the one closest to $\boldsymbol{u}^*$ in kinetic
energy (Gauss's principle of least constraint).

## Integration Steps

Integration of an articulated system is performed in two stages: `integrate1` and `integrate2`.

The following steps are performed in `integrate1`

1. If the time step, the control mode, the PD gains, the damping or the springs changed, update the diagonal $\boldsymbol{w}$ added to the mass matrix (the effective inertia of the implicit springs, dampers and PD gains)
2. Update positions of the collision bodies
3. Detect collisions (called by the world instance)
4. The world assigns contacts on each object and computes the contact normal
5. Compute the mass matrix, nonlinear term, and inverse inertia matrix
6. Compute (Sparse) Jacobians of contacts

After this step, all kinematic/dynamic properties are computed.
Users can access them if they are necessary for the controller.
Next, `integrate2` computes the rest of the simulation.

7. If the controller changed the control mode, the PD gains, the damping or the springs after `integrate1`, update $\boldsymbol{w}$ as in step 1, then compute contact properties
8. Compute the PD torque (if used), add the feedforward generalized force, clamp this sum to the joint effort limits, and add the joint damping (never clipped, see [Joint Damping and Friction](./JointDampingAndFriction.md))
9. Clip the actuator torques to the motor operating regions, map them to the joints and add them; then add the springs and the external forces/torques, and compute the velocity without contacts
10. Contact solver (called by the world instance). It also solves the joint limits and the joint friction
11. Integrate the velocity
12. Integrate the position with the integration scheme

Steps 8-12 are written out exactly in [Joint torque of a simulation step](#joint-torque-of-a-simulation-step).

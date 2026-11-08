# State and Kinematics

## State Representation

The state of an articulated system can be represented by a **generalized state** $\boldsymbol{S}$, which is composed of a **generalized coordinate** $\boldsymbol{q}$ and a **generalized velocity** $\boldsymbol{u}$.
Since we are not constraining their parameterization, in general,

$$
\begin{equation}
  \boldsymbol{u}\neq\dot{\boldsymbol{q}}.
\end{equation}
$$

A generalized coordinate fully represents the configuration of the articulated system and a generalized velocity fully represents the velocity state of the articulated system.
They are independently defined in general.

Every joint has a corresponding generalized coordinate and generalized velocity.
The concatenation of all joint generalized coordinates and velocities forms the generalized coordinates and velocities of the articulated system, respectively.
The order of this concatenation is called **joint order**.
The joint order can be accessed through `getMovableJointNames()`.
Note the keyword "movable".
The fixed joints contribute to neither the generalized coordinate nor the generalized velocity.
Only movable joints do.

The **root body** is the first body (body 0) of the articulated system.
For floating-base systems, the root body is the floating base. Its floating joint comes first in the joint order and is named `ROOT` in `getMovableJointNames()`.
For fixed-base systems, the root body is the one rigidly attached to the world. Its joint is fixed, so it is not part of the joint order, and the joint order starts with the first movable joint.
Even though the fixed base cannot move physically, users can move it using `setBaseOrientation` and `setBasePos`.

To set the state of the system, the following methods can be used

- `setGeneralizedCoordinate`
- `setGeneralizedVelocity`
- `setState`

To obtain the state of the system, the following methods can be used

- `getGeneralizedCoordinate`
- `getGeneralizedVelocity`
- `getState`

The dimensions of each vector can be obtained respectively by

- `getGeneralizedCoordinateDim`
- `getDOF` or `getGeneralizedVelocityDim` (these two methods are identical)

<a id="articulated-systems"></a>

## Joints

Here are the available joints in RaiSim.

**Joint Properties ($|\cdot|$ is a symbol for dimension size (i.e., cardinality))**

|  | Fixed | Floating | Revolute | Prismatic | Spherical |
| --- | --- | --- | --- | --- | --- |
| $\|\boldsymbol{u}\|$ | 0 | 6 | 1 | 1 | 3 |
| $\|\boldsymbol{q}\|$ | 0 | 7 | 1 | 1 | 4 |
| Velocity |  | $m/s$, $rad/s$ | $rad/s$ | $m/s$ | $rad/s$ |
| Position |  | $m$, $rad$ | $rad$ | $m$ | $rad$ |
| Force |  | $N$, $Nm$ | $Nm$ | $N$ | $Nm$ |

The generalized coordinates/velocities of a joint are expressed in the **joint frame** and with respect to the **parent body**.
Joint frame is the frame attached to every joint and fixed to the parent body.
Parent body is the one closer to the root body among the two bodies connected via the joint.
Note that the angular velocity of a floating base is also expressed in the parent frame (which is the **world frame**).
Other libraries (e.g., RBDL) might have a different convention, so special care is required during conversion.

## Types of Indices

The ArticulatedSystem class contains multiple types of indices. To query a specific quantity, you have to provide an index of the right type. The types of indices in Articulated Systems are:

- **Body/Joint Index**: Links connected by fixed joints are merged into a single body. Each body has a unique body index. Because every body has exactly one parent joint, there is a 1-to-1 mapping between the joints and the bodies and they share the same index. For a fixed-base system, the body rigidly fixed to the world is body 0. For a floating-base system, the floating base is body 0. Retrieve it with `getBodyIdx()`.
- **Generalized Velocity (DOF) Index**: The index of an entry in the generalized velocity (and in the generalized force). Each movable joint occupies a block of 1 (revolute, prismatic), 3 (spherical) or 6 (floating) consecutive entries.
- **Generalized Coordinate Index**: The index of an entry in the generalized coordinate. Each movable joint occupies a block of 1 (revolute, prismatic), 4 (spherical) or 7 (floating) consecutive entries.
- **Frame Index**: The index of a frame in `getFrames()`. Every joint, including fixed joints, has a frame (see [Frames](#frames)). Retrieve it with `getFrameIdxByName()`.

### Conversions Between Indices

- A body index to a generalized velocity index: `ArticulatedSystem::getMappingFromBodyIndexToGeneralizedVelocityIndex()`
- A body index to a generalized coordinate index: `ArticulatedSystem::getMappingFromBodyIndexToGeneralizedCoordinateIndex()`

## Kinematics

### Frames

The position and velocity of a specific point on a body of an articulated system can be obtained by attaching a **frame**.
**Frames** are rigidly attached to a body of the system and have a constant position and orientation (w.r.t. parent frame).
This is the recommended way to get kinematics information for a point of an articulated system in RaiSim.

All joints have a frame attached and their names are the same as the joint name.
To create a custom frame, define a fixed joint at the point of interest.
A dummy link with zero inertia and zero mass must be added as the child of the fixed joint to complete the kinematic tree.

A frame can be stored locally as an index in user code. For example:

```cpp
#include "raisim/World.hpp"

int main() {
  raisim::World world;
  auto anymal = world.addArticulatedSystem(PATH_TO_URDF);
  auto footFrameIndex = anymal->getFrameIdxByName("foot_joint"); // the URDF has a joint named "foot_joint"
  raisim::Vec<3> footPosition, footVelocity, footAngularVelocity;
  raisim::Mat<3,3> footOrientation;
  anymal->getFramePosition(footFrameIndex, footPosition);
  anymal->getFrameOrientation(footFrameIndex, footOrientation);
  anymal->getFrameVelocity(footFrameIndex, footVelocity);
  anymal->getFrameAngularVelocity(footFrameIndex, footAngularVelocity);
}
```

You can also store a Frame reference.
For example, you can replace `getFrameIdxByName` with `getFrameByName` in the example above.
In this way, you can access internal variables and modify them.
Modifying frames does not affect the joints.
Frames are instantiated during initialization of the articulated system instance and affect neither kinematics nor dynamics, even if you change them.

### Joint limits

Joint limits can be defined in a URDF file **per joint** as follows:

```xml
<limit effort="80" lower="-6.28" upper="6.28" velocity="15"/>
```

The `lower` and `upper` are joint position limits and the `velocity` is the joint velocity limit.
The joint limits are implemented as if there is a hard stop at the limits.
This means that there is a hard collision (with a restitution coefficient of 0) when the joint hits a limit.
A spherical joint has a single limit: `lower` is the largest total rotation angle $2\arccos|q_w|$ (twist included) away from the zero configuration, and `upper` is ignored.
The joint stops at that angle like at any other hard stop; torque that pushes it outward is resisted, while motion along or back from the limit is free.
Velocity limits do not apply to spherical joints.
The `effort` is not a joint limit but the actuation limit of the joint: it bounds the feedforward generalized force plus the built-in PD torque (see [Dynamics and Control](./DynamicsAndControl.md)).
It does not bound actuator torques, which are limited by the operating regions of their motors (see [Actuators](../Actuators.md)).

You can modify the joint position limits in C++ using `raisim::ArticulatedSystem::setJointLimits()` (one entry per degree of freedom) and the joint velocity limits using `raisim::ArticulatedSystem::setJointVelocityLimits()` (one entry per body; a non-finite entry means no limit).

During simulation, you can get information on joint limit violations using `raisim::ArticulatedSystem::getJointLimitViolations(*world.getContactProblem())`.
Even though joint limits are collisions (and thus handled by a contact solver), they are not listed in `raisim::Object::getContacts()`.

### Jacobians

Jacobians of a point in RaiSim satisfy the following equation:

$$
\begin{equation}
  \boldsymbol{J}\boldsymbol{u} = \boldsymbol{v}
\end{equation}
$$

where $\boldsymbol{v}$ represents the linear velocity of the associated point.
If a rotational Jacobian is used, the right-hand side changes to a rotational velocity expressed in the world frame.

To get the Jacobians associated with the linear velocity, the following methods are used

- `getSparseJacobian`
- `getDenseJacobian` -- this method only fills non-zero values. The matrix should be initialized to a zero matrix of an appropriate size.

To get the rotational Jacobians, the following methods are used

- `getSparseRotationalJacobian`
- `getDenseRotationalJacobian` -- this method only fills non-zero values. The matrix should be initialized to a zero matrix of an appropriate size.

The main Jacobian class in RaiSim is `raisim::SparseJacobian`.
RaiSim uses only sparse Jacobians because it is more memory-efficient.
Note that only the joints between the child body and the root body affect the motion of the point.

The class `raisim::SparseJacobian` has a member `idx` which stores the indices of columns whose values are non-zero.
The member `v` stores the Jacobian except the zero columns.
In other words, ith column of `v` corresponds to `idx[i]` generalized velocity dimension.

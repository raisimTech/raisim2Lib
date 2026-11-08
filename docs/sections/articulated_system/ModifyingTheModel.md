# Modifying the Model in Code

RaiSim allows users to modify most of the robot parameters freely in code.
This allows users to create randomized robot models, which might be useful for AI applications (i.e., **dynamic randomization**).
Note that a random model might be kinematically and dynamically unrealistic.
For example, joints can be locked by collision bodies.
In such cases, simulation cannot be performed reliably and it is advised to carefully check randomly generated robot models.

Here is a list of modifiable kinematic/dynamic parameters.

- **Joint Position (relative to the parent joint) Expressed in the Parent Frame**

`getJointPos_P` method returns (a non-const reference to) a `std::vector` of position vectors from the parent joint to the child joint expressed in the respective parent joint frames.
This should be changed with care since it can result in unrealistic collision geometry.
**This method does not change the position of the end-effector with respect to its parent** as the position of the last link is defined by the collision body position, not by the joint position.
The elements are ordered by the joint indices.

- **Joint Axis in the Parent Frame**

`getJointAxis_P` method returns (a non-const reference to) a `std::vector` of joint axes expressed in the respective parent joint frame.
This method should also be changed with care.
The elements are ordered by the joint indices.
Call `setGeneralizedCoordinate()` or `updateKinematics()` after changing an axis: a step starts from the kinematics computed before the change and updates them only at its end. If you keep the returned reference and change the axes through it later, call `updateMassInfo()` after the change and before `setGeneralizedCoordinate()` or `updateKinematics()`.

- **Mass of the Links**

`getMass` method returns (a non-const reference to) a `std::vector` of link masses.
**IMPORTANT!** You must call `updateMassInfo()` after changing mass values.
The elements are ordered by the body indices (which is the same as the joint indices in RaiSim).

- **Center of Mass Position**

`getBodyCOM_B` method returns (a non-const reference to) a `std::vector` of the COM of the bodies.
The elements are ordered by the body indices.

- **Link Inertia**

`getInertia` method returns (a non-const reference to) a `std::vector` of link inertia.
The elements are ordered by the body indices.

- **Collision Bodies**

`getCollisionBodies` method returns (a non-const reference to) a `raisim::CollisionSet`, a `std::vector` of `raisim::CollisionDefinition`.
This vector contains all collision bodies associated with the articulated system.

`getCollisionBody` method returns a specific collision body instead.
All collision bodies are named "LINK_NAME" + "/INDEX".
For example, the 2nd collision body of a link named "FOOT" is named "FOOT/1" (1 because the index starts from 0).

A `CollisionDefinition` holds the position and orientation offset from the body frame (`posOffset`, `rotOffset`), the body index (`localIdx`), the shape and its parameters, and the collision body handle (`getCollisionBody()`).
To change the shape or the offsets of a collision body, use `setCollisionBodyShapeParameters()` (primitive shapes only), `setCollisionBodyPositionOffset()` and `setCollisionBodyOrientationOffset()` of the articulated system with the index of the collision body in `getCollisionBodies()`.
Users can also change the material of a collision body with `setMaterial()`, which affects the contact dynamics, and its collision group and mask with `setCollisionGroup()` and `setCollisionMask()`.

## Collision

Apart from the collision mask and collision group, users can also disable a collision between a certain pair of bodies with `ignoreCollisionBetween(bodyIdx1, bodyIdx2)`.
If the system is already in a world, call `updateSelfCollisionCache(world)` afterwards.

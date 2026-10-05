# Strandbeest

A 12-legged walking mechanism inspired by Theo Jansen's Strandbeest, with 36 closed kinematic
loops. Converted from `data/Strandbeest.urdf` of
[StrandbeestRobot.jl](https://github.com/rdeits/StrandbeestRobot.jl) (MIT, see LICENSE.md; the
URDF was originally developed as part of the Drake project).

- 92 links and 91 tree joints (73 revolute, 18 fixed). The crank `joint_crossbar_crank` drives
  all legs.
- The original's 42 `<loop_joint>` hinges are closed by RaiSim `<equality>` constraints, except
  six: the crank axle's bearings in the crossbar lie on the crank joint's axis and repeat that
  joint, so they are left out. Every loop is planar (all hinge axes are parallel), so each hinge
  constrains its two anchors along the two axes normal to its hinge axis.
- The original's zero configuration is not assembled. `nominal_config` holds the joint angles
  solved by least squares so every loop closes (worst residual 4e-13 m), with the crank at zero.
- Masses and inertias are the original's (1 kg and 1 kg m^2 per link).
- Visual rods have a 2 cm radius, with a 3 cm crank axle, so the mechanism is easier to see.
  Foot collision spheres retain their original 1 cm radius.

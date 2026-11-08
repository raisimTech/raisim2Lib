# CustomLengthConstraint migration

This class is no longer exposed in Python. In C++ it remains only as a
source-compatibility view of a two-site tendon, returned by
`World::addCustomWire`. For new code, use a two-site tendon and call
`Tendon::setTension` (positive tension pulls) or `Tendon::setActuationForce`
(positive force lengthens the tendon).
See [Constraints](./Constraints.md) for the migration table and
[Tendon code examples](./tendons/CodeExamples.md) for complete, compiled examples.

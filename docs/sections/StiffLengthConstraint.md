# StiffLengthConstraint migration

This class is no longer exposed in Python. In C++ it remains only as a
source-compatibility view of a two-site tendon, returned by
`World::addStiffWire`. For new code, use a two-site tendon with a lower
bound, upper bound, or equal bounds.
See [Constraints](./Constraints.md) for the migration table and
[Tendon code examples](./tendons/CodeExamples.md) for complete, compiled examples.

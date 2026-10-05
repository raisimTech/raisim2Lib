CompliantLengthConstraint migration
===================================

This class is no longer exposed in Python. In C++ it remains only as a
source-compatibility view of a two-site tendon, returned by
``World::addCompliantWire``. For new code, use a two-site tendon with
stiffness and a spring rest interval.
See :doc:`Constraints` for the migration table and
:doc:`tendons/CodeExamples` for complete, compiled examples.

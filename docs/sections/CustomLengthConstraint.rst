CustomLengthConstraint migration
================================

This class is no longer exposed in Python. In C++ it remains only as a
source-compatibility view of a two-site tendon, returned by
``World::addCustomWire``. For new code, use a two-site tendon and call
``Tendon::setTension`` (positive tension pulls) or ``Tendon::setActuationForce``
(positive force lengthens the tendon).
See :doc:`Constraints` for the migration table and
:doc:`tendons/CodeExamples` for complete, compiled examples.

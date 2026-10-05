#############################
Determinism
#############################

A RaiSim world is deterministic: stepping the same world from the same state
with the same inputs reproduces the same trajectory. The one piece of state that
is easy to overlook is the contact solver's sweep direction. The solver
alternates between forward and backward sweeps through the contacts and flips
the direction after every solve, so the result of a step also depends on how
many solves preceded it.

To make a step depend only on the world state, set the direction explicitly
with ``raisim::World::setContactSolverIterationOrder(order)`` before stepping,
for example after resetting a world to an initial state or before each step of
a rollout that must match another one. ``World::captureCheckpoint()`` and
``restoreCheckpoint()`` save and restore the solver direction together with the
rest of the world state, so a replay from a restored checkpoint follows the
original run. See :doc:`WorldSystem` for both APIs.

Bit-identical results are expected for repeated runs of the same program with
the same RaiSim library. Different builds, compilers, or CPU architectures can
round floating-point operations differently and produce slightly different
trajectories.

Determinism is critical for two primary reasons.
First, it permits precise control over simulation stochasticity.
This facilitates the implementation of numerous sampling-based stochastic optimal control methods.
Second, determinism typically indicates a more robust simulation implementation.
Non-determinism frequently arises from unintended memory access patterns.
Simulator developers rarely implement stochasticity intentionally within the core physics engine, with the exception of specific components such as stochastic contact solvers.

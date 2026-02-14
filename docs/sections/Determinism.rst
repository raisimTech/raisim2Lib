#############################
Determinism
#############################

RaiSim can achieve full determinism by defining the iteration order of the contact solver.
This order is configured via ``raisim::World::setContactSolverIterationOrder``.

Determinism is critical for two primary reasons.
First, it permits precise control over simulation stochasticity.
This facilitates the implementation of numerous sampling-based stochastic optimal control methods.
Second, determinism typically indicates a more robust simulation implementation.
Non-determinism frequently arises from unintended memory access patterns.
Simulator developers rarely implement stochasticity intentionally within the core physics engine, with the exception of specific components such as stochastic contact solvers.

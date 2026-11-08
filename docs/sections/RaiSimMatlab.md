# RaisimMatlab

The current public `raisim2Lib` workspace does not ship a MATLAB wrapper
source directory or CMake target. `RAISIM_MATLAB` remains declared as a
compatibility/reserved CMake option, but enabling it does not build a wrapper.

Use the C++ API directly or [RaisimPy](./RaiSimPy.md) for a currently maintained language
binding. Do not rely on `RAISIM_MATLAB=ON` as an installation step.

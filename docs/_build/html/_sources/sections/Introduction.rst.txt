#############################
Introduction
#############################

RaiSim is a high-performance physics engine developed by Jemin Hwangbo, KAIST.
It is engineered to deliver both accuracy and computational efficiency for simulating robotic systems.
It serves as a general-purpose rigid-body simulator capable of efficiently simulating diverse rigid-body dynamics.

**Why RaiSim?**

* Its performance is benchmarked against other popular physics engines (`[1] <https://github.com/leggedrobotics/SimBenchmark>`_).
* The accuracy of RaiSim has been validated in numerous academic publications (`[2] <https://robotics.sciencemag.org/content/4/26/eaau5872/tab-article-info>`_, `[3] <https://arxiv.org/pdf/1901.07517.pdf>`_, `[4] <https://robotics.sciencemag.org/content/5/47/eabc5986>`_,  `[5] <https://arxiv.org/abs/1909.08399>`_,  `[6] <https://arxiv.org/abs/2011.08811>`_)
* A C++ simulation library designed for ease of learning and use.
* Minimal external dependencies (requires only STL and Eigen).

System Requirements
=====================
- **Linux**
    * Ubuntu 22.04 or higher is recommended, though RaiSim may function on other distributions. Compatibility is limited to x86 CPUs with the AVX2 instruction set (Intel Haswell or higher).

- **Windows 10**
    * Visual Studio 2019 or higher. x86 CPU only.

- **macOS**
    * Latest version. Intel systems require AVX2 support. Only apple silicon (M1 and higher) is supported.

Example Code
===================
The following is an example of a RaiSim application. Visualization is supported via either RaisimUnity or RaisimUnreal.

.. code-block:: cpp

  #include "raisim/World.hpp"
  #include "raisim/RaisimServer.hpp"

  int main() {
    raisim::World::setActivationKey("PATH_TO_THE_ACTIVATION_KEY");
    raisim::World world;
    auto anymal = world.addArticulatedSystem(PATH_TO_URDF);
    auto ball = world.addSphere(1, 1);
    auto ground = world.addGround();
    world.setTimeStep(0.002);

    /// Launch the RaiSim server for visualization. Visualize in RaisimUnity.
    raisim::RaisimServer server(&world);
    server.launchServer();

    while (1) {
      raisim::MSLEEP(2);
      server.integrateWorldThreadSafe();
    }

    server.killServer();
  }

Below is a CMake file for compiling the aforementioned application.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.10)
  project(raisim_examples LANGUAGES CXX)

  find_package(raisim CONFIG REQUIRED)
  find_package(Eigen3 REQUIRED)
  
  include_directories (${EIGEN3_INCLUDE_DIRS})

  add_executable(APP_NAME ${file_name})
  target_link_libraries(APP_NAME PUBLIC raisim::raisim pthread)

A functional version is available here (RAISIM_EXAMPLE_).

.. _RAISIM_EXAMPLE: https://github.com/raisimTech/raisimExample

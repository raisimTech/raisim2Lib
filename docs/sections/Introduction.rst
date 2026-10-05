#############################
Introduction
#############################

RaiSim is a high-performance physics engine developed for robotics and AI. It
is designed for deterministic, single-process simulation of rigid bodies,
articulated systems, contacts, sensors, deformable bodies, and granular
particles.

**Why RaiSim?**

* RaiSim is optimized for high-throughput robotics simulation.
* The accuracy of RaiSim has been validated in numerous academic publications (`[1] <https://robotics.sciencemag.org/content/4/26/eaau5872/tab-article-info>`__, `[2] <https://arxiv.org/pdf/1901.07517.pdf>`__, `[3] <https://robotics.sciencemag.org/content/5/47/eabc5986>`__, `[4] <https://arxiv.org/abs/1909.08399>`__, `[5] <https://arxiv.org/abs/2011.08811>`__).
* The C++ API is organized around explicit worlds, objects, materials, sensors,
  and visualizer integration.
* The public ``raisim2Lib`` repository provides example and rayrai viewer
  sources, resources, and documentation; its release archives provide the
  prebuilt RaiSim and rayrai headers and libraries.

System Requirements
=====================
* **Linux**: Ubuntu 22.04 or newer is recommended. x86 builds require AVX2.
* **Windows**: Windows 10 or newer with Visual Studio 2019 or newer.
* **macOS**: current macOS releases. Apple silicon is supported; Intel systems
  require AVX2.

See :doc:`Installation` for dependency and activation details.

Example Code
===================
The following is the shape of a simple RaiSim application. It creates a world,
adds objects, publishes the scene through ``RaisimServer``, and steps the world
through the server's thread-safe helper. Run ``rayrai_tcp_viewer`` to see the
scene (see :doc:`QuickStart`).

.. code-block:: cpp

  #include "raisim/World.hpp"
  #include "raisim/RaisimServer.hpp"

  int main() {
    // Optional when the key is at the default location (see Installation).
    raisim::World::setActivationKey("/path/to/activation.raisim");

    raisim::World world;
    auto* robot = world.addArticulatedSystem("/path/to/robot.urdf");
    auto* ball = world.addSphere(0.5, 1.0);  // radius 0.5 m, mass 1 kg
    ball->setPosition(2.0, 0.0, 2.0);
    auto* ground = world.addGround();
    world.setTimeStep(0.002);

    raisim::RaisimServer server(&world);
    server.launchServer();

    while (true) {
      raisim::MSLEEP(2);
      server.integrateWorldThreadSafe();
    }
  }

Below is a minimal downstream CMake file for an installed RaiSim package.
Configure it with ``-DCMAKE_PREFIX_PATH=/path/to/raisim2Lib/raisim`` (add the
``rayrai`` prefix as well when the application uses rayrai). The
``raisim::raisim`` target brings in Eigen and requires C++20.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.16)
  project(raisim_examples LANGUAGES CXX)

  find_package(raisim CONFIG REQUIRED)
  find_package(Eigen3 REQUIRED)

  add_executable(app main.cpp)
  target_link_libraries(app PUBLIC raisim::raisim)
  if (UNIX)
    target_link_libraries(app PUBLIC pthread)
  endif()

For source-built examples and visualization workflows, see :doc:`QuickStart` and
:doc:`Examples`.

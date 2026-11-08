---
layout: home
markdownStyles: false
title: RaiSim
titleTemplate: Train robots faster than reality
description: RaiSim is a fast physics engine for robot learning, simulating contact-rich robots hundreds of times faster than real time, with photoreal cameras and LiDAR through rayrai.
---

<HomePage>
<template #cpp>

```cpp
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

int main() {
  raisim::World world;
  world.setTimeStep(0.001);
  world.addGround();

  auto anymal = world.addArticulatedSystem("rsc/anymal_c/urdf/anymal.urdf");
  Eigen::VectorXd q(anymal->getGeneralizedCoordinateDim());
  q << 0, 0, 0.54, 1, 0, 0, 0,
       0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
  anymal->setGeneralizedCoordinate(q);

  // PD control on the 12 leg joints, none on the floating base.
  Eigen::VectorXd kp = Eigen::VectorXd::Zero(18), kd = Eigen::VectorXd::Zero(18);
  kp.tail(12).setConstant(200.0);
  kd.tail(12).setConstant(10.0);
  anymal->setPdGains(kp, kd);
  anymal->setPdTarget(q, Eigen::VectorXd::Zero(18));

  // Stream the world to rayrai_tcp_viewer while it runs.
  raisim::RaisimServer server(&world);
  server.launchServer();
  for (int i = 0; i < 100000; ++i) {
    RS_TIMED_LOOP(1e6 * world.getTimeStep())
    server.integrateWorldThreadSafe();
  }
  server.killServer();
}
```

</template>
<template #python>

```python
import time
import raisimpy as raisim

world = raisim.World()
world.setTimeStep(0.001)
world.addGround()
ball = world.addSphere(0.2, 1.0)
ball.setPosition(0.0, 0.0, 1.0)

# Stream the world to rayrai_tcp_viewer while it runs.
server = raisim.RaisimServer(world)
server.launchServer(8080)
for _ in range(10_000):
    server.integrateWorldThreadSafe()
    time.sleep(world.getTimeStep())  # run at roughly real time
server.killServer()
```

</template>
</HomePage>

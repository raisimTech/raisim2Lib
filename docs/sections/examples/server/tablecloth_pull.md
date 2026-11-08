# Server Example: Tablecloth Pull

## Overview

The tablecloth trick: three dishes stand on a cloth on a table, and a sharp pull
on the cloth's edge whips it out from under them. Kinetic friction (0.3) acts on
the dishes only for the fraction of a second the cloth slides under them, too
briefly to take them along: they shift by about 3 cm and stay on the table. The
cloth is solved together with the dishes and the table
([`World::setMonolithicDeformables()`](../../DeformableObject.md#solving-deformables-together-with-the-bodies)).
Every three seconds the world starts over, and the example prints how far the
dishes moved.

## Target

CMake target: `tablecloth_pull`.

## Run

```bash
./build-examples/examples/tablecloth_pull
```

Connect `rayrai_tcp_viewer` to port 8080. On Windows, run
`build-examples/bin/tablecloth_pull.exe`.

## Details

- The cloth is 0.8 m by 0.64 m, 41 x 33 particles weighing 0.2 kg in total,
  with nearly inextensible edges and a little air drag.
- The dishes are 0.6 kg cylinders, 0.14 m across. The friction coefficient is
  0.3 between the cloth and both the table and the dishes, and 0.5 between the
  dishes and the table.
- One second into each cycle, a 60 N force pulls the cloth's edge outward and
  slightly down for 0.15 s.
- The scene lives in `include/tablecloth_pull.hpp`.

# RaiSim vs Newton

For changing control targets, see [Random actuation with RL-style control intervals](#random-actuation).

## Fixed-target ten-step performance per USD

This benchmark compares **ten-step RaiSim scheduling on a Core Ultra 7 265K**
with **Newton SolverMuJoCo / MuJoCo Warp on an RTX 5070**. Higher is better.
FPS/USD means aggregate world physics steps per second divided by the entire component
price, including when only one CPU worker is used. Each environment has its own world, ground and robot.

[Open the interactive ten-step FPS/USD report](/raisim_newton_ten_step_value.html)
· [Download the recorded data](/raisim_newton_ten_step_value.json).

Prices checked 15 November 2026: **265K $339.99**, a recent
[Newegg advertised-price scenario](https://www.newegg.com/Intel-x-Newegg-Anniversary-Deals/EventSaleStore/ID-10632)
from an index crawled last month; the
[direct CPU listing](https://www.newegg.com/intel-core-ultra-7-265k-arrow-lake-lga-1851-desktop-cpu-processor/p/N82E16819118506?Item=N82E16819118506)
is now out of stock, so this is not a verified currently buyable quote.
**RTX 5070 $899.99** comes from the
[new MSI Ventus listing sold/shipped by Newegg](https://www.newegg.com/msi-geforce-rtx-5070-12g-ventus-2x-oc-video-card/p/N82E16814137994?Item=N82E16814137994),
shown with Add to cart. That board is a pricing proxy for the measured RTX 5070.
Tax, host system, RAM, motherboard, cooling, energy and licensing are excluded;
the GPU also needs a host CPU. Prices are editable in the interactive report.

### Reading the price-normalized results

At the prices above, the RTX 5070 costs **2.65× as much as the 265K**
($899.99 / $339.99). Its throughput must therefore be 2.65× higher to match
RaiSim's FPS/USD. Dividing GPU throughput by the higher price reduces its value score.

At 4,096 worlds, Newton's measured throughput is **1.41× higher for ANYmal** and
**1.27× higher for Strandbeest**. These gains are smaller than the 2.65× price difference,
so RaiSim provides **1.88× and 2.08× the FPS/USD**, respectively.
The last column below is a **performance-per-dollar ratio**, not a hardware-price ratio
or a raw-throughput ratio:

`RaiSim / Newton FPS-per-dollar = (RaiSim FPS / Newton FPS) × ($899.99 / $339.99)`.

### ANYmal standing

<figure>
<img class="diagram" src="../../rsc/docs/image/raisim_newton_anymal_fps_per_usd.svg" alt="ANYmal standing: grouped bars comparing ten-step RaiSim on a 265K and historical Newton on an RTX 5070 in aggregate physics steps per second per USD, at increasing world counts. CPU measurements stop at 4096 worlds; larger counts show Newton only. Higher is better.">
<figcaption>Ten-step performance per USD at $339.99 for the CPU and $899.99 for the GPU; the GPU costs 2.65× as much. CPU values above 4,096 worlds are unmeasured. Prices and comparison limits are described above and below.</figcaption>
</figure>

| Worlds | CPU workers | RaiSim FPS/USD | Newton FPS/USD | RaiSim / Newton FPS-per-dollar |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1,068.60 | 5.55 | 192.43× |
| 64 | 16 | 13,146.06 | 321.84 | 40.85× |
| 256 | 20 | 15,630.98 | 1,178.54 | 13.26× |
| 1,024 | 20 | 13,360.45 | 3,761.72 | 3.55× |
| 4,096 | 20 | 12,312.24 | 6,542.08 | 1.88× |
| 8,192 | — | Not measured | 7,644.33 | — |
| 16,384 | — | Not measured | 7,897.44 | — |
| 32,768 | — | Not measured | 7,706.77 | — |

### Strandbeest

<figure>
<img class="diagram" src="../../rsc/docs/image/raisim_newton_strandbeest_fps_per_usd.svg" alt="Strandbeest: grouped bars comparing ten-step RaiSim on a 265K and historical Newton on an RTX 5070 in aggregate physics steps per second per USD, at increasing world counts. CPU measurements stop at 4096 worlds; larger counts show Newton only. Higher is better.">
<figcaption>Ten-step performance per USD at $339.99 for the CPU and $899.99 for the GPU; the GPU costs 2.65× as much. CPU values above 4,096 worlds are unmeasured. Prices and comparison limits are described above and below.</figcaption>
</figure>

| Worlds | CPU workers | RaiSim FPS/USD | Newton FPS/USD | RaiSim / Newton FPS-per-dollar |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 117.60 | 3.30 | 35.67× |
| 64 | 16 | 1,493.57 | 127.10 | 11.75× |
| 256 | 20 | 1,553.38 | 365.93 | 4.25× |
| 1,024 | 20 | 1,513.19 | 606.58 | 2.49× |
| 4,096 | 20 | 1,511.11 | 725.41 | 2.08× |
| 8,192 | — | Not measured | 740.77 | — |
| 16,384 | — | Not measured | 714.88 | — |

### Scaling beyond 4,096 worlds

Newton supports more than 4,096 environments; this was a benchmark sweep endpoint,
not a solver limit. Fresh runs retain the same ten-step graphs, solver parameters,
precision, time horizons and three-trial protocol. Larger batches can increase
aggregate throughput while reducing steps/s per environment. Once the GPU is
saturated, additional environments mainly consume memory and can reduce throughput.

- **ANYmal:** best measured FPS/USD was **7,897.44** at **16,384 worlds**, 1.21× the fresh 4,096-world value. Largest tested batch: **32,768**, sampled process GPU-memory peak **2,438 MiB**.
- **Strandbeest:** best measured FPS/USD was **740.77** at **8,192 worlds**, 1.02× the fresh 4,096-world value. Largest tested batch: **16,384**, sampled process GPU-memory peak **5,770 MiB**.

The fresh 4,096-world throughput is within 1.1% of the previous historical value
for both robots. Among the fresh ANYmal trials, FPS/USD ranged from **7,792.52–7,898.03** at 16,384 worlds and
**7,603.89–7,715.17** at 32,768 worlds.
The small decline at 32,768 worlds is an observed result, not proof of a universal
optimum. The downloadable records include all three trial timings for every fresh case.

All reported fresh cases passed finite-state, constraint-quality and zero-overflow
checks. Sampled GPU-memory peaks come from periodic nvidia-smi process readings,
not allocator-exact peaks. The RTX 5070 has 12 GB VRAM; the sweep avoids doubling
a workload once sampled usage exceeds 5,000 MiB because it could approach or exceed available VRAM.
This safeguard does not establish a maximum supported world count.

RaiSim was not measured above 4,096 worlds in this comparison; missing CPU values
are not zero or extrapolated. GPU results through 1,024 worlds remain historical
14 November measurements; GPU results at 4,096 and above are fresh 15 November runs.
[Download full scaling measurements, validation and GPU-memory samples](/raisim_newton_large_worlds.json).

### Protocol and limits

CPU workers integrate each world ten times before visiting the next world. Newton
uses graphs of ten GPU steps. Workers own disjoint world ranges and are CPU-pinned;
integration inside each world is single-threaded. Worker counts are recorded
configurations, not a demonstrated optimum. Setup, warmup, validation and external
policy computation are excluded. Time steps are 2 ms for ANYmal and 1 ms for Strandbeest.
CPU throughput uses median elapsed wall time; single-world cases have 21 trials
across three processes and multiworld cases have 10 across two, except ANYmal
4,096/20 scores six of ten after the prespecified first-two-per-process exclusion.
Processes ran serially; timed horizons vary with world count.

RaiSim CPU measurements are from 15 November 2026 and use FP64. Newton GPU measurements
use FP32: counts through 1,024 are from 14 November, and counts from 4,096 onward
were rerun on 15 November for the larger-world sweep.
The scenes/controllers were matched, but solvers, constraint formulations and
trajectories differ; these ratios do not compare identical numerical work. ANYmal
has four contacts/world. Historical Strandbeest trajectory means are approximately
0.959 contacts/world for RaiSim and 0.686–0.700 for Newton; the current RaiSim ending
count of one/world is a different statistic. RaiSim merges fixed links (74 bodies),
while Newton retains them (92). All 18 CPU scheduling cases matched final state hashes
and contacts against one-step scheduling, with six extra partial-block/uneven-range checks.
The tables retain the ten primary CPU comparison cases and add the measured larger GPU batches.


## Random actuation with RL-style control intervals {#random-actuation}

This workload changes each environment's control targets independently every ten physics
steps, instead of holding a standing pose or a constant crank speed. It measures
**random action generation, target updates and physics**, with a CPU worker barrier at
each control interval and ordered GPU graph execution. It does **not** include a
learned policy, observations, rewards, episode resets or rendering, so it is not a
full RL-training throughput measurement. Each control transition contains ten physics
steps: divide the reported physics FPS/USD by **10** for control transitions/s/USD.

- **ANYmal:** 12 joint-position targets, sampled uniformly within ±0.15 rad of the
  nominal pose, with position gain 200 and damping 10. Actions update every 20 ms.
- **Strandbeest:** the single actuated crank's velocity target varies by ±25% around
  2π rad/s, with damping 200. The linkage joints remain passive. Actions update every 10 ms.
- Seed **20261115**, keyed by environment, control interval and joint. CPU and GPU
  targets use the same float32 hash-to-action calculation and canonical joint mapping.
  Cross-engine target checks matched bit-for-bit; serial and multi-threaded RaiSim
  checks matched final state hashes. Action-on versus action-off checks changed trajectories.

[Open the interactive random-action report](/raisim_newton_random_actions.html)
· [Download random-action measurements, per-trial timings, contacts and validation](/raisim_newton_random_actions.json).
Component-price assumptions remain **$339.99 CPU / $899.99 GPU**, with the availability
limitations described above. Higher FPS/USD is better; missing CPU results are unmeasured.

### ANYmal with random actions

<figure>
<img class="diagram" src="../../rsc/docs/image/raisim_newton_anymal_random_fps_per_usd.svg" alt="ANYmal random actions: RaiSim and Newton aggregate physics steps per second per USD across independent world counts; missing CPU measurements have no bars.">
<figcaption>Ten physics steps per action; random generation, target update and physics are timed. CPU values beyond 4,096 worlds are unmeasured.</figcaption>
</figure>

| Worlds | CPU workers | RaiSim FPS/USD | Newton FPS/USD | RaiSim / Newton FPS-per-dollar |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 954.41 | 3.76 | 253.51× |
| 64 | 16 | 5,600.58 | 148.35 | 37.75× |
| 256 | 20 | 10,535.55 | 514.19 | 20.49× |
| 1,024 | 20 | 10,466.75 | 1,596.15 | 6.56× |
| 4,096 | 20 | 10,845.10 | 3,188.73 | 3.40× |
| 8,192 | — | Not measured | 3,862.81 | — |
| 16,384 | — | Not measured | 4,134.67 | — |
| 32,768 | — | Not measured | 4,111.08 | — |

### Strandbeest with random actions

<figure>
<img class="diagram" src="../../rsc/docs/image/raisim_newton_strandbeest_random_fps_per_usd.svg" alt="Strandbeest random actions: RaiSim and Newton aggregate physics steps per second per USD across independent world counts; missing CPU measurements have no bars.">
<figcaption>Ten physics steps per action; random generation, target update and physics are timed. CPU values beyond 4,096 worlds are unmeasured.</figcaption>
</figure>

| Worlds | CPU workers | RaiSim FPS/USD | Newton FPS/USD | RaiSim / Newton FPS-per-dollar |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 117.02 | 3.29 | 35.56× |
| 64 | 16 | 1,207.31 | 122.08 | 9.89× |
| 256 | 20 | 1,289.24 | 352.62 | 3.66× |
| 1,024 | 20 | 1,410.19 | 595.12 | 2.37× |
| 4,096 | 20 | 1,470.17 | 721.71 | 2.04× |
| 8,192 | — | Not measured | 737.88 | — |
| 16,384 | — | Not measured | 712.86 | — |

### Random-action validation and limits

All **25 benchmark cases** passed the finite-state, contact, timing and target-sequence
checks. GPU cases reported zero solver overflow; all Strandbeest closure gaps were below
1 mm. At 4,096 worlds, trajectory-mean contacts/world were **2.683 / 2.336** for ANYmal
(RaiSim / Newton) and **0.945 / 0.696** for Strandbeest. These differences are retained in
the dataset rather than treated as equal contact work.

Both engines warm with nominal actions: 1,100 ANYmal steps or 100 Strandbeest steps.
Each timed trial then restores the post-warmup ANYmal state or the initial Strandbeest
state, resets solver warm starts and the action counter, and replays the same action
sequence. Trials contain **5,000 ANYmal steps** or **3,000 Strandbeest steps** at every
world count. Results use medians of three trials, except single-world RaiSim uses seven.
No trials are discarded. Processes run serially; CPU workers are pinned, with one
independent world per environment and no engine-internal threading.

An extra untimed replay measures trajectory-mean contacts and checks finite states,
solver overflow and Strandbeest closure gaps. RaiSim closure gaps and finite states
are sampled at ten-step boundaries; Newton gaps/overflow are checked every physics
step, and its finite state is checked at the end. These are different validation
sampling rates, not a claim of identical accuracy. Full trial times and contact means
are included in the downloadable records. Equal action targets do not imply equal
contact trajectories, because the solvers, precision and constraint formulations differ.

The earlier fixed-target numbers use a different driver protocol (no CPU control-interval
barrier and varying CPU trial horizons). Differences between the two workloads therefore
include synchronization, action-update overhead and changed physics; they do not isolate
random actuation cost alone. No episode resets are timed, and no claim is made about
policy or full training throughput. The larger-world GPU memory safeguard is unchanged.

For the single-threaded CPU comparison, see [RaiSim vs MuJoCo](./Benchmark.md).

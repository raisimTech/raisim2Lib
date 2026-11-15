# RaiSim vs Newton

## Ten-step performance per USD

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

For the single-threaded CPU comparison, see [RaiSim vs MuJoCo](./Benchmark.md).

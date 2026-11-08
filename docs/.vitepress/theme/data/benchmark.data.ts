import { readFileSync } from 'node:fs'

// Build-time summary of public/benchmark_results.json (the recorded comparison
// behind sections/Benchmark.md) for the home page's speed chart.

export interface BenchmarkRow {
  id: string
  label: string
  raisimSeconds: number
  otherSeconds: number
  speedup: number
}

// How much faster than real time RaiSim ran one workload: simulated time over
// the median wall-clock time, which includes building the scene.
export interface RealTimeFactor {
  id: string
  timeStep: number
  factor: number
}

export interface BenchmarkSummary {
  measuredOn: string
  cpu: string
  compiler: string
  rows: BenchmarkRow[]
  realTime: RealTimeFactor
}

declare const data: BenchmarkSummary
export { data }

// Same order as the table on the Benchmark page, with shorter names for the chart.
const LABELS: [string, string][] = [
  ['chain10_speed', 'Chain, 10 links'],
  ['chain20_speed', 'Chain, 20 links'],
  ['anymal_standing', 'ANYmal standing'],
  ['anymal_falling', 'ANYmal falling'],
  ['heightmap_anymal_speed', 'ANYmal on heightmap'],
  ['primitive_speed', 'Primitive contacts'],
  ['strandbeest', 'Strandbeest'],
  ['heightmap_lidar', 'Heightmap LiDAR'],
  ['primitive_ray_scaling', 'Ray queries'],
  ['deformable_cloth', 'Deformable cloth'],
  ['deformable_cube_stack', 'Deformable cube stack'],
]

// The workload behind the home page's real-time figure. The JSON records only
// wall-clock time; its default step count and time step are in the Workloads
// table of sections/Benchmark.md.
const REAL_TIME_WORKLOAD = { id: 'heightmap_anymal_speed', steps: 200_000, timeStep: 0.002 }

interface Timing {
  median_seconds: number
}

interface Result {
  benchmark_id: string
  backend: string
  status: string
  exit_code: number
  scene_config: { args: string[] }
  timings: Timing
}

interface Report {
  build: { compiler: string; compiler_version: string }
  system: { cpu: string }
  results: Result[]
}

interface Recorded {
  measured_on: string
  excluded_results: { benchmark_id: string; backend: string }[]
  reports: Report[]
}

export default {
  watch: ['../../../public/benchmark_results.json'],
  load(files: string[]): BenchmarkSummary {
    const recorded: Recorded = JSON.parse(readFileSync(files[0], 'utf8'))
    const excluded = new Set(recorded.excluded_results.map((r) => `${r.benchmark_id}/${r.backend}`))
    const medians = new Map<string, number>()
    for (const report of recorded.reports) {
      for (const result of report.results) {
        const key = `${result.benchmark_id}/${result.backend}`
        if (excluded.has(key)) continue
        if (result.status !== 'ok' || result.exit_code !== 0) {
          throw new Error(`benchmark_results.json: unsuccessful result ${key}`)
        }
        if (result.benchmark_id === REAL_TIME_WORKLOAD.id && result.scene_config.args.length > 0) {
          throw new Error(`benchmark_results.json: ${key} ran with non-default arguments; update REAL_TIME_WORKLOAD`)
        }
        medians.set(key, result.timings.median_seconds)
      }
    }

    const rows: BenchmarkRow[] = []
    for (const [id, label] of LABELS) {
      const raisimSeconds = medians.get(`${id}/raisim`)
      const otherSeconds = medians.get(`${id}/mujoco`)
      if (raisimSeconds === undefined) {
        throw new Error(`benchmark_results.json: no valid RaiSim timing for ${id}`)
      }
      // A comparison without a valid timing on the other side has no ratio.
      if (otherSeconds === undefined) continue
      rows.push({ id, label, raisimSeconds, otherSeconds, speedup: otherSeconds / raisimSeconds })
    }

    const { id, steps, timeStep } = REAL_TIME_WORKLOAD
    const realTimeSeconds = medians.get(`${id}/raisim`)
    if (realTimeSeconds === undefined) {
      throw new Error(`benchmark_results.json: no valid RaiSim timing for ${id}`)
    }

    const report = recorded.reports[0]
    return {
      measuredOn: recorded.measured_on,
      cpu: report.system.cpu.replace(/\s+\d+-Core Processor$/, ''),
      compiler: `${report.build.compiler} ${report.build.compiler_version.split('.')[0]}`,
      rows,
      realTime: { id, timeStep, factor: (steps * timeStep) / realTimeSeconds },
    }
  },
}

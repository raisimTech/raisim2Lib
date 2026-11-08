<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import LazyVideo from '../LazyVideo.vue'
import { clips } from '../../media'
import { data as benchmark } from '../../data/benchmark.data'

// Bars use a logarithmic axis from 1x to AXIS_MAX, like the Benchmark page's chart.
const AXIS_MAX = 200
const ticks = [1, 10, 100]

function position(speedup: number): number {
  return (Math.log10(Math.max(1, speedup)) / Math.log10(AXIS_MAX)) * 100
}

function formatSpeedup(speedup: number): string {
  return speedup >= 100 ? speedup.toFixed(0) : speedup >= 10 ? speedup.toFixed(1) : speedup.toFixed(2)
}

const measured = new Date(`${benchmark.measuredOn}T00:00:00Z`).toLocaleDateString('en-US', {
  year: 'numeric',
  month: 'long',
  timeZone: 'UTC',
})

// RaiSim and the competition simulate deformables with different models (see
// the Benchmark page), so the chart says which of its rows do.
const deformable = benchmark.rows.filter((row) => row.id.startsWith('deformable_')).map((row) => row.label.toLowerCase())
const deformableNote =
  deformable.length === 0
    ? ''
    : `The ${deformable.join(' and ')} ${deformable.length === 1 ? 'row compares' : 'rows compare'} different deformable models.`

const chart = ref<HTMLElement>()
const drawn = ref(false)
let observer: IntersectionObserver | undefined

onMounted(() => {
  if (!chart.value) return
  observer = new IntersectionObserver(
    ([entry]) => {
      if (!entry.isIntersecting) return
      drawn.value = true
      observer?.disconnect()
    },
    { threshold: 0.3 },
  )
  observer.observe(chart.value)
})

onBeforeUnmount(() => observer?.disconnect())
</script>

<template>
  <section id="speed" v-in-view class="rs-section rs-speed">
    <div class="rs-container rs-speed__grid">
      <div class="rs-speed__intro">
        <p v-reveal class="rs-eyebrow">Performance</p>
        <h2 v-reveal="60" class="rs-h2">More experience per hour. <span class="rs-speed__dim">Faster policies.</span></h2>
        <p v-reveal="120" class="rs-lead">
          Reinforcement learning needs billions of steps. RaiSim's lean solvers give you more of them from the hardware
          you already have: <strong class="rs-lead-claim rs-grad-text">a quadruped learns to walk in 10 seconds</strong>, so you iterate on rewards and controllers while the
          idea is still fresh.
        </p>
        <div v-reveal="180" v-glow class="rs-card rs-speed__clip">
          <LazyVideo :src="clips.army.src" :poster="clips.army.poster" label="64 ANYmal robots stepping in place on rough terrain" />
          <div class="rs-speed__clip-shade" />
          <div class="rs-speed__clip-text">
            <strong>64 quadrupeds</strong>
            <span>Stepping on rough terrain, all in one world</span>
          </div>
        </div>
      </div>

      <div ref="chart" v-reveal="120" class="rs-card rs-speed__chart" :data-drawn="drawn || undefined">
        <div class="rs-speed__chart-head">
          <div>
            <h3>Speedup over the competition</h3>
            <p>The same robotics workloads on the same machine; higher is better.</p>
          </div>
          <span class="rs-speed__legend">log scale</span>
        </div>

        <div class="rs-speed__rows">
          <div class="rs-speed__axis" aria-hidden="true">
            <span v-for="tick in ticks" :key="tick" :style="{ left: `${position(tick)}%` }">
              <em>{{ tick }}×</em>
            </span>
          </div>
          <div
            v-for="(row, index) in benchmark.rows"
            :key="row.id"
            class="rs-speed__row"
            :style="{ '--rs-row-delay': `${index * 70}ms` }"
          >
            <span class="rs-speed__label">{{ row.label }}</span>
            <span class="rs-speed__track">
              <span class="rs-speed__bar" :style="{ width: `${position(row.speedup)}%` }" />
            </span>
            <span class="rs-speed__value">{{ formatSpeedup(row.speedup) }}×</span>
          </div>
        </div>

        <p class="rs-speed__foot">
          {{ benchmark.cpu }}, median of three runs, {{ measured }}. {{ deformableNote }}
          <a class="rs-link" :href="withBase('/sections/Benchmark.html')">
            <span>Workloads and method</span>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </a>
        </p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-speed__grid {
  display: grid;
  grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
  gap: 56px;
  align-items: start;
}

.rs-speed__dim {
  color: var(--rs-text-3);
}

.rs-speed__clip {
  position: relative;
  margin-top: 40px;
  overflow: hidden;
  aspect-ratio: 16 / 9;
  background: #000;
}

.rs-speed__clip :deep(video) {
  position: absolute;
  inset: 0;
}

.rs-speed__clip-shade {
  position: absolute;
  inset: 0;
  background: linear-gradient(0deg, rgba(5, 5, 7, 0.9), rgba(5, 5, 7, 0) 55%);
}

.rs-speed__clip-text {
  position: absolute;
  left: 22px;
  right: 22px;
  bottom: 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.rs-speed__clip-text strong {
  font-size: 19px;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: #fff;
}

.rs-speed__clip-text span {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.72);
}

.rs-speed__chart {
  padding: 30px 30px 26px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.01));
}

.rs-speed__chart-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 30px;
}

.rs-speed__chart-head h3 {
  margin: 0;
  font-size: 19px;
  font-weight: 700;
  letter-spacing: -0.015em;
}

.rs-speed__chart-head p {
  margin: 6px 0 0;
  font-size: 14px;
  color: var(--rs-text-3);
}

.rs-speed__legend {
  flex: none;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--rs-text-2);
  border: 1px solid var(--rs-line-2);
}

.rs-speed__rows {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 13px;
  padding-bottom: 30px;
  --rs-label-w: 170px;
  --rs-value-w: 58px;
}

.rs-speed__axis {
  position: absolute;
  top: -8px;
  bottom: 0;
  left: calc(var(--rs-label-w) + 14px);
  right: calc(var(--rs-value-w) + 14px);
  pointer-events: none;
}

.rs-speed__axis span {
  position: absolute;
  top: 0;
  bottom: 0;
  border-left: 1px dashed rgba(255, 255, 255, 0.12);
}

.rs-speed__axis em {
  position: absolute;
  bottom: 0;
  left: 0;
  transform: translateX(-50%);
  font-size: 12px;
  font-style: normal;
  color: var(--rs-text-3);
}

.rs-speed__row {
  position: relative;
  display: grid;
  grid-template-columns: var(--rs-label-w) minmax(0, 1fr) var(--rs-value-w);
  align-items: center;
  gap: 14px;
}

.rs-speed__label {
  font-size: 14px;
  color: var(--rs-text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.rs-speed__track {
  position: relative;
  height: 12px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.05);
}

.rs-speed__bar {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  min-width: 6px;
  border-radius: 6px;
  background: var(--rs-grad);
  transform-origin: 0 50%;
  transition: transform 1.3s var(--rs-ease);
  transition-delay: var(--rs-row-delay);
  box-shadow: 0 0 18px -4px rgba(242, 39, 127, 0.55);
}

.rs-speed__value {
  font-size: 15px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  text-align: right;
  transition: opacity 0.6s ease, transform 0.6s var(--rs-ease);
  transition-delay: calc(var(--rs-row-delay) + 0.7s);
}

.rs-speed__foot {
  margin: 18px 0 0;
  font-size: 13px;
  line-height: 1.6;
  color: var(--rs-text-3);
}

.rs-speed__foot .rs-link {
  margin-left: 6px;
  font-size: 14px;
}

@media (max-width: 1100px) {
  .rs-speed__grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 40px;
  }
}

@media (max-width: 560px) {
  .rs-speed__chart {
    padding: 22px 18px 20px;
  }

  .rs-speed__rows {
    --rs-label-w: 112px;
    --rs-value-w: 46px;
    gap: 11px;
  }

  .rs-speed__row {
    gap: 10px;
  }

  .rs-speed__axis {
    left: calc(var(--rs-label-w) + 10px);
    right: calc(var(--rs-value-w) + 10px);
  }

  .rs-speed__label {
    font-size: 12px;
    white-space: normal;
    line-height: 1.25;
  }

  .rs-speed__value {
    font-size: 13px;
  }
}
</style>

<style>
/*
 * The bars grow when the chart scrolls into view. The hidden starting state
 * needs `rs-js` on <html> (see config.mts), so without JavaScript the chart is
 * drawn; a scoped style cannot select <html>.
 */
@media (prefers-reduced-motion: no-preference) {
  .rs-js .rs-speed__chart:not([data-drawn]) .rs-speed__bar {
    transform: scaleX(0);
  }

  .rs-js .rs-speed__chart:not([data-drawn]) .rs-speed__value {
    opacity: 0;
    transform: translateX(-8px);
  }
}
</style>

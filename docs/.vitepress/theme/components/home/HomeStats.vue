<script setup lang="ts">
import CountUp from '../CountUp.vue'
import { data as benchmark } from '../../data/benchmark.data'

function speedup(id: string): number {
  const row = benchmark.rows.find((r) => r.id === id)
  if (!row) throw new Error(`Benchmark row ${id} is missing`)
  return row.speedup
}

const realTime = benchmark.realTime

const stats = [
  {
    // The user's figure (2026-10-25); the repository has no measurement of it yet.
    value: 10,
    decimals: 0,
    unit: ' s',
    label: 'to learn quadrupedal locomotion',
    note: 'A walking quadruped trained from scratch with raisimGymTorch and PPO',
  },
  {
    // Server throughput reported by the user (2026-10-28).
    value: 10,
    decimals: 0,
    unit: 'k×',
    label: 'faster than real time',
    note: `Multithreaded server throughput; ${realTime.factor.toFixed(0)}× in the single-thread ANYmal benchmark`,
  },
  {
    value: speedup('heightmap_lidar'),
    decimals: 1,
    unit: '×',
    label: 'faster LiDAR than the competition',
    note: '64-beam spinning LiDAR over a 200 m terrain, for perception in the loop',
  },
]
</script>

<template>
  <section id="stats" class="rs-stats">
    <div class="rs-container rs-stats__grid">
      <div v-for="(stat, index) in stats" :key="stat.label" v-reveal="index * 90" v-glow class="rs-card rs-stats__card">
        <div class="rs-stats__value">
          <CountUp :to="stat.value" :decimals="stat.decimals" /><span class="rs-stats__unit">{{ stat.unit }}</span>
        </div>
        <div class="rs-stats__label">{{ stat.label }}</div>
        <div class="rs-stats__note">{{ stat.note }}</div>
      </div>
      <div v-reveal="270" v-glow class="rs-card rs-stats__card">
        <div class="rs-stats__value">Free</div>
        <div class="rs-stats__label">for academic research</div>
        <div class="rs-stats__note">Academic licenses cost nothing, and a one-month trial is free for everyone</div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-stats {
  position: relative;
  z-index: 2;
  padding-top: 12px;
}

.rs-stats__grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.rs-stats__card {
  padding: 28px 26px 26px;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.015));
}

.rs-stats__value {
  font-size: clamp(42px, 4.6vw, 60px);
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.045em;
  background: var(--rs-grad);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.rs-stats__unit {
  margin-left: 2px;
  font-weight: 700;
}

.rs-stats__label {
  margin-top: 16px;
  font-size: 17px;
  font-weight: 650;
  line-height: 1.35;
  color: var(--rs-text);
}

.rs-stats__note {
  margin-top: 6px;
  font-size: 14px;
  line-height: 1.5;
  color: var(--rs-text-3);
}

@media (max-width: 1023px) {
  .rs-stats__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 520px) {
  .rs-stats__grid {
    gap: 12px;
  }

  .rs-stats__card {
    padding: 20px 18px;
  }

  .rs-stats__value {
    font-size: 38px;
  }

  .rs-stats__label {
    margin-top: 12px;
    font-size: 15px;
  }

  .rs-stats__note {
    font-size: 13px;
  }
}
</style>

<script setup lang="ts">
import { withBase } from 'vitepress'
import FeatureIcon, { type FeatureIconName } from '../FeatureIcon.vue'

const features: { icon: FeatureIconName; title: string; text: string; link: string }[] = [
  {
    icon: 'articulated',
    title: 'Articulated systems',
    text: 'Legged robots, arms, hands and tracked vehicles, floating or fixed, with any joint type.',
    link: '/sections/ArticulatedSystem.html',
  },
  {
    icon: 'contact',
    title: 'Contact and friction',
    text: 'Hard contacts with Coulomb, rolling and spinning friction, for walking, grasping and pushing.',
    link: '/sections/Contact.html',
  },
  {
    icon: 'deformable',
    title: 'Deformable bodies',
    text: 'Cloth, soft solids and deformable terrain that collide with rigid bodies and robots.',
    link: '/sections/DeformableObject.html',
  },
  {
    icon: 'granular',
    title: 'Granular media',
    text: 'Sand and gravel as individual grains that pile up, flow and push back on robots.',
    link: '/sections/GranularMedia.html',
  },
  {
    icon: 'terrain',
    title: 'Heightmap terrain',
    text: 'Large, rotatable heightmaps, generated procedurally or loaded from images and files.',
    link: '/sections/HeightMap.html',
  },
  {
    icon: 'lidar',
    title: 'Sensors',
    text: 'Spinning LiDAR, RGB and depth cameras and IMUs with noise models, for perception in the loop.',
    link: '/sections/Sensors.html',
  },
  {
    icon: 'tendon',
    title: 'Tendons and constraints',
    text: 'Spatial tendons over pulleys, pins, mimic joints and length constraints.',
    link: '/sections/Tendons.html',
  },
  {
    icon: 'import',
    title: 'Model import',
    text: 'URDF, MJCF and OpenUSD robots, plus XML worlds and .rscene scenes.',
    link: '/sections/OpenUSD.html',
  },
  {
    icon: 'determinism',
    title: 'Deterministic',
    text: 'The same inputs reproduce the same trajectory. Save and restore whole worlds for resets and debugging.',
    link: '/sections/Determinism.html',
  },
  {
    icon: 'server',
    title: 'RaisimServer',
    text: 'Stream any world to the rayrai viewer over TCP, and pause, step or nudge it from there.',
    link: '/sections/RaisimServer.html',
  },
  {
    icon: 'learning',
    title: 'Python and RL',
    text: 'Vectorized C++ environments with PPO in PyTorch, plus RaisimPy for scripting.',
    link: '/sections/RaisimGymTorch.html',
  },
  {
    icon: 'platform',
    title: 'Cross-platform',
    text: 'Linux and Windows on x86-64; Linux and macOS (Apple silicon) on ARM64.',
    link: '/sections/Installation.html',
  },
]
</script>

<template>
  <section id="features" class="rs-section rs-features">
    <div class="rs-container">
      <div class="rs-features__head">
        <p v-reveal class="rs-eyebrow">Capabilities</p>
        <h2 v-reveal="60" class="rs-h2">Everything your robot needs to learn.</h2>
      </div>
      <div class="rs-features__grid">
        <a
          v-for="(feature, index) in features"
          :key="feature.title"
          v-reveal="(index % 4) * 70"
          v-in-view
          v-glow
          class="rs-card rs-feature"
          :href="withBase(feature.link)"
        >
          <span class="rs-feature__icon"><FeatureIcon :name="feature.icon" /></span>
          <span class="rs-feature__text">
            <span class="rs-feature__title">{{ feature.title }}</span>
            <span class="rs-feature__desc">{{ feature.text }}</span>
          </span>
        </a>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-features__head {
  margin-bottom: 52px;
}

.rs-features__grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.rs-feature {
  display: flex;
  flex-direction: column;
  gap: 22px;
  padding: 26px 24px 28px;
  color: inherit !important;
  text-decoration: none !important;
}

.rs-feature__icon {
  position: relative;
  display: grid;
  place-items: center;
  width: 64px;
  height: 64px;
  padding: 10px;
  border-radius: 18px;
  background: radial-gradient(circle at 30% 20%, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.02) 70%);
  border: 1px solid var(--rs-line-2);
  transition: transform 0.45s var(--rs-ease), box-shadow 0.45s var(--rs-ease);
}

.rs-feature__text {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.rs-feature__title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.015em;
  color: var(--rs-text);
}

.rs-feature__desc {
  font-size: 14.5px;
  line-height: 1.6;
  color: var(--rs-text-2);
}

@media (hover: hover) {
  .rs-feature:hover {
    transform: translateY(-4px);
  }

  .rs-feature:hover .rs-feature__icon {
    transform: scale(1.08) rotate(-3deg);
    box-shadow: 0 10px 30px -10px rgba(242, 39, 127, 0.55);
  }
}

@media (max-width: 1100px) {
  .rs-features__grid {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 820px) {
  .rs-features__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .rs-features__head {
    margin-bottom: 36px;
  }

  .rs-features__grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 10px;
  }

  .rs-feature {
    flex-direction: row;
    align-items: flex-start;
    gap: 16px;
    padding: 18px;
  }

  .rs-feature__icon {
    flex: none;
    width: 52px;
    height: 52px;
    padding: 8px;
    border-radius: 14px;
  }

  .rs-feature__title {
    font-size: 16px;
  }

  .rs-feature__desc {
    font-size: 14px;
  }
}
</style>

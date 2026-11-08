<script setup lang="ts">
import { withBase } from 'vitepress'
import LazyVideo from '../LazyVideo.vue'
import { clips } from '../../media'

const tiles = [
  {
    key: 'forest',
    clip: clips.forest,
    title: 'Dense foliage',
    text: 'Wind-blown trees and grass that cast real shadows, so outdoor perception sees clutter it will meet in the field.',
    tags: ['Instancing', 'Wind', 'LOD'],
  },
  {
    key: 'glass',
    clip: clips.glass,
    title: 'Glass and refraction',
    text: 'Glass and transparent objects refract light the way a camera sees them, a classic failure case for depth and grasping.',
    tags: ['Refraction', 'Ray queries'],
  },
  {
    key: 'rain',
    small: true,
    clip: clips.rain,
    title: 'Weather',
    text: 'Rain, fog, snow and time of day, for policies that hold up in bad weather.',
    tags: ['Rain', 'Wet materials'],
  },
  {
    key: 'city',
    small: true,
    clip: clips.city,
    title: 'Photoreal scenes',
    text: 'Cities, forests and warehouses with physically based materials and lighting.',
    tags: ['PBR', 'IBL'],
  },
  {
    key: 'lidar',
    clip: clips.lidar,
    title: 'LiDAR point clouds',
    text: 'Spinning LiDAR that scans the same world the robot walks in, ready for mapping and navigation.',
    tags: ['Sensors'],
  },
  {
    key: 'depth',
    small: true,
    clip: clips.depth,
    title: 'Depth camera',
    text: 'Depth aligned with the RGB camera, for terrain perception and grasping.',
    tags: ['Sensors'],
  },
  {
    key: 'rgb',
    small: true,
    clip: clips.rgb,
    title: 'RGB camera',
    text: 'Robot-mounted cameras rendered headless on a server, with sensor noise.',
    tags: ['Sensors'],
  },
]
</script>

<template>
  <section id="showcase" class="rs-section rs-showcase">
    <div class="rs-container">
      <header class="rs-showcase__head">
        <div>
          <p v-reveal class="rs-eyebrow">Perception</p>
          <h2 v-reveal="60" class="rs-h2">Train on images that look real.</h2>
        </div>
        <div v-reveal="120" class="rs-showcase__intro">
          <p class="rs-lead">
            rayrai, RaiSim's renderer, turns the running simulation into RGB, depth and LiDAR in the same process,
            headless on a training server or in a window, with the lighting, weather and materials your robot will meet
            outside.
          </p>
          <a class="rs-link" :href="withBase('/sections/Rayrai.html')">
            <span>Explore rayrai</span>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </a>
        </div>
      </header>

      <div class="rs-bento">
        <article
          v-for="(tile, index) in tiles"
          :key="tile.key"
          v-reveal="(index % 4) * 80"
          v-glow
          class="rs-card rs-bento__tile"
          :class="[`rs-bento__tile--${tile.key}`, { 'rs-bento__tile--small': tile.small }]"
        >
          <div class="rs-bento__media">
            <LazyVideo :src="tile.clip.src" :poster="tile.clip.poster" :label="tile.title" />
          </div>
          <div class="rs-bento__shade" />
          <div class="rs-bento__body">
            <ul class="rs-bento__tags">
              <li v-for="tag in tile.tags" :key="tag">{{ tag }}</li>
            </ul>
            <h3 class="rs-bento__title">{{ tile.title }}</h3>
            <p class="rs-bento__text">{{ tile.text }}</p>
          </div>
        </article>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-showcase__head {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
  align-items: end;
  gap: 32px 64px;
  margin-bottom: 56px;
}

.rs-showcase__intro .rs-lead {
  margin: 0 0 20px;
}

.rs-bento {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  grid-auto-rows: clamp(190px, 17vw, 270px);
  gap: 16px;
}

.rs-bento__tile {
  overflow: hidden;
  min-height: 0;
  background: #000;
}

.rs-bento__tile--forest {
  grid-column: 1 / 3;
  grid-row: 1 / 3;
}

.rs-bento__tile--glass {
  grid-column: 3 / 5;
}

.rs-bento__tile--lidar {
  grid-column: 1 / 3;
}

.rs-bento__media {
  position: absolute;
  inset: 0;
  transition: transform 1.2s var(--rs-ease);
}

.rs-bento__shade {
  position: absolute;
  inset: 0;
  background: linear-gradient(0deg, rgba(5, 5, 7, 0.92) 0%, rgba(5, 5, 7, 0.55) 32%, rgba(5, 5, 7, 0) 62%);
  transition: opacity 0.4s ease;
}

.rs-bento__body {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 22px 24px;
}

.rs-bento__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 0 0 10px;
  padding: 0;
  list-style: none;
}

.rs-bento__tags li {
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 650;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.86);
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.14);
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
}

.rs-bento__title {
  margin: 0;
  font-size: 21px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #fff;
}

.rs-bento__tile--forest .rs-bento__title {
  font-size: clamp(24px, 2.4vw, 32px);
}

.rs-bento__text {
  margin: 6px 0 0;
  max-width: 460px;
  font-size: 14px;
  line-height: 1.55;
  color: rgba(255, 255, 255, 0.74);
}

@media (hover: hover) and (min-width: 768px) {
  .rs-bento__text {
    max-height: 0;
    opacity: 0;
    overflow: hidden;
    transition: max-height 0.5s var(--rs-ease), opacity 0.4s ease;
  }

  .rs-bento__tile--forest .rs-bento__text {
    max-height: 4.8em;
    opacity: 1;
  }

  .rs-bento__tile:hover .rs-bento__text {
    max-height: 4.8em;
    opacity: 1;
  }

  .rs-bento__tile:hover .rs-bento__media {
    transform: scale(1.05);
  }
}

/*
 * In the four-column layout the small tiles show only their tags and title on
 * touch screens, which cannot reveal the description on hover, and below
 * 1100 px, where the tags can wrap and leave no room for it.
 */
@media (min-width: 1024px) and (hover: none), (min-width: 1024px) and (max-width: 1099px) {
  .rs-bento__tile--small .rs-bento__text {
    display: none;
  }
}

@media (max-width: 1023px) {
  .rs-showcase__head {
    grid-template-columns: 1fr;
    margin-bottom: 40px;
  }

  .rs-bento {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    grid-auto-rows: clamp(200px, 30vw, 280px);
  }

  .rs-bento__tile--forest,
  .rs-bento__tile--glass,
  .rs-bento__tile--lidar {
    grid-column: span 2;
    grid-row: auto;
  }

  .rs-bento__tile--forest {
    grid-row: span 2;
  }
}

@media (max-width: 600px) {
  .rs-bento {
    grid-template-columns: minmax(0, 1fr);
    grid-auto-rows: auto;
    gap: 12px;
  }

  .rs-bento__tile,
  .rs-bento__tile--forest,
  .rs-bento__tile--glass,
  .rs-bento__tile--lidar {
    grid-column: auto;
    grid-row: auto;
    aspect-ratio: 4 / 3;
  }

  .rs-bento__body {
    padding: 18px;
  }
}
</style>

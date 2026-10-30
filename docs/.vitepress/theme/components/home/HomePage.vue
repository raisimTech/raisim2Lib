<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import HomeCode from './HomeCode.vue'
import HomeCta from './HomeCta.vue'
import HomeEcosystem from './HomeEcosystem.vue'
import HomeFeatures from './HomeFeatures.vue'
import HomeFooter from './HomeFooter.vue'
import HomeHallOfFame from './HomeHallOfFame.vue'
import HomeHero from './HomeHero.vue'
import HomePhysics from './HomePhysics.vue'
import HomeShowcase from './HomeShowcase.vue'
import HomeSpeed from './HomeSpeed.vue'
import HomeStats from './HomeStats.vue'
import '../../styles/home.css'

// html.rs-home-route switches the navigation bar to its over-the-video style. The
// inline script in config.mts sets it before the first paint on a direct visit;
// these hooks keep it right across client-side navigation.
onMounted(() => document.documentElement.classList.add('rs-home-route'))
onBeforeUnmount(() => document.documentElement.classList.remove('rs-home-route'))
</script>

<template>
  <div class="rs-home">
    <HomeHero />
    <HomePhysics />
    <HomeStats />
    <HomeEcosystem />
    <HomeSpeed />
    <HomeShowcase />
    <HomeHallOfFame />
    <HomeFeatures />
    <HomeCode>
      <template #cpp><slot name="cpp" /></template>
      <template #python><slot name="python" /></template>
    </HomeCode>
    <HomeCta />
    <HomeFooter />
  </div>
</template>

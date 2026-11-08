<script setup lang="ts">
// Animated line icons for the capability tiles. Each icon moves the way its
// feature does; the motion runs only while the tile is on screen (.is-in-view
// from v-in-view) and never with reduced motion.
export type FeatureIconName =
  | 'articulated'
  | 'contact'
  | 'deformable'
  | 'granular'
  | 'terrain'
  | 'lidar'
  | 'tendon'
  | 'import'
  | 'determinism'
  | 'server'
  | 'learning'
  | 'platform'

const props = defineProps<{ name: FeatureIconName }>()
const gradientId = `rs-ig-${props.name}`
const stroke = `url(#${gradientId})`
</script>

<template>
  <svg class="rs-icon" :class="`rs-icon--${name}`" viewBox="0 0 48 48" aria-hidden="true">
    <defs>
      <linearGradient :id="gradientId" x1="4" y1="44" x2="44" y2="4" gradientUnits="userSpaceOnUse">
        <stop offset="0" stop-color="#ccf722" />
        <stop offset="0.45" stop-color="#f2277f" />
        <stop offset="0.75" stop-color="#cf50d5" />
        <stop offset="1" stop-color="#5c70d3" />
      </linearGradient>
    </defs>

    <template v-if="name === 'articulated'">
      <rect class="rs-icon__line" x="6" y="38.5" width="16" height="4.5" rx="1.5" />
      <g class="rs-ic-arm1">
        <path class="rs-icon__line" d="M14 36 L22 20" />
        <g class="rs-ic-arm2">
          <path :stroke="stroke" class="rs-icon__accent" d="M22 20 L36 15 M36 15 l3.5 -4.5 M36 15 l5 1.5" />
          <circle class="rs-icon__joint" cx="22" cy="20" r="3" />
        </g>
      </g>
      <circle class="rs-icon__joint" cx="14" cy="36" r="3.2" />
    </template>

    <template v-else-if="name === 'contact'">
      <path class="rs-icon__line" d="M6 41.5 H42" />
      <ellipse class="rs-ic-ripple" :fill="stroke" cx="24" cy="41.5" rx="10" ry="1.8" />
      <g class="rs-ic-ball">
        <circle :fill="stroke" cx="24" cy="34" r="6.2" />
        <path d="M20.6 32.2 a4 4 0 0 1 3.6 -3.2" fill="none" stroke="#fff" stroke-opacity="0.7" stroke-width="1.6" stroke-linecap="round" />
      </g>
    </template>

    <template v-else-if="name === 'deformable'">
      <path class="rs-icon__line" d="M7 41.5 H41" />
      <g class="rs-ic-jelly">
        <rect :fill="stroke" fill-opacity="0.16" :stroke="stroke" class="rs-icon__accent" x="12" y="17" width="24" height="24" rx="6" />
        <path class="rs-icon__faint" d="M12 25 H36 M12 33 H36 M20 17 V41 M28 17 V41" />
      </g>
    </template>

    <template v-else-if="name === 'granular'">
      <path class="rs-icon__line" d="M13 5 L21 12.5 M35 5 L27 12.5" />
      <circle class="rs-ic-grain" :fill="stroke" cx="24" cy="12" r="1.7" />
      <circle class="rs-ic-grain rs-ic-grain--2" :fill="stroke" cx="22.6" cy="12" r="1.5" />
      <circle class="rs-ic-grain rs-ic-grain--3" :fill="stroke" cx="25.4" cy="12" r="1.5" />
      <circle class="rs-ic-grain rs-ic-grain--4" :fill="stroke" cx="24" cy="12" r="1.3" />
      <path :fill="stroke" fill-opacity="0.18" :stroke="stroke" class="rs-icon__accent" d="M6 42 Q24 22 42 42 Z" />
      <g :fill="stroke" fill-opacity="0.85">
        <circle cx="18" cy="37" r="1.3" /><circle cx="23" cy="34" r="1.3" /><circle cx="28" cy="36.5" r="1.3" />
        <circle cx="14" cy="40" r="1.3" /><circle cx="21" cy="39.5" r="1.3" /><circle cx="26" cy="40.2" r="1.3" />
        <circle cx="32" cy="39.6" r="1.3" /><circle cx="25.5" cy="30.6" r="1.1" />
      </g>
    </template>

    <template v-else-if="name === 'terrain'">
      <path class="rs-icon__faint" d="M4 24 C10 19 16 26 23 18 S35 13 44 18" />
      <path class="rs-icon__line" d="M4 32 C11 26 17 34 25 26 S37 22 44 26" />
      <path class="rs-icon__line" d="M4 40 C12 34 19 42 27 35 S39 31 44 34" />
      <path class="rs-ic-scan" :stroke="stroke" pathLength="100" d="M4 24 C10 19 16 26 23 18 S35 13 44 18" />
      <path class="rs-ic-scan rs-ic-scan--2" :stroke="stroke" pathLength="100" d="M4 32 C11 26 17 34 25 26 S37 22 44 26" />
      <path class="rs-ic-scan rs-ic-scan--3" :stroke="stroke" pathLength="100" d="M4 40 C12 34 19 42 27 35 S39 31 44 34" />
    </template>

    <template v-else-if="name === 'lidar'">
      <circle class="rs-icon__faint rs-icon__dash" cx="24" cy="24" r="18" />
      <circle class="rs-icon__faint rs-icon__dash" cx="24" cy="24" r="10.5" />
      <g class="rs-ic-sweep">
        <path :fill="stroke" fill-opacity="0.32" d="M24 24 L24 6 A18 18 0 0 0 12.43 10.21 Z" />
        <path :stroke="stroke" class="rs-icon__accent" d="M24 24 L24 6" />
      </g>
      <circle class="rs-ic-hit" :fill="stroke" cx="33" cy="14" r="1.8" style="animation-delay: 0.28s" />
      <circle class="rs-ic-hit" :fill="stroke" cx="36" cy="32" r="1.8" style="animation-delay: 0.82s" />
      <circle class="rs-ic-hit" :fill="stroke" cx="12" cy="30" r="1.8" style="animation-delay: 1.62s" />
      <circle class="rs-ic-hit" :fill="stroke" cx="15" cy="13" r="1.8" style="animation-delay: 2.14s" />
      <circle class="rs-icon__joint" cx="24" cy="24" r="3.4" />
    </template>

    <template v-else-if="name === 'tendon'">
      <g class="rs-ic-wheel">
        <circle :stroke="stroke" class="rs-icon__accent" cx="30" cy="14" r="8" />
        <path class="rs-icon__faint" d="M30 6.5 V21.5 M22.5 14 H37.5" />
      </g>
      <circle class="rs-icon__joint" cx="30" cy="14" r="2" />
      <path class="rs-icon__line rs-ic-rope" d="M9 42 L22.4 15.6 A8 8 0 0 1 38 14" />
      <path class="rs-icon__line rs-ic-drop" d="M38 14 V30" />
      <g class="rs-ic-weight">
        <rect :fill="stroke" x="33" y="30" width="10" height="10" rx="2.2" />
      </g>
    </template>

    <template v-else-if="name === 'import'">
      <g class="rs-ic-sheet rs-ic-sheet--1">
        <path class="rs-icon__line rs-icon__sheet" d="M15 8 h12 l7 7 v23 a2 2 0 0 1 -2 2 h-17 a2 2 0 0 1 -2 -2 v-28 a2 2 0 0 1 2 -2 z" />
      </g>
      <g class="rs-ic-sheet rs-ic-sheet--3">
        <path class="rs-icon__line rs-icon__sheet" d="M15 8 h12 l7 7 v23 a2 2 0 0 1 -2 2 h-17 a2 2 0 0 1 -2 -2 v-28 a2 2 0 0 1 2 -2 z" />
      </g>
      <g class="rs-ic-sheet rs-ic-sheet--2">
        <path :stroke="stroke" class="rs-icon__accent rs-icon__sheet" d="M15 8 h12 l7 7 v23 a2 2 0 0 1 -2 2 h-17 a2 2 0 0 1 -2 -2 v-28 a2 2 0 0 1 2 -2 z" />
        <path :stroke="stroke" class="rs-icon__accent" d="M27 8 v7 h7" />
        <path class="rs-icon__faint" d="M18 23 h11 M18 28 h11 M18 33 h7" />
      </g>
    </template>

    <template v-else-if="name === 'determinism'">
      <path class="rs-icon__faint" d="M4 20 C10 4 15 6 19 15 S29 26 34 12 S41 6 44 9" />
      <path class="rs-icon__faint" d="M4 38 C10 22 15 24 19 33 S29 44 34 30 S41 24 44 27" />
      <path class="rs-ic-trace" :stroke="stroke" pathLength="100" d="M4 20 C10 4 15 6 19 15 S29 26 34 12 S41 6 44 9" />
      <path class="rs-ic-trace" :stroke="stroke" pathLength="100" d="M4 38 C10 22 15 24 19 33 S29 44 34 30 S41 24 44 27" />
    </template>

    <template v-else-if="name === 'server'">
      <path :fill="stroke" fill-opacity="0.18" :stroke="stroke" class="rs-icon__accent" d="M11 15 l7 4 v8 l-7 4 l-7 -4 v-8 z M4 19 l7 4 l7 -4 M11 23 v8" />
      <rect class="rs-icon__line" x="28" y="12" width="16" height="12" rx="2" />
      <path class="rs-icon__line" d="M33 30 h6 M36 24 v6" />
      <path class="rs-ic-flow" :stroke="stroke" d="M19 23 C23 23 24 18 28 18" />
      <path class="rs-ic-wave" :stroke="stroke" d="M8 9 a4.5 4.5 0 0 1 6 0" />
      <path class="rs-ic-wave rs-ic-wave--2" :stroke="stroke" d="M5.5 6 a8 8 0 0 1 11 0" />
    </template>

    <template v-else-if="name === 'learning'">
      <g class="rs-icon__faint">
        <path d="M8 14 L24 10 M8 14 L24 20 M8 14 L24 28 M8 14 L24 38 M8 24 L24 10 M8 24 L24 20 M8 24 L24 28 M8 24 L24 38" />
        <path d="M8 34 L24 10 M8 34 L24 20 M8 34 L24 28 M8 34 L24 38" />
        <path d="M24 10 L40 18 M24 20 L40 18 M24 28 L40 18 M24 38 L40 18 M24 10 L40 30 M24 20 L40 30 M24 28 L40 30 M24 38 L40 30" />
      </g>
      <g class="rs-ic-layer">
        <circle cx="8" cy="14" r="3" /><circle cx="8" cy="24" r="3" /><circle cx="8" cy="34" r="3" />
      </g>
      <g class="rs-ic-layer rs-ic-layer--2">
        <circle cx="24" cy="10" r="3" /><circle cx="24" cy="20" r="3" /><circle cx="24" cy="28" r="3" /><circle cx="24" cy="38" r="3" />
      </g>
      <g class="rs-ic-layer rs-ic-layer--3">
        <circle cx="40" cy="18" r="3" /><circle cx="40" cy="30" r="3" />
      </g>
      <circle :fill="stroke" class="rs-ic-node-fill" cx="8" cy="24" r="3" />
      <circle :fill="stroke" class="rs-ic-node-fill rs-ic-node-fill--2" cx="24" cy="20" r="3" />
      <circle :fill="stroke" class="rs-ic-node-fill rs-ic-node-fill--3" cx="40" cy="30" r="3" />
    </template>

    <template v-else-if="name === 'platform'">
      <rect :stroke="stroke" class="rs-icon__accent" x="13" y="13" width="22" height="22" rx="4" />
      <rect :fill="stroke" class="rs-ic-core" x="19" y="19" width="10" height="10" rx="2" />
      <path
        v-for="(pin, index) in [
          'M18 6 V12', 'M24 6 V12', 'M30 6 V12', 'M36 18 H42', 'M36 24 H42', 'M36 30 H42',
          'M30 36 V42', 'M24 36 V42', 'M18 36 V42', 'M6 30 H12', 'M6 24 H12', 'M6 18 H12',
        ]"
        :key="pin"
        class="rs-icon__line rs-ic-pin"
        :d="pin"
        :style="{ animationDelay: `${index * 0.15}s` }"
      />
    </template>
  </svg>
</template>

<style>
.rs-icon {
  width: 100%;
  height: 100%;
  overflow: visible;
}

/*
 * The `animation` shorthands below reset animation-play-state, so these rules
 * are more specific than any of them.
 */
svg.rs-icon * {
  animation-play-state: paused;
}

.is-in-view svg.rs-icon * {
  animation-play-state: running;
}

.rs-icon__line,
.rs-icon__accent,
.rs-icon__faint {
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.rs-icon__line {
  stroke: rgba(255, 255, 255, 0.82);
}

.rs-icon__faint {
  stroke: rgba(255, 255, 255, 0.24);
  stroke-width: 1.5;
}

.rs-icon__dash {
  stroke-dasharray: 2 3.2;
}

.rs-icon__sheet {
  fill: #12121a;
}

.rs-icon__joint {
  fill: #12121a;
  stroke: rgba(255, 255, 255, 0.9);
  stroke-width: 2;
}

/* articulated: a two-link arm reaching back and forth */
.rs-ic-arm1 {
  transform-origin: 14px 36px;
  animation: rs-ic-arm1 3.4s ease-in-out infinite alternate;
}

.rs-ic-arm2 {
  transform-origin: 22px 20px;
  animation: rs-ic-arm2 3.4s ease-in-out infinite alternate;
}

@keyframes rs-ic-arm1 {
  from { transform: rotate(-12deg); }
  to { transform: rotate(14deg); }
}

@keyframes rs-ic-arm2 {
  from { transform: rotate(20deg); }
  to { transform: rotate(-30deg); }
}

/* contact: a ball that drops, squashes and rebounds */
.rs-ic-ball {
  transform-origin: 24px 40.5px;
  animation: rs-ic-bounce 1.4s infinite;
}

.rs-ic-ripple {
  opacity: 0;
  transform-origin: 24px 41.5px;
  animation: rs-ic-ripple 1.4s ease-out infinite;
}

@keyframes rs-ic-bounce {
  0%, 100% { transform: translateY(-21px); animation-timing-function: cubic-bezier(0.55, 0, 0.9, 0.45); }
  46% { transform: translateY(0) scale(1, 1); }
  53% { transform: translateY(0) scale(1.18, 0.8); animation-timing-function: cubic-bezier(0.1, 0.55, 0.45, 1); }
  62% { transform: translateY(-4px) scale(0.95, 1.06); }
}

@keyframes rs-ic-ripple {
  0%, 44% { opacity: 0; transform: scaleX(0.3); }
  52% { opacity: 0.85; transform: scaleX(0.7); }
  85%, 100% { opacity: 0; transform: scaleX(1.5); }
}

/* deformable: a soft cube that squashes and recovers */
.rs-ic-jelly {
  transform-origin: 24px 41px;
  animation: rs-ic-jelly 2.6s ease-in-out infinite;
}

@keyframes rs-ic-jelly {
  0%, 100% { transform: scale(1, 1); }
  18% { transform: scale(1.16, 0.82); }
  36% { transform: scale(0.9, 1.12); }
  54% { transform: scale(1.06, 0.95); }
  72% { transform: scale(0.98, 1.02); }
}

/* granular: grains streaming onto a pile */
.rs-ic-grain {
  opacity: 0;
  animation: rs-ic-fall 1.2s cubic-bezier(0.5, 0, 0.9, 0.6) infinite;
}

.rs-ic-grain--2 { animation-delay: 0.3s; }
.rs-ic-grain--3 { animation-delay: 0.6s; }
.rs-ic-grain--4 { animation-delay: 0.9s; }

@keyframes rs-ic-fall {
  0% { opacity: 0; transform: translateY(0); }
  15% { opacity: 1; }
  80% { opacity: 1; }
  100% { opacity: 0; transform: translateY(19px); }
}

/* terrain: light running along the contour lines */
.rs-ic-scan {
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-dasharray: 18 100;
  stroke-dashoffset: 118;
  animation: rs-ic-scan 3s ease-in-out infinite;
}

.rs-ic-scan--2 { animation-delay: 0.35s; }
.rs-ic-scan--3 { animation-delay: 0.7s; }

@keyframes rs-ic-scan {
  to { stroke-dashoffset: 0; }
}

/* lidar: a rotating beam; returns light up as it passes */
.rs-ic-sweep {
  transform-origin: 24px 24px;
  animation: rs-ic-spin 2.4s linear infinite;
}

.rs-ic-hit {
  opacity: 0.15;
  transform-box: fill-box;
  transform-origin: center;
  animation: rs-ic-hit 2.4s ease-out infinite;
}

@keyframes rs-ic-spin {
  to { transform: rotate(360deg); }
}

@keyframes rs-ic-hit {
  0% { opacity: 1; transform: scale(1.7); }
  45%, 100% { opacity: 0.15; transform: scale(1); }
}

/* tendon: a pulley turning while the load rises and falls */
.rs-ic-wheel {
  transform-origin: 30px 14px;
  animation: rs-ic-wheel 1.8s ease-in-out infinite alternate;
}

.rs-ic-rope {
  stroke-dasharray: 3 2.6;
  animation: rs-ic-rope 1.8s ease-in-out infinite alternate;
}

.rs-ic-drop {
  transform-origin: 38px 14px;
  animation: rs-ic-drop 1.8s ease-in-out infinite alternate;
}

.rs-ic-weight {
  animation: rs-ic-weight 1.8s ease-in-out infinite alternate;
}

@keyframes rs-ic-wheel {
  to { transform: rotate(43deg); }
}

@keyframes rs-ic-rope {
  to { stroke-dashoffset: -6; }
}

@keyframes rs-ic-drop {
  to { transform: scaleY(1.375); }
}

@keyframes rs-ic-weight {
  to { transform: translateY(6px); }
}

/* import: model files fanning out */
.rs-ic-sheet {
  transform-origin: 24px 40px;
  animation: rs-ic-fan 3.2s var(--rs-ease) infinite alternate;
}

.rs-ic-sheet--1 { --rs-fan: -16deg; --rs-shift: -3px; }
.rs-ic-sheet--2 { --rs-fan: 0deg; --rs-shift: 0px; }
.rs-ic-sheet--3 { --rs-fan: 16deg; --rs-shift: 3px; }

@keyframes rs-ic-fan {
  0%, 25% { transform: none; }
  75%, 100% { transform: translateX(var(--rs-shift)) rotate(var(--rs-fan)); }
}

/* determinism: two runs drawing the same trajectory in lockstep */
.rs-ic-trace {
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-dasharray: 100;
  stroke-dashoffset: 100;
  animation: rs-ic-draw 3.2s ease-in-out infinite;
}

@keyframes rs-ic-draw {
  0% { stroke-dashoffset: 100; opacity: 1; }
  60% { stroke-dashoffset: 0; opacity: 1; }
  85% { stroke-dashoffset: 0; opacity: 1; }
  100% { stroke-dashoffset: 0; opacity: 0; }
}

/* server: a world streaming frames to a viewer */
.rs-ic-flow {
  fill: none;
  stroke-width: 2.2;
  stroke-linecap: round;
  stroke-dasharray: 0.5 4;
  animation: rs-ic-flow 0.9s linear infinite;
}

.rs-ic-wave {
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  opacity: 0;
  animation: rs-ic-wave 1.8s ease-out infinite;
}

.rs-ic-wave--2 { animation-delay: 0.3s; }

@keyframes rs-ic-flow {
  to { stroke-dashoffset: -9; }
}

@keyframes rs-ic-wave {
  0% { opacity: 0; }
  30% { opacity: 1; }
  100% { opacity: 0; }
}

/* learning: activations moving through a network */
.rs-ic-layer circle {
  fill: #12121a;
  stroke: rgba(255, 255, 255, 0.82);
  stroke-width: 1.8;
}

.rs-ic-node-fill {
  opacity: 0;
  animation: rs-ic-fire 1.8s ease-in-out infinite;
}

.rs-ic-node-fill--2 { animation-delay: 0.4s; }
.rs-ic-node-fill--3 { animation-delay: 0.8s; }

@keyframes rs-ic-fire {
  0%, 100% { opacity: 0; }
  25%, 45% { opacity: 1; }
}

/* platform: pins of a chip lighting up in turn */
.rs-ic-pin {
  animation: rs-ic-pin 1.8s ease-in-out infinite;
}

.rs-ic-core {
  transform-box: fill-box;
  transform-origin: center;
  animation: rs-ic-core 1.8s ease-in-out infinite;
}

@keyframes rs-ic-pin {
  0%, 30%, 100% { stroke: rgba(255, 255, 255, 0.3); }
  10% { stroke: #f2277f; }
}

@keyframes rs-ic-core {
  0%, 100% { transform: scale(1); opacity: 0.9; }
  50% { transform: scale(0.82); opacity: 0.6; }
}

@media (prefers-reduced-motion: reduce) {
  .rs-icon * {
    animation: none !important;
  }

  .rs-ic-grain,
  .rs-ic-ripple,
  .rs-ic-wave,
  .rs-ic-node-fill {
    opacity: 1;
  }

  .rs-ic-trace,
  .rs-ic-scan {
    stroke-dashoffset: 0;
  }
}
</style>

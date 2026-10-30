<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { withBase } from 'vitepress'
import { heroClips } from '../../media'

// Full-screen background reel: the clips play one after another and cross-fade.
// Only the playing clip and the one after it are downloaded. Small screens and
// data-saver connections get the 960 px encodes.
const version = __RAISIM_VERSION__
const active = ref(0)
const progress = ref(0)
const paused = ref(false)
const reducedMotion = ref(false)
const sources = reactive<(string | undefined)[]>(heroClips.map(() => undefined))
const videos: (HTMLVideoElement | undefined)[] = []
const hero = ref<HTMLElement>()

let useSmall = false
let onScreen = true
let frame = 0
let observer: IntersectionObserver | undefined

function setVideo(index: number, element: unknown): void {
  videos[index] = (element as HTMLVideoElement | null) ?? undefined
}

function load(index: number): void {
  sources[index] ??= useSmall ? heroClips[index].srcSmall : heroClips[index].src
}

function canPlay(): boolean {
  return !reducedMotion.value && !paused.value && onScreen && !document.hidden
}

function syncPlayback(): void {
  const video = videos[active.value]
  if (!video) return
  if (canPlay()) {
    video.muted = true
    video.play().catch(() => {})
  } else {
    video.pause()
  }
}

function select(index: number): void {
  if (index === active.value) return
  videos[active.value]?.pause()
  active.value = index
  progress.value = 0
  load(index)
  nextTick(() => {
    const video = videos[index]
    if (video && video.readyState > 0) video.currentTime = 0
    syncPlayback()
  })
}

function onEnded(index: number): void {
  if (index === active.value) select((index + 1) % heroClips.length)
}

function togglePause(): void {
  paused.value = !paused.value
  syncPlayback()
}

// The progress bar follows the active clip frame by frame while it plays; the
// loop stops when the clip pauses or ends and restarts on its `playing` event.
function tick(): void {
  frame = 0
  const video = videos[active.value]
  if (!video || video.paused || video.ended) return
  if (video.duration > 0) {
    progress.value = Math.min(1, video.currentTime / video.duration)
    if (progress.value > 0.3) load((active.value + 1) % heroClips.length)
  }
  frame = requestAnimationFrame(tick)
}

function onPlaying(index: number): void {
  if (index === active.value && frame === 0) frame = requestAnimationFrame(tick)
}

function onVisibilityChange(): void {
  syncPlayback()
}

onMounted(() => {
  reducedMotion.value = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  const connection = (navigator as Navigator & { connection?: { saveData?: boolean } }).connection
  useSmall = window.matchMedia('(max-width: 900px)').matches || connection?.saveData === true
  if (reducedMotion.value) return
  load(0)
  nextTick(syncPlayback)
  document.addEventListener('visibilitychange', onVisibilityChange)
  if (hero.value) {
    observer = new IntersectionObserver(([entry]) => {
      onScreen = entry.isIntersecting
      syncPlayback()
    })
    observer.observe(hero.value)
  }
})

onBeforeUnmount(() => {
  cancelAnimationFrame(frame)
  observer?.disconnect()
  document.removeEventListener('visibilitychange', onVisibilityChange)
})
</script>

<template>
  <section ref="hero" v-in-view class="rs-hero" aria-label="RaiSim">
    <div class="rs-hero__media" aria-hidden="true">
      <video
        v-for="(clip, index) in heroClips"
        :key="clip.title"
        :ref="(element) => setVideo(index, element)"
        class="rs-hero__video"
        :class="{ 'is-active': index === active }"
        :preload="sources[index] ? 'auto' : 'none'"
        :src="sources[index]"
        :poster="clip.poster"
        muted
        playsinline
        disablepictureinpicture
        @playing="onPlaying(index)"
        @ended="onEnded(index)"
      />
    </div>
    <div class="rs-hero__scrim" />

    <div class="rs-container rs-hero__content">
      <a class="rs-hero__chip" :href="withBase('/sections/Changelog.html')">
        <span class="rs-hero__chip-tag">v{{ version }}</span>
        <span>Read the release notes</span>
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3l5 5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
      </a>
      <h1 class="rs-hero__title">
        <span class="rs-hero__line">Train robots</span>
        <span class="rs-hero__line"><span class="rs-grad-text">faster than reality.</span></span>
      </h1>
      <p class="rs-hero__lead">
        <strong class="rs-hero__claim">Quadrupedal locomotion, learned in 10 seconds.</strong>
        Contact-rich robot physics at 10,000× real time on a multithreaded server, with photoreal cameras and LiDAR.
      </p>
      <div class="rs-hero__actions">
        <a class="rs-btn rs-btn--primary" :href="withBase('/sections/QuickStart.html')">
          Get started
          <svg class="rs-arrow" viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11M11 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
        </a>
        <a class="rs-btn rs-btn--ghost" href="#physics">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 4.8v10.4L15.5 10z" fill="currentColor" /></svg>
          See it in action
        </a>
      </div>
    </div>

    <div class="rs-container rs-hero__bar">
      <p class="rs-hero__caption">
        <span class="rs-hero__live" aria-hidden="true" />
        <span>{{ heroClips[active].caption }}</span>
        <span class="rs-hero__caption-note">Simulated with RaiSim, rendered with rayrai</span>
      </p>
      <div class="rs-hero__nav">
        <button
          v-for="(clip, index) in heroClips"
          :key="clip.title"
          class="rs-hero__step"
          :class="{ 'is-active': index === active }"
          type="button"
          :aria-label="`Show the ${clip.title} clip`"
          :aria-pressed="index === active"
          @click="select(index)"
        >
          <span class="rs-hero__step-label">
            <span class="rs-hero__step-num">0{{ index + 1 }}</span>{{ clip.title }}
          </span>
          <span class="rs-hero__track">
            <span
              class="rs-hero__fill"
              :style="{ transform: `scaleX(${index === active ? (reducedMotion ? 1 : progress) : 0})` }"
            />
          </span>
        </button>
        <button
          v-if="!reducedMotion"
          class="rs-hero__pause"
          type="button"
          :aria-label="paused ? 'Play the background video' : 'Pause the background video'"
          @click="togglePause"
        >
          <svg v-if="paused" viewBox="0 0 20 20" aria-hidden="true"><path d="M7 4.8v10.4L15.5 10z" fill="currentColor" /></svg>
          <svg v-else viewBox="0 0 20 20" aria-hidden="true"><path d="M6.5 4.5h2.5v11H6.5zM11 4.5h2.5v11H11z" fill="currentColor" /></svg>
        </button>
      </div>
    </div>

  </section>
</template>

<style scoped>
/*
 * One screen tall (up to 1120 px), with the content at the bottom. On short
 * screens the hero grows with its content instead, and the top padding keeps the
 * content clear of the fixed navigation bar.
 */
.rs-hero {
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  min-height: min(100vh, 1120px);
  min-height: min(100svh, 1120px);
  padding-top: calc(var(--vp-nav-height) + 16px);
  overflow: hidden;
  background: #000;
}

.rs-hero__media,
.rs-hero__scrim {
  position: absolute;
  inset: 0;
}

.rs-hero__video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  opacity: 0;
  transform: scale(1.08);
  transition: opacity 1.4s ease, transform 1.4s ease;
}

.rs-hero__video.is-active {
  opacity: 1;
  transform: scale(1);
  transition: opacity 1.4s ease, transform 11s linear;
}

.rs-hero__scrim {
  background:
    /* A darker band under the navigation bar keeps the menu legible over bright frames. */
    linear-gradient(180deg, rgba(5, 5, 7, 0.78) 0%, rgba(5, 5, 7, 0.5) 9%, rgba(5, 5, 7, 0) 24%),
    linear-gradient(0deg, #050507 0%, rgba(5, 5, 7, 0.7) 10%, rgba(5, 5, 7, 0.12) 34%, rgba(5, 5, 7, 0) 46%),
    linear-gradient(90deg, rgba(5, 5, 7, 0.6) 0%, rgba(5, 5, 7, 0.26) 30%, rgba(5, 5, 7, 0) 52%);
}

.rs-hero__content {
  position: relative;
  z-index: 1;
  padding-bottom: 32px;
}

.rs-hero__chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 4px 12px 4px 4px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 550;
  color: var(--rs-text) !important;
  text-decoration: none !important;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.16);
  -webkit-backdrop-filter: blur(14px);
  backdrop-filter: blur(14px);
  animation: rs-rise 0.9s var(--rs-ease) both;
  transition: background-color 0.25s ease, border-color 0.25s ease;
}

.rs-hero__chip:hover {
  background: rgba(255, 255, 255, 0.14);
  border-color: rgba(255, 255, 255, 0.3);
}

.rs-hero__chip svg {
  width: 14px;
  height: 14px;
  transition: transform 0.25s var(--rs-ease);
}

.rs-hero__chip:hover svg {
  transform: translateX(3px);
}

.rs-hero__chip-tag {
  padding: 2px 9px;
  border-radius: 999px;
  font-size: 11.5px;
  font-weight: 750;
  letter-spacing: 0.02em;
  color: #0b0b10;
  background: var(--rs-grad-warm);
}

.rs-hero__title {
  margin: 18px 0 0;
  font-size: clamp(34px, 4.8vw, 70px);
  font-weight: 850;
  line-height: 1;
  letter-spacing: -0.038em;
  color: #fff;
}

.rs-hero__line {
  display: block;
  padding-bottom: 0.06em;
  animation: rs-rise 1.1s var(--rs-ease) both;
  animation-delay: 0.08s;
}

.rs-hero__line + .rs-hero__line {
  animation-delay: 0.2s;
}

.rs-hero__lead {
  max-width: 460px;
  margin: 16px 0 0;
  font-size: clamp(15px, 1.2vw, 17px);
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.82);
  text-shadow: 0 1px 18px rgba(0, 0, 0, 0.4);
  animation: rs-rise 1.1s var(--rs-ease) 0.32s both;
}

.rs-hero__claim {
  display: block;
  margin-bottom: 4px;
  font-weight: 700;
  color: #fff;
}

.rs-hero__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 24px;
  animation: rs-rise 1.1s var(--rs-ease) 0.44s both;
}

.rs-hero__actions .rs-btn {
  height: 46px;
  padding: 0 22px;
  font-size: 15px;
}

.rs-hero__bar {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 32px;
  padding-bottom: 40px;
  animation: rs-fade 1.4s ease 0.7s both;
}

.rs-hero__caption {
  display: grid;
  grid-template-columns: 8px minmax(0, 1fr);
  align-items: center;
  gap: 4px 12px;
  margin: 0;
  max-width: 520px;
  font-size: 14px;
  line-height: 1.5;
  color: rgba(255, 255, 255, 0.86);
}

.rs-hero__caption-note {
  grid-column: 2;
  font-size: 12px;
  letter-spacing: 0.04em;
  color: rgba(255, 255, 255, 0.5);
}

.rs-hero__live {
  position: relative;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--rs-pink);
}

.rs-hero__live::after {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  border: 1px solid var(--rs-pink);
  animation: rs-ping 2s ease-out infinite;
}

.rs-hero__nav {
  display: flex;
  align-items: flex-end;
  gap: 18px;
}

.rs-hero__step {
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 112px;
  padding: 0;
  text-align: left;
  color: rgba(255, 255, 255, 0.55);
  background: none;
  border: 0;
  cursor: pointer;
  transition: color 0.3s ease;
}

.rs-hero__step:hover,
.rs-hero__step.is-active {
  color: #fff;
}

.rs-hero__step-label {
  font-size: 13px;
  font-weight: 600;
}

.rs-hero__step-num {
  margin-right: 8px;
  font-variant-numeric: tabular-nums;
  color: rgba(255, 255, 255, 0.45);
}

.rs-hero__track {
  position: relative;
  height: 2px;
  overflow: hidden;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.22);
}

.rs-hero__fill {
  position: absolute;
  inset: 0;
  transform-origin: 0 50%;
  background: var(--rs-grad);
}

.rs-hero__pause {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  margin-left: 4px;
  border-radius: 50%;
  color: #fff;
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.2);
  cursor: pointer;
  transition: background-color 0.25s ease;
}

.rs-hero__pause:hover {
  background: rgba(255, 255, 255, 0.18);
}

.rs-hero__pause svg {
  width: 16px;
  height: 16px;
}

@media (max-width: 1023px) {
  .rs-hero__bar {
    flex-direction: column;
    align-items: stretch;
    gap: 22px;
  }

  .rs-hero__caption {
    max-width: none;
  }

  .rs-hero__step {
    flex: 1;
    width: auto;
  }
}

@media (max-width: 640px) {
  /* On phones the text covers the lower half, so the shade reaches that far; the upper half stays clear. */
  .rs-hero__scrim {
    background:
      linear-gradient(180deg, rgba(5, 5, 7, 0.78) 0%, rgba(5, 5, 7, 0.45) 8%, rgba(5, 5, 7, 0) 20%),
      linear-gradient(0deg, #050507 0%, rgba(5, 5, 7, 0.78) 18%, rgba(5, 5, 7, 0.4) 44%, rgba(5, 5, 7, 0) 62%);
  }

  .rs-hero__content {
    padding-bottom: 28px;
  }

  .rs-hero__actions {
    flex-wrap: nowrap;
    gap: 10px;
    margin-top: 20px;
  }

  .rs-hero__actions .rs-btn {
    flex: 1 1 0;
    height: 44px;
    padding: 0 12px;
  }

  .rs-hero__step-label {
    display: none;
  }

  /* Keep a finger-sized target around the thin progress track. */
  .rs-hero__step {
    padding: 15px 0;
  }

  .rs-hero__nav {
    gap: 8px;
    align-items: center;
  }

  .rs-hero__pause {
    width: 32px;
    height: 32px;
  }

  .rs-hero__bar {
    padding-bottom: 26px;
  }
}

/* The lead paragraph is left out where the hero would not fit on one screen with it. */
@media (max-height: 700px), (max-width: 640px) and (max-height: 720px) {
  .rs-hero__lead {
    display: none;
  }
}

@keyframes rs-rise {
  from {
    opacity: 0;
    transform: translate3d(0, 36px, 0);
  }
  to {
    opacity: 1;
    transform: none;
  }
}

@keyframes rs-fade {
  from {
    opacity: 0;
  }
}

@keyframes rs-ping {
  from {
    opacity: 0.9;
    transform: scale(0.6);
  }
  to {
    opacity: 0;
    transform: scale(1.6);
  }
}
</style>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

// A muted, looping clip that loads when it nears the viewport and plays only
// while visible. With reduced motion it shows the poster frame only.
//
// A clip that leaves the viewport gives up its source, which frees its decoder
// and buffered frames; it reloads from the browser cache when it comes back.
// Paused clips keep their decoders otherwise, and a page with dozens of clips
// (the example gallery, the home page) then runs Chrome out of memory.
const props = defineProps<{
  src: string
  poster?: string
  label: string
}>()

const video = ref<HTMLVideoElement>()
let observer: IntersectionObserver | undefined

function play(element: HTMLVideoElement): void {
  element.muted = true
  element.play().catch(() => {})
}

function release(element: HTMLVideoElement): void {
  if (!element.hasAttribute('src')) return
  element.pause()
  element.removeAttribute('src')
  // Without a source, load() empties the element and drops what it buffered.
  element.load()
}

onMounted(() => {
  const element = video.value
  if (!element) return
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  observer = new IntersectionObserver(
    ([entry]) => {
      if (entry.isIntersecting) {
        if (element.getAttribute('src') !== props.src) element.src = props.src
        play(element)
      } else {
        release(element)
      }
    },
    { rootMargin: '120px 0px', threshold: 0.15 },
  )
  observer.observe(element)
})

onBeforeUnmount(() => {
  observer?.disconnect()
  if (video.value) release(video.value)
})
</script>

<template>
  <video
    ref="video"
    class="rs-lazy-video"
    :poster="poster || undefined"
    :aria-label="label"
    muted
    loop
    playsinline
    disablepictureinpicture
    preload="none"
  />
</template>

<style scoped>
.rs-lazy-video {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  background: #000;
}
</style>

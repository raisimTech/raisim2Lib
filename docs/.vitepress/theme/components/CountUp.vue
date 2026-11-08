<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

// A number that counts up from zero the first time it scrolls into view.
// The server-rendered text is the final value.
const props = withDefaults(
  defineProps<{
    to: number
    decimals?: number
    duration?: number
  }>(),
  { decimals: 0, duration: 1600 },
)

const root = ref<HTMLElement>()
const text = ref(props.to.toFixed(props.decimals))
let observer: IntersectionObserver | undefined
let frame = 0

function animate(): void {
  const start = performance.now()
  const step = (now: number) => {
    const t = Math.min(1, (now - start) / props.duration)
    const eased = 1 - Math.pow(1 - t, 4)
    text.value = (props.to * eased).toFixed(props.decimals)
    if (t < 1) frame = requestAnimationFrame(step)
  }
  frame = requestAnimationFrame(step)
}

onMounted(() => {
  if (!root.value || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
  text.value = (0).toFixed(props.decimals)
  observer = new IntersectionObserver(
    ([entry]) => {
      if (!entry.isIntersecting) return
      observer?.disconnect()
      animate()
    },
    { threshold: 0.6 },
  )
  observer.observe(root.value)
})

onBeforeUnmount(() => {
  observer?.disconnect()
  cancelAnimationFrame(frame)
})
</script>

<template>
  <span ref="root" class="rs-count">{{ text }}</span>
</template>

<style scoped>
.rs-count {
  font-variant-numeric: tabular-nums;
}
</style>

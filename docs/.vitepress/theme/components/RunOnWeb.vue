<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useData } from 'vitepress'
import { webExamples } from '../webExamples'

const { page } = useData()
const name = computed(() => page.value.relativePath.match(/^sections\/examples\/[^/]+\/([a-z0-9_]+)\.md$/)?.[1])
const link = ref('')
watch(name, async value => {
  link.value = ''
  if (!value || typeof window === 'undefined') return
  const examples = await webExamples()
  if (value === name.value) link.value = examples.get(value) ?? ''
}, { immediate: true })
</script>

<template>
  <a v-if="link" class="run-on-web" :href="link" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.6v12.8a.8.8 0 0 0 1.2.7l10-6.4a.8.8 0 0 0 0-1.4l-10-6.4A.8.8 0 0 0 8 5.6z" /></svg>
    Run on web
  </a>
</template>

<style scoped>
.run-on-web {
  /* The page title after it is positioned (for its anchor link) and would cover the float. */
  position: relative;
  z-index: 1;
  float: right;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 38px;
  margin: 0 0 12px 16px;
  padding: 0 18px;
  border-radius: 999px;
  color: #0b0b10;
  background: var(--rs-grad-warm-loop);
  background-size: 300% 100%;
  box-shadow: 0 10px 30px -12px rgba(242, 39, 127, 0.6);
  font-size: 14px;
  font-weight: 650;
  text-decoration: none;
  transition: transform 0.25s var(--rs-ease), box-shadow 0.25s var(--rs-ease);
}
.run-on-web:hover { color: #0b0b10; text-decoration: none; transform: translateY(-2px); }
.run-on-web svg { width: 16px; height: 16px; fill: currentColor; }
@media (prefers-reduced-motion: reduce) { .run-on-web { transition: none; } }
</style>

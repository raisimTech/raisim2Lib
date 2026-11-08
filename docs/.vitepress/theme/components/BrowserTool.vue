<script setup lang="ts">
import { withBase } from 'vitepress'

const props = defineProps<{ tool: 'viewer' | 'editor'; title: string }>()
const source = withBase(`/web-examples/${props.tool === 'editor' ? 'editor' : 'index'}.html`)
</script>

<template>
  <div class="browser-tool">
    <a class="browser-tool-launch" :href="source" target="_self">
      Open the {{ title.toLowerCase() }}
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" /></svg>
    </a>
    <span class="browser-tool-note">Opens full screen and runs locally with WebGPU.</span>
  </div>
</template>

<style scoped>
.browser-tool { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 18px; margin: 28px 0; }
.browser-tool-launch {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  height: 46px;
  padding: 0 22px;
  border-radius: 999px;
  color: #0b0b10;
  background: var(--rs-grad-warm-loop);
  background-size: 300% 100%;
  box-shadow: 0 10px 30px -12px rgba(242, 39, 127, 0.6);
  font-size: 15px;
  font-weight: 650;
  text-decoration: none;
  transition: transform 0.25s var(--rs-ease), box-shadow 0.25s var(--rs-ease);
}
.browser-tool-launch:hover { color: #0b0b10; text-decoration: none; transform: translateY(-2px); box-shadow: 0 16px 40px -12px rgba(242, 39, 127, 0.75); }
.browser-tool-launch svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; transition: transform 0.25s var(--rs-ease); }
.browser-tool-launch:hover svg { transform: translateX(3px); }
.browser-tool-note { color: var(--vp-c-text-2); font-size: 14px; }
@media (prefers-reduced-motion: reduce) { .browser-tool-launch, .browser-tool-launch svg { transition: none; } }
</style>

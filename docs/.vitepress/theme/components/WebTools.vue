<script setup lang="ts">
import { withBase } from 'vitepress'

const tools = [
  {
    name: 'Rayrai TCP viewer',
    link: '/web-examples/index.html',
    kind: 'Explore',
    description: 'The desktop viewer in your browser. Run any RaisimServer example from the docs, or connect to a server running on this computer.',
    action: 'Open the viewer',
  },
  {
    name: 'Browser scene editor',
    link: '/web-examples/editor.html',
    kind: 'Create',
    description: 'Add objects, adjust their properties and preview the physics. Export your scene as a .rscene file for desktop RaiSim.',
    action: 'Launch scene editor',
  },
]
</script>

<template>
  <div class="web-tools" aria-label="Available web tools">
    <a v-for="tool in tools" :key="tool.link" class="web-tool-card" :href="withBase(tool.link)" target="_self">
      <span class="web-tool-kind">{{ tool.kind }}</span>
      <h2>{{ tool.name }}</h2>
      <p>{{ tool.description }}</p>
      <span class="web-tool-action">{{ tool.action }} <span aria-hidden="true">→</span></span>
    </a>
  </div>
</template>

<style scoped>
.web-tools { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; margin: 32px 0; }
.web-tool-card { display: flex; flex-direction: column; align-items: flex-start; padding: 28px; border: 1px solid var(--vp-c-divider); border-radius: 12px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); text-decoration: none; font-weight: 400; transition: border-color .2s; }
.web-tool-card:hover { border-color: var(--vp-c-brand-1); color: var(--vp-c-text-1); text-decoration: none; }
.web-tool-card:focus-visible { outline: 2px solid var(--vp-c-brand-1); outline-offset: 4px; }
.web-tool-kind { padding: 3px 10px; border-radius: 6px; color: var(--vp-c-brand-1); background: var(--vp-c-brand-soft); font-size: 12px; font-weight: 600; }
.web-tool-card h2 { margin: 20px 0 12px; padding: 0; border: 0; font-size: 21px; line-height: 1.4; letter-spacing: -.02em; }
.web-tool-card p { margin: 0 0 24px; color: var(--vp-c-text-2); font-size: 14px; line-height: 1.7; }
.web-tool-action { margin-top: auto; color: var(--vp-c-brand-1); font-size: 14px; font-weight: 600; }
@media (max-width: 600px) { .web-tools { grid-template-columns: minmax(0, 1fr); } .web-tool-card { padding: 24px; } }
@media (prefers-reduced-motion: reduce) { .web-tool-card { transition: none; } }
</style>

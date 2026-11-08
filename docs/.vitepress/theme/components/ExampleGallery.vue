<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { withBase } from 'vitepress'
import LazyVideo from './LazyVideo.vue'
import { data as gallery } from '../data/examples.data'
import { exampleCategoryIcons } from '../exampleCategoryIcons'
import { webExamples } from '../webExamples'

// The example gallery: <ExampleGallery />. Tabs select one category of
// sections/examples/gallery.json at a time, so a page never holds more than a
// category's few clips. A tile links to its example page and plays the clip
// rsc/docs/video/examples/<name>.mp4; examples without a window (gallery.json's
// "console" map) get a tile with their command instead. Server examples the
// deployed browser viewer lists also get a Run on web link. The chosen category
// is kept for the browser tab, so going back from an example page returns to it.
const STORAGE_KEY = 'raisim.exampleCategory'
const creditsLink = withBase('/sections/AssetCredits.html')

function byName(files: Record<string, string>): Record<string, string> {
  const named: Record<string, string> = {}
  for (const [path, url] of Object.entries(files)) {
    named[path.slice(path.lastIndexOf('/') + 1).replace(/\.\w+$/, '')] = url
  }
  return named
}

const clips = byName(
  import.meta.glob('../../../../rsc/docs/video/examples/*.mp4', { eager: true, query: '?url', import: 'default' }),
)
const posters = byName(
  import.meta.glob('../../../../rsc/docs/video/examples/*.jpg', { eager: true, query: '?url', import: 'default' }),
)

const KINDS: Record<string, string> = {
  benchmark: 'Benchmark',
  rayrai: 'rayrai',
  server: 'Server',
  worlds: 'Server',
  xml: 'XML world',
}

// "Server Example: Primitive Grid" -> "Primitive Grid"
function shortTitle(title: string): string {
  return title.replace(/^[\w\s]*?(Example|Workflow):\s*/i, '').replace(/^Rayrai\s+/i, '')
}

function tile(path: string) {
  const title = gallery.titles[path]
  if (!title) throw new Error(`ExampleGallery: sections/examples/${path}.md does not exist`)
  const name = path.slice(path.lastIndexOf('/') + 1)
  const isConsole = path in gallery.console
  if (!isConsole && !(clips[name] && posters[name])) {
    throw new Error(`ExampleGallery: rsc/docs/video/examples/${name}.mp4 or .jpg is missing`)
  }
  return {
    path,
    name,
    title: shortTitle(title),
    fullTitle: title,
    kind: KINDS[path.slice(0, path.indexOf('/'))] ?? 'Example',
    clip: isConsole ? undefined : clips[name],
    poster: isConsole ? undefined : posters[name],
    command: isConsole ? gallery.console[path] : undefined,
    link: withBase(`/sections/examples/${path}.html`),
  }
}

const categories = gallery.categories.map((category) => {
  const icon = exampleCategoryIcons[category.id]
  if (!icon) throw new Error(`ExampleGallery: exampleCategoryIcons.ts has no icon for "${category.id}"`)
  return { ...category, icon, tiles: category.examples.map(tile) }
})
if (!categories.length) throw new Error('ExampleGallery: gallery.json has no categories')

const selected = ref(categories[0].id)
const runnable = ref(new Map<string, string>())
const current = computed(() => categories.find((category) => category.id === selected.value) ?? categories[0])

function select(id: string): void {
  selected.value = id
  try {
    sessionStorage.setItem(STORAGE_KEY, id)
  } catch {
    // Storage can be unavailable (private windows, blocked site data).
  }
}

onMounted(() => {
  try {
    const stored = sessionStorage.getItem(STORAGE_KEY)
    if (categories.some((category) => category.id === stored)) selected.value = stored!
  } catch {
    // The first category stays selected.
  }
  webExamples().then((links) => {
    runnable.value = links
  })
})
</script>

<template>
  <div class="rs-gallery">
    <div class="rs-ex-tabs" role="group" aria-label="Example categories">
      <button
        v-for="category in categories"
        :key="category.id"
        type="button"
        class="rs-ex-tab"
        :aria-pressed="category.id === current.id"
        @click="select(category.id)"
      >
        <span class="rs-ex-tab-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path v-for="d in category.icon" :key="d" :d="d" /></svg>
        </span>
        <span class="rs-ex-tab-label">{{ category.title }}</span>
      </button>
    </div>
    <p class="rs-ex-description">{{ current.description }}</p>
    <div :key="current.id" class="rs-ex">
      <div v-for="(item, index) in current.tiles" :key="item.path" class="rs-ex-cell" :style="{ '--i': index }">
        <a class="rs-ex-tile" :href="item.link" :title="item.fullTitle">
          <div class="rs-ex-media">
            <LazyVideo v-if="item.clip" :src="item.clip" :poster="item.poster" :label="item.fullTitle" />
            <div v-else class="rs-ex-console" aria-hidden="true">
              <span class="rs-ex-prompt">$</span>
              <span class="rs-ex-command">{{ item.command }}<span class="rs-ex-cursor" /></span>
            </div>
          </div>
          <div class="rs-ex-body">
            <span class="rs-ex-kind">{{ item.kind }}</span>
            <span class="rs-ex-title">{{ item.title }}</span>
          </div>
        </a>
        <a
          v-if="runnable.has(item.name)"
          class="rs-ex-run"
          :href="runnable.get(item.name)"
          target="_blank"
          rel="noopener"
          :aria-label="`Run ${item.fullTitle} on the web`"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.6v12.8a.8.8 0 0 0 1.2.7l10-6.4a.8.8 0 0 0 0-1.4l-10-6.4A.8.8 0 0 0 8 5.6z" /></svg>
          Run on web
        </a>
      </div>
    </div>
    <p class="rs-ex-credits">
      Some clips show third-party models licensed under CC BY 4.0; they are credited in
      <a :href="creditsLink">Asset Credits</a>.
    </p>
  </div>
</template>

<style scoped>
.rs-gallery {
  margin: 20px 0 32px;
}

.rs-ex-tabs {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
}

.rs-ex-tab {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  padding: 7px 4px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 11px;
  font-size: 16px;
  font-weight: 600;
  line-height: 22px;
  text-align: left;
  color: var(--vp-c-text-2);
  background: var(--vp-c-bg-soft);
  transition: color 0.25s ease, border-color 0.25s ease, background-color 0.25s ease;
}

.rs-ex-tab:hover {
  border-color: var(--vp-c-brand-2);
  color: var(--vp-c-text-1);
}

.rs-ex-tab:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.rs-ex-tab[aria-pressed='true'] {
  border-color: var(--vp-c-brand-2);
  color: var(--vp-c-brand-1);
  background: var(--vp-c-brand-soft);
}

.rs-ex-tab-icon {
  display: grid;
  flex: none;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: 7px;
  color: var(--vp-c-text-2);
  background: var(--vp-c-default-soft);
  transition: color 0.25s ease, background-color 0.25s ease;
}

.rs-ex-tab-icon svg {
  width: 22px;
  height: 22px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.rs-ex-tab:hover .rs-ex-tab-icon {
  color: var(--vp-c-text-1);
}

/* The chosen category's icon sits on the warm part of the logo gradient, the part
   dark strokes stay legible on. */
.rs-ex-tab[aria-pressed='true'] .rs-ex-tab-icon {
  color: #0b0b10;
  background: var(--rs-grad-warm);
  animation: rs-ex-pop 0.4s cubic-bezier(0.2, 0.7, 0.1, 1);
}

.rs-ex-tab-label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.vp-doc p.rs-ex-description {
  margin: 14px 0 0;
  font-size: 14px;
  line-height: 22px;
  color: var(--vp-c-text-2);
}

.vp-doc p.rs-ex-credits {
  margin: 14px 0 0;
  font-size: 13px;
  line-height: 20px;
  color: var(--vp-c-text-3);
}

.rs-ex {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(196px, 1fr));
  gap: 14px;
  margin-top: 16px;
}

.rs-ex-cell {
  position: relative;
  display: grid;
  transition: transform 0.3s ease;
  /* Tiles of a newly chosen category rise in one after another. */
  animation: rs-ex-in 0.5s cubic-bezier(0.2, 0.7, 0.1, 1) backwards;
  animation-delay: calc(var(--i) * 40ms);
}

.rs-ex-cell:hover {
  transform: translateY(-3px);
}

.rs-ex a.rs-ex-tile {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-radius: 12px;
  color: inherit;
  text-decoration: none;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.rs-ex-cell:hover a.rs-ex-tile {
  border-color: var(--vp-c-brand-2);
  box-shadow: 0 12px 28px -14px rgba(0, 0, 0, 0.45);
}

.rs-ex a.rs-ex-tile:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.rs-ex-media {
  position: relative;
  aspect-ratio: 16 / 10;
  overflow: hidden;
  background: #0b0b10;
}

.rs-ex-media :deep(video) {
  position: absolute;
  inset: 0;
  transition: transform 0.6s ease;
}

.rs-ex-cell:hover .rs-ex-media :deep(video) {
  transform: scale(1.05);
}

.rs-ex a.rs-ex-run {
  position: absolute;
  top: 8px;
  right: 8px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 26px;
  padding: 0 10px 0 8px;
  border-radius: 999px;
  color: #0b0b10;
  background: var(--rs-grad-warm-loop);
  background-size: 300% 100%;
  box-shadow: 0 6px 18px -8px rgba(242, 39, 127, 0.7);
  font-size: 12px;
  font-weight: 650;
  line-height: 1;
  white-space: nowrap;
  text-decoration: none;
  transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.rs-ex a.rs-ex-run:hover {
  color: #0b0b10;
  transform: translateY(-1px);
  box-shadow: 0 8px 22px -8px rgba(242, 39, 127, 0.85);
}

.rs-ex a.rs-ex-run:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}

.rs-ex-run svg {
  width: 12px;
  height: 12px;
  fill: currentColor;
}

.rs-ex-console {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 16px;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  color: #d8dbe5;
  background: radial-gradient(120% 120% at 0% 0%, rgba(92, 112, 211, 0.3), transparent 60%), #0b0b10;
}

.rs-ex-prompt {
  align-self: flex-start;
  color: #ccf722;
}

.rs-ex-command {
  line-height: 1.5;
  word-break: break-all;
}

.rs-ex-cursor {
  display: inline-block;
  width: 8px;
  height: 14px;
  margin-left: 4px;
  vertical-align: -2px;
  background: #f2277f;
  animation: rs-ex-blink 1.1s steps(1) infinite;
}

.rs-ex-body {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px 12px;
}

.rs-ex-kind {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--vp-c-text-3);
}

.rs-ex-title {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.35;
  color: var(--vp-c-text-1);
}

@keyframes rs-ex-pop {
  40% {
    transform: scale(1.14);
  }
}

@keyframes rs-ex-in {
  from {
    opacity: 0;
    transform: translateY(14px) scale(0.98);
  }
}

@keyframes rs-ex-blink {
  50% {
    opacity: 0;
  }
}

@media (max-width: 1023px) {
  .rs-ex-tab {
    gap: 4px;
    font-size: 14px;
  }
}

@media (max-width: 640px) {
  .rs-ex-tabs {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  /* An odd last tab spans the row. */
  .rs-ex-tab:last-child:nth-child(odd) {
    grid-column: 1 / -1;
  }
}

@media (max-width: 480px) {
  .rs-ex {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }

  .rs-ex-title {
    font-size: 13px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .rs-ex-tab,
  .rs-ex-tab-icon,
  .rs-ex-cell,
  .rs-ex a.rs-ex-tile,
  .rs-ex a.rs-ex-run,
  .rs-ex-media :deep(video) {
    transition: none;
  }

  .rs-ex-tab[aria-pressed='true'] .rs-ex-tab-icon,
  .rs-ex-cell,
  .rs-ex-cursor {
    animation: none;
  }
}
</style>

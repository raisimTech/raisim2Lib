<script lang="ts">
import { ref } from 'vue'

export type Platform = 'linux' | 'macos' | 'windows'

const STORAGE_KEY = 'raisim.platform'
const PLATFORMS: { id: Platform; label: string }[] = [
  { id: 'linux', label: 'Linux' },
  { id: 'macos', label: 'macOS' },
  { id: 'windows', label: 'Windows' },
]

// One choice for every tab group on every page, so switching to Windows in one
// place switches them all.
const chosen = ref<Platform>('linux')
let initialized = false

function detect(): Platform {
  const nav = navigator as Navigator & { userAgentData?: { platform?: string } }
  const platform = `${nav.userAgentData?.platform ?? ''} ${navigator.userAgent}`.toLowerCase()
  if (platform.includes('win')) return 'windows'
  if (platform.includes('mac') && !/iphone|ipad/.test(platform)) return 'macos'
  return 'linux'
}

function initialize(): void {
  if (initialized) return
  initialized = true
  let stored: string | null = null
  try {
    stored = localStorage.getItem(STORAGE_KEY)
  } catch {
    // Storage can be unavailable (private windows, blocked site data).
  }
  chosen.value = PLATFORMS.some((p) => p.id === stored) ? (stored as Platform) : detect()
}

function choose(platform: Platform): void {
  chosen.value = platform
  try {
    localStorage.setItem(STORAGE_KEY, platform)
  } catch {
    // The choice still holds until the page is reloaded.
  }
}
</script>

<script setup lang="ts">
import { computed, onMounted, useId, useSlots } from 'vue'

// Platform-specific instructions as tabs. In Markdown, give each platform a
// named slot, with blank lines around its content:
//   <PlatformTabs>
//   <template #linux>
//
//   ...Markdown...
//
//   </template>
//   <template #macos> ... <template #windows> ...
//   </PlatformTabs>
// The reader's platform is selected first. Every panel is rendered, so search
// indexes all of them; only the chosen one is shown.
const slots = useSlots()
const id = useId()
const tabs = computed(() => PLATFORMS.filter((p) => slots[p.id]))
const active = computed(() => (tabs.value.some((t) => t.id === chosen.value) ? chosen.value : tabs.value[0]?.id))

onMounted(initialize)

function onKeydown(event: KeyboardEvent, index: number): void {
  const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
  const target = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.value.length - 1 : index + step
  if (step === 0 && event.key !== 'Home' && event.key !== 'End') return
  event.preventDefault()
  const next = (target + tabs.value.length) % tabs.value.length
  choose(tabs.value[next].id)
  const list = (event.currentTarget as HTMLElement).parentElement
  ;(list?.children[next] as HTMLElement | undefined)?.focus()
}
</script>

<template>
  <div class="rs-platform-tabs">
    <div class="rs-platform-tabs__list" role="tablist" aria-label="Operating system">
      <button
        v-for="(tab, index) in tabs"
        :id="`${id}-${tab.id}-tab`"
        :key="tab.id"
        type="button"
        role="tab"
        class="rs-platform-tabs__tab"
        :aria-selected="tab.id === active"
        :aria-controls="`${id}-${tab.id}`"
        :tabindex="tab.id === active ? 0 : -1"
        @click="choose(tab.id)"
        @keydown="onKeydown($event, index)"
      >
        {{ tab.label }}
      </button>
    </div>
    <div
      v-for="tab in tabs"
      v-show="tab.id === active"
      :id="`${id}-${tab.id}`"
      :key="tab.id"
      role="tabpanel"
      class="rs-platform-tabs__panel"
      :aria-labelledby="`${id}-${tab.id}-tab`"
    >
      <slot :name="tab.id" />
    </div>
  </div>
</template>

<style scoped>
.rs-platform-tabs {
  margin: 16px 0 24px;
}

.rs-platform-tabs__list {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--vp-c-divider);
}

.rs-platform-tabs__tab {
  position: relative;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-2);
  transition: color 0.25s ease;
}

.rs-platform-tabs__tab::after {
  content: '';
  position: absolute;
  right: 8px;
  bottom: -1px;
  left: 8px;
  height: 2px;
  border-radius: 2px;
  background: transparent;
  transition: background-color 0.25s ease;
}

.rs-platform-tabs__tab:hover {
  color: var(--vp-c-text-1);
}

.rs-platform-tabs__tab[aria-selected='true'] {
  color: var(--vp-c-brand-1);
}

.rs-platform-tabs__tab[aria-selected='true']::after {
  background: var(--vp-c-brand-1);
}

.rs-platform-tabs__tab:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: -2px;
  border-radius: 6px;
}

.rs-platform-tabs__panel > :deep(:first-child) {
  margin-top: 16px;
}
</style>

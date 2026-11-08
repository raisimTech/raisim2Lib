<script setup lang="ts">
import { ref } from 'vue'
import { withBase } from 'vitepress'

// The highlighted samples come from index.md through the `cpp` and `python`
// slots, so Shiki colors them at build time like any other code block.
const tab = ref<'cpp' | 'python'>('cpp')

const steps = [
  { title: 'Clone raisim2Lib', text: 'The first CMake configure downloads the matching RaiSim and rayrai binaries.' },
  { title: 'Build the examples', text: 'One CMake build on Linux, macOS or Windows.' },
  { title: 'Run and watch', text: 'Start the TCP viewer and run any server example next to it.' },
]
</script>

<template>
  <section id="code" v-in-view class="rs-section rs-code">
    <div class="rs-container rs-code__grid">
      <div>
        <p v-reveal class="rs-eyebrow">Get going</p>
        <h2 v-reveal="60" class="rs-h2">A running world in a few lines.</h2>
        <ol class="rs-code__steps">
          <li v-for="(step, index) in steps" :key="step.title" v-reveal="120 + index * 80">
            <span class="rs-code__num">{{ index + 1 }}</span>
            <span>
              <strong>{{ step.title }}</strong>
              <span>{{ step.text }}</span>
            </span>
          </li>
        </ol>
        <div v-reveal="360" class="rs-code__actions">
          <a class="rs-btn rs-btn--primary" :href="withBase('/sections/QuickStart.html')">
            Quick start
            <svg class="rs-arrow" viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11M11 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </a>
          <a class="rs-btn rs-btn--ghost" :href="withBase('/sections/Examples.html')">Browse examples</a>
        </div>
      </div>

      <div v-reveal="120" class="rs-code__window">
        <div class="rs-code__bar">
          <span class="rs-code__dots" aria-hidden="true"><i /><i /><i /></span>
          <div class="rs-code__tabs" role="tablist" aria-label="Language">
            <button
              type="button"
              role="tab"
              :aria-selected="tab === 'cpp'"
              :class="{ 'is-active': tab === 'cpp' }"
              @click="tab = 'cpp'"
            >
              main.cpp
            </button>
            <button
              type="button"
              role="tab"
              :aria-selected="tab === 'python'"
              :class="{ 'is-active': tab === 'python' }"
              @click="tab = 'python'"
            >
              main.py
            </button>
          </div>
        </div>
        <div class="rs-code__body">
          <div v-show="tab === 'cpp'" role="tabpanel"><slot name="cpp" /></div>
          <div v-show="tab === 'python'" role="tabpanel"><slot name="python" /></div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-code__grid {
  display: grid;
  grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
  gap: 64px;
  align-items: center;
}

.rs-code__steps {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin: 40px 0 0;
  padding: 0;
  list-style: none;
}

.rs-code__steps li {
  display: flex;
  gap: 16px;
  align-items: flex-start;
}

.rs-code__steps li > span:last-child {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 15px;
  line-height: 1.55;
  color: var(--rs-text-2);
}

.rs-code__steps strong {
  font-size: 17px;
  font-weight: 700;
  color: var(--rs-text);
}

.rs-code__num {
  display: grid;
  flex: none;
  place-items: center;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  font-size: 14px;
  font-weight: 750;
  color: #0b0b10;
  background: var(--rs-grad);
}

.rs-code__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-top: 40px;
}

.rs-code__window {
  overflow: hidden;
  border-radius: 18px;
  background: #0c0c12;
  border: 1px solid var(--rs-line-2);
  box-shadow: 0 40px 120px -40px rgba(92, 112, 211, 0.45), 0 0 0 1px rgba(0, 0, 0, 0.6);
}

.rs-code__bar {
  display: flex;
  align-items: center;
  gap: 18px;
  padding: 0 16px;
  height: 48px;
  background: #13131b;
  border-bottom: 1px solid var(--rs-line);
}

.rs-code__dots {
  display: flex;
  gap: 7px;
}

.rs-code__dots i {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: #2a2a35;
}

.rs-code__dots i:first-child {
  background: #f2277f;
}

.rs-code__dots i:nth-child(2) {
  background: #e7774b;
}

.rs-code__dots i:nth-child(3) {
  background: #ccf722;
}

.rs-code__tabs {
  display: flex;
  gap: 4px;
}

.rs-code__tabs button {
  padding: 6px 12px;
  border-radius: 8px;
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  color: var(--rs-text-3);
  background: none;
  border: 0;
  cursor: pointer;
  transition: color 0.2s ease, background-color 0.2s ease;
}

.rs-code__tabs button:hover {
  color: var(--rs-text);
}

.rs-code__tabs button.is-active {
  color: var(--rs-text);
  background: rgba(255, 255, 255, 0.08);
}

.rs-code__body {
  min-height: 420px;
}

/* The slotted blocks are VitePress code blocks; strip their page styling. */
.rs-code__body :deep(div[class*='language-']) {
  margin: 0;
  border-radius: 0;
  background: transparent;
}

.rs-code__body :deep(pre) {
  margin: 0;
  padding: 22px 24px;
  overflow-x: auto;
  font-family: var(--vp-font-family-mono);
  font-size: 13.5px;
  line-height: 1.75;
}

.rs-code__body :deep(.shiki span) {
  color: var(--shiki-dark, inherit);
}

.rs-code__body :deep(.lang),
.rs-code__body :deep(button.copy) {
  display: none;
}

@media (max-width: 1023px) {
  .rs-code__grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 48px;
  }

  .rs-code__body {
    min-height: 0;
  }
}

@media (max-width: 560px) {
  .rs-code__body :deep(pre) {
    padding: 18px 16px;
    font-size: 12px;
  }
}
</style>

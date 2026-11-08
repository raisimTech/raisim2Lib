<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<template>
  <section v-in-view class="rs-section rs-cta">
    <div class="rs-container">
      <div v-reveal class="rs-cta__panel">
        <div class="rs-cta__aurora" aria-hidden="true"><i /><i /><i /></div>
        <div class="rs-cta__content">
          <h2 class="rs-h2">Start training today.</h2>
          <p class="rs-lead">
            Academic and trial licenses are free. Commercial licenses are per person, per year; three or more
            include a master key that works on any machine.
          </p>
          <div class="rs-cta__actions">
            <a class="rs-btn rs-btn--light" :href="withBase('/sections/QuickStart.html')">
              Get started
              <svg class="rs-arrow" viewBox="0 0 20 20" aria-hidden="true"><path d="M4 10h11M11 5l5 5-5 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>
            </a>
            <a class="rs-btn rs-btn--ghost" :href="withBase('/sections/License.html')">Request a license</a>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-cta__panel {
  position: relative;
  overflow: hidden;
  isolation: isolate;
  padding: clamp(56px, 8vw, 104px) clamp(24px, 6vw, 88px);
  border-radius: 28px;
  background: #0b0b12;
  border: 1px solid var(--rs-line-2);
}

.rs-cta__aurora {
  position: absolute;
  inset: -40%;
  z-index: -1;
  filter: blur(70px) saturate(130%);
  opacity: 0.75;
}

/* Longhands, so the play state below is not reset by an animation shorthand. The
   blurred blobs drift only while the section is on screen (v-in-view). */
.rs-cta__aurora i {
  position: absolute;
  width: 46%;
  height: 46%;
  border-radius: 50%;
  mix-blend-mode: screen;
  animation-timing-function: ease-in-out;
  animation-iteration-count: infinite;
  animation-direction: alternate;
  animation-play-state: paused;
}

.rs-cta.is-in-view .rs-cta__aurora i {
  animation-play-state: running;
}

.rs-cta__aurora i:nth-child(1) {
  left: 18%;
  top: 22%;
  background: #f2277f;
  animation-name: rs-drift-a;
  animation-duration: 16s;
}

.rs-cta__aurora i:nth-child(2) {
  left: 44%;
  top: 30%;
  background: #5c70d3;
  animation-name: rs-drift-b;
  animation-duration: 19s;
}

.rs-cta__aurora i:nth-child(3) {
  left: 30%;
  top: 44%;
  width: 34%;
  height: 34%;
  background: #ccf722;
  opacity: 0.55;
  animation-name: rs-drift-c;
  animation-duration: 22s;
}

.rs-cta__content {
  position: relative;
  max-width: 720px;
}

.rs-cta__content .rs-lead {
  color: rgba(255, 255, 255, 0.82);
}

.rs-cta__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 36px;
}

.rs-btn--light {
  color: #0b0b10 !important;
  background: #fff;
  box-shadow: 0 12px 34px -12px rgba(0, 0, 0, 0.6);
}

.rs-btn--light:hover {
  transform: translateY(-2px);
}

@keyframes rs-drift-a {
  to {
    transform: translate(28%, 18%) scale(1.15);
  }
}

@keyframes rs-drift-b {
  to {
    transform: translate(-30%, -12%) scale(0.9);
  }
}

@keyframes rs-drift-c {
  to {
    transform: translate(40%, -26%) scale(1.25);
  }
}

@media (max-width: 560px) {
  .rs-cta__panel {
    border-radius: 22px;
  }

  .rs-cta__actions .rs-btn {
    flex: 1 1 100%;
  }
}
</style>

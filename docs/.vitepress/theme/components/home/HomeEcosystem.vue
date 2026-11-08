<script setup lang="ts">
import { withBase } from 'vitepress'
import LazyVideo from '../LazyVideo.vue'
import { clips, images } from '../../media'
</script>

<template>
  <section id="ecosystem" v-in-view class="rs-section rs-eco">
    <div class="rs-container">
      <p v-reveal class="rs-eyebrow">Robot learning</p>
      <h2 v-reveal="60" class="rs-h2">From simulation to a trained policy.</h2>
      <p v-reveal="120" class="rs-lead">
        <strong class="rs-lead-claim rs-grad-text">A quadruped learns to walk in 10 seconds.</strong> Build environments in C++ or Python, train with PPO in PyTorch,
        render camera observations headless, and watch any run live, all on the same RaiSim world.
      </p>

      <div class="rs-eco__grid">
        <a v-reveal v-glow class="rs-card rs-eco__card rs-eco__card--rayrai" :href="withBase('/sections/Rayrai.html')">
          <div class="rs-eco__media">
            <LazyVideo :src="clips.offroadDrive.src" :poster="clips.offroadDrive.poster" label="Off-road buggy rendered with rayrai" />
          </div>
          <div class="rs-eco__shade" />
          <div class="rs-eco__body">
            <span class="rs-eco__kicker">Observations</span>
            <h3>rayrai</h3>
            <p>Render camera and LiDAR observations headless for training, or embed photoreal views in your own tools.</p>
          </div>
        </a>

        <a v-reveal="90" v-glow class="rs-card rs-eco__card rs-eco__card--viewer" :href="withBase('/sections/RayraiTcpViewer.html')">
          <div class="rs-eco__media">
            <img class="rs-cover" :src="images.tcpViewer" alt="The rayrai TCP viewer showing a forest scene" loading="lazy" />
          </div>
          <div class="rs-eco__shade" />
          <div class="rs-eco__body">
            <span class="rs-eco__kicker">Debugging</span>
            <h3>TCP viewer</h3>
            <p>Watch a running simulation live, inspect contacts, and pause, step or drag the robot.</p>
          </div>
        </a>

        <a v-reveal v-glow class="rs-card rs-eco__card rs-eco__card--gym" :href="withBase('/sections/RaisimGymTorch.html')">
          <div class="rs-eco__media">
            <LazyVideo :src="clips.gymPolicy.src" :poster="clips.gymPolicy.poster" label="A quadruped walking over rough steps with a policy trained in raisimGymTorch" />
          </div>
          <div class="rs-eco__shade" />
          <div class="rs-eco__body">
            <span class="rs-eco__kicker">Reinforcement learning</span>
            <h3>raisimGymTorch</h3>
            <p>Vectorized environments and PPO in PyTorch: quadrupedal locomotion in 10 seconds, tens of billions of steps on one desktop.</p>
          </div>
        </a>

        <a v-reveal="90" v-glow class="rs-card rs-eco__card rs-eco__card--py" :href="withBase('/sections/RaiSimPy.html')">
          <div class="rs-eco__code" aria-hidden="true">
            <span><i class="k">import</i> raisimpy <i class="k">as</i> raisim</span>
            <span>world = raisim.World()</span>
            <span>world.addGround()</span>
            <span>ball = world.addSphere(<i class="n">0.2</i>, <i class="n">1.0</i>)</span>
            <span><i class="k">for</i> _ <i class="k">in</i> range(<i class="n">10_000</i>):</span>
            <span>&nbsp;&nbsp;&nbsp;&nbsp;world.integrate()</span>
          </div>
          <div class="rs-eco__body rs-eco__body--static">
            <span class="rs-eco__kicker">Python</span>
            <h3>RaisimPy</h3>
            <p>The full C++ API from Python with NumPy arrays, for prototyping scenes and controllers.</p>
          </div>
        </a>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-eco__grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  grid-auto-rows: 300px;
  gap: 18px;
  margin-top: 56px;
}

.rs-eco__card {
  display: block;
  overflow: hidden;
  color: inherit !important;
  text-decoration: none !important;
  background: #000;
}

.rs-eco__card--rayrai {
  grid-column: span 7;
  grid-row: span 2;
}

.rs-eco__card--viewer {
  grid-column: span 5;
}

.rs-eco__card--gym {
  grid-column: span 5;
}

.rs-eco__card--py {
  grid-column: span 12;
  grid-row: span 1;
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
  align-items: center;
  background: radial-gradient(120% 140% at 0% 0%, rgba(92, 112, 211, 0.22), transparent 55%),
    radial-gradient(120% 140% at 100% 100%, rgba(242, 39, 127, 0.16), transparent 55%), var(--rs-card);
}

.rs-eco__media {
  position: absolute;
  inset: 0;
  transition: transform 1.2s var(--rs-ease);
}

.rs-eco__shade {
  position: absolute;
  inset: 0;
  background: linear-gradient(0deg, rgba(5, 5, 7, 0.94) 0%, rgba(5, 5, 7, 0.5) 38%, rgba(5, 5, 7, 0) 70%);
}

.rs-eco__body {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 26px 28px;
}

.rs-eco__body--static {
  position: relative;
  padding: 32px 40px 32px 8px;
}

.rs-eco__kicker {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.6);
}

.rs-eco__body h3 {
  margin: 6px 0 0;
  font-size: clamp(24px, 2.4vw, 32px);
  font-weight: 800;
  letter-spacing: -0.03em;
  color: #fff;
}

.rs-eco__body p {
  margin: 8px 0 0;
  max-width: 440px;
  font-size: 15px;
  line-height: 1.6;
  color: rgba(255, 255, 255, 0.76);
}

.rs-eco__code {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 32px 0 32px 36px;
  padding: 22px 26px;
  border-radius: 14px;
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  line-height: 1.7;
  white-space: pre;
  overflow: hidden;
  color: #e6e6ee;
  background: rgba(0, 0, 0, 0.45);
  border: 1px solid var(--rs-line-2);
}

.rs-eco__code i {
  font-style: normal;
}

.rs-eco__code .k {
  color: #f784b6;
}

.rs-eco__code .n {
  color: #b9c3ff;
}

@media (hover: hover) {
  .rs-eco__card:hover .rs-eco__media {
    transform: scale(1.05);
  }
}

@media (max-width: 1023px) {
  /* rayrai on a full-width row, the viewer and gym cards side by side, then Python. */
  .rs-eco__grid {
    grid-template-rows: 360px 260px minmax(260px, auto);
  }

  .rs-eco__card--rayrai,
  .rs-eco__card--py {
    grid-column: span 12;
  }

  .rs-eco__card--rayrai {
    grid-row: span 1;
  }

  .rs-eco__card--viewer,
  .rs-eco__card--gym {
    grid-column: span 6;
  }
}

@media (max-width: 680px) {
  .rs-eco__grid {
    display: flex;
    gap: 14px;
    margin: 40px -16px 0;
    padding: 0 16px 8px;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    scroll-padding: 0 16px;
    scrollbar-width: none;
  }

  .rs-eco__grid::-webkit-scrollbar {
    display: none;
  }

  .rs-eco__card,
  .rs-eco__card--rayrai {
    flex: 0 0 84%;
    height: 400px;
    min-height: 0;
    scroll-snap-align: start;
  }

  .rs-eco__card--py {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .rs-eco__code {
    margin: 18px 18px 0;
    padding: 16px;
    font-size: 12px;
  }

  .rs-eco__body {
    padding: 20px;
  }

  .rs-eco__body--static {
    padding: 20px;
  }
}
</style>

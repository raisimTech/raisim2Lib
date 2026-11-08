<script setup lang="ts">
import { withBase } from 'vitepress'

// The build year, so the server-rendered page and the hydrated app agree.
const year = __BUILD_YEAR__

const columns = [
  {
    title: 'Get started',
    links: [
      ['Quick start', '/sections/QuickStart.html'],
      ['Installation', '/sections/Installation.html'],
      ['Examples', '/sections/Examples.html'],
      ['Changelog', '/sections/Changelog.html'],
    ],
  },
  {
    title: 'Physics',
    links: [
      ['World', '/sections/WorldSystem.html'],
      ['Articulated systems', '/sections/ArticulatedSystem.html'],
      ['Contact', '/sections/Contact.html'],
      ['Sensors', '/sections/Sensors.html'],
    ],
  },
  {
    title: 'Tools',
    links: [
      ['rayrai renderer', '/sections/Rayrai.html'],
      ['TCP viewer', '/sections/RayraiTcpViewer.html'],
      ['raisimGymTorch', '/sections/RaisimGymTorch.html'],
      ['API reference', '/api/'],
    ],
  },
  {
    title: 'Company',
    links: [
      ['License', '/sections/License.html'],
      ['Support', '/sections/Support.html'],
      ['Benchmark', '/sections/Benchmark.html'],
      ['Asset credits', '/sections/AssetCredits.html'],
      ['GitHub', 'https://github.com/raisimTech/raisim2Lib'],
    ],
  },
]

function href(link: string): string {
  return link.startsWith('http') ? link : withBase(link)
}
</script>

<template>
  <footer class="rs-footer">
    <div class="rs-container">
      <div class="rs-footer__top">
        <div class="rs-footer__brand">
          <img :src="withBase('/logo-mark.svg')" alt="" width="44" height="30" />
          <span>RaiSim</span>
          <p>Fast physics and photoreal rendering for robotics and AI.</p>
        </div>
        <nav class="rs-footer__cols" aria-label="Footer">
          <div v-for="column in columns" :key="column.title">
            <h4>{{ column.title }}</h4>
            <ul>
              <li v-for="[text, link] in column.links" :key="text">
                <a :href="href(link)" :target="link.startsWith('http') ? '_blank' : undefined" :rel="link.startsWith('http') ? 'noreferrer' : undefined">{{ text }}</a>
              </li>
            </ul>
          </div>
        </nav>
      </div>
      <div class="rs-footer__bottom">
        <span>© {{ year }} RaiSim Tech Inc.</span>
        <span>
          Every clip on this page is a RaiSim simulation. Third-party models in them are credited in
          <a :href="withBase('/sections/AssetCredits.html')">Asset credits</a>.
        </span>
      </div>
    </div>
  </footer>
</template>

<style scoped>
.rs-footer {
  margin-top: 136px;
  padding: 64px 0 40px;
  border-top: 1px solid var(--rs-line);
  background: #030304;
}

.rs-footer__top {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2fr);
  gap: 48px;
}

.rs-footer__brand {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
}

.rs-footer__brand img {
  border-radius: 5px;
}

.rs-footer__brand span {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--rs-text);
}

.rs-footer__brand p {
  flex-basis: 100%;
  max-width: 300px;
  margin: 8px 0 0;
  font-size: 14px;
  line-height: 1.6;
  color: var(--rs-text-3);
}

.rs-footer__cols {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 32px;
}

.rs-footer h4 {
  margin: 0 0 14px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rs-text-3);
}

.rs-footer ul {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.rs-footer a {
  font-size: 15px;
  color: var(--rs-text-2) !important;
  text-decoration: none !important;
  transition: color 0.2s ease;
}

.rs-footer a:hover {
  color: var(--rs-text) !important;
}

.rs-footer__bottom a {
  text-decoration: underline !important;
  text-underline-offset: 3px;
}

.rs-footer__bottom {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 12px;
  margin-top: 56px;
  padding-top: 24px;
  border-top: 1px solid var(--rs-line);
  font-size: 13px;
  color: var(--rs-text-3);
}

@media (max-width: 900px) {
  .rs-footer__top {
    grid-template-columns: minmax(0, 1fr);
  }

  .rs-footer__cols {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 767px) {
  .rs-footer {
    margin-top: 88px;
  }
}
</style>

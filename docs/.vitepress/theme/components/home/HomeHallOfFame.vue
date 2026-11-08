<script setup lang="ts">
import { withBase } from 'vitepress'
import { papers, type Paper } from '../../data/papers'

// Papers whose results came from RaiSim: the Nature paper as the feature, every
// Science Robotics paper as a card, and the rest in a short list.
const featured = papers.filter((paper) => paper.kind === 'nature')
const scienceRobotics = papers.filter((paper) => paper.kind === 'science-robotics')
const others = papers.filter((paper) => paper.kind === 'other')

function byline(paper: Paper): string {
  const surname = (author: string) => author.slice(author.lastIndexOf(' ') + 1)
  const first = surname(paper.authors[0])
  const who = paper.authors.length > 2 ? `${first} et al.` : paper.authors.map(surname).join(' and ')
  return `${who} · ${paper.institutions}`
}
</script>

<template>
  <section id="hall-of-fame" v-in-view class="rs-section rs-hof">
    <div class="rs-container">
      <p v-reveal class="rs-eyebrow">Hall of fame</p>
      <h2 v-reveal="60" class="rs-h2">Research that ran on RaiSim.</h2>
      <p v-reveal="120" class="rs-lead">
        Controllers trained in RaiSim have run a marathon, raced across beach sand and hiked alpine trails. These are
        the papers in Nature and Science Robotics, and more, whose robots learned in RaiSim.
      </p>

      <a
        v-for="paper in featured"
        :key="paper.url"
        v-reveal="160"
        v-glow
        class="rs-card rs-hof__featured"
        :href="paper.url"
        target="_blank"
        rel="noreferrer"
      >
        <div class="rs-hof__journal">
          <span class="rs-grad-text">{{ paper.venue }}</span>
          <span class="rs-hof__year">{{ paper.year }}</span>
        </div>
        <div class="rs-hof__featured-body">
          <h3>{{ paper.title }}</h3>
          <p>{{ paper.highlight }}</p>
          <span class="rs-hof__by">{{ byline(paper) }}</span>
          <span class="rs-hof__more" aria-hidden="true">
            Read the paper
            <svg viewBox="0 0 16 16"><path d="M5 11L11 5M6 5h5v5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </span>
        </div>
      </a>

      <div class="rs-hof__grid">
        <a
          v-for="(paper, index) in scienceRobotics"
          :key="paper.url"
          v-reveal="(index % 3) * 90"
          v-glow
          class="rs-card rs-hof__card"
          :href="paper.url"
          target="_blank"
          rel="noreferrer"
        >
          <span class="rs-hof__venue">{{ paper.venue }} · {{ paper.year }}</span>
          <h3>{{ paper.title }}</h3>
          <p>{{ paper.highlight }}</p>
          <span class="rs-hof__by">{{ byline(paper) }}</span>
        </a>

        <div v-reveal="180" v-glow class="rs-card rs-hof__card rs-hof__others">
          <span class="rs-hof__venue">More papers</span>
          <ul>
            <li v-for="paper in others" :key="paper.url">
              <a :href="paper.url" target="_blank" rel="noreferrer">{{ paper.short ?? paper.title }}</a>
              <span>{{ paper.venueShort ?? paper.venue }} {{ paper.year }}</span>
            </li>
          </ul>
          <a class="rs-hof__all" :href="withBase('/sections/HallOfFame.html')">
            All papers and citations
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4l4 4-4 4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" /></svg>
          </a>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.rs-hof__featured {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 2.2fr);
  gap: 40px;
  align-items: center;
  margin-top: 56px;
  padding: 40px 44px;
  color: inherit !important;
  text-decoration: none !important;
  background:
    radial-gradient(120% 140% at 0% 0%, rgba(242, 39, 127, 0.16), transparent 55%),
    radial-gradient(120% 140% at 100% 100%, rgba(92, 112, 211, 0.18), transparent 55%),
    var(--rs-card);
  border-color: var(--rs-line-2);
}

.rs-hof__journal {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.rs-hof__journal .rs-grad-text {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: clamp(48px, 6vw, 84px);
  font-weight: 700;
  line-height: 1;
  letter-spacing: -0.02em;
}

.rs-hof__year {
  font-size: 15px;
  font-weight: 650;
  letter-spacing: 0.12em;
  color: var(--rs-text-3);
}

.rs-hof__featured-body h3 {
  margin: 0;
  font-size: clamp(22px, 2.2vw, 30px);
  font-weight: 750;
  line-height: 1.2;
  letter-spacing: -0.02em;
  color: var(--rs-text);
}

.rs-hof__featured-body p {
  margin: 12px 0 18px;
  font-size: 17px;
  line-height: 1.55;
  color: var(--rs-text-2);
}

.rs-hof__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin-top: 18px;
}

.rs-hof__card {
  display: flex;
  flex-direction: column;
  padding: 26px 26px 24px;
  color: inherit !important;
  text-decoration: none !important;
}

.rs-hof__venue {
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--rs-text-3);
}

.rs-hof__card h3 {
  margin: 12px 0 0;
  font-size: 19px;
  font-weight: 700;
  line-height: 1.3;
  letter-spacing: -0.015em;
  color: var(--rs-text);
}

.rs-hof__card p {
  flex: 1;
  margin: 10px 0 18px;
  font-size: 15px;
  line-height: 1.6;
  color: var(--rs-text-2);
}

.rs-hof__by {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: var(--rs-text-3);
}

.rs-hof__more,
.rs-hof__all {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-top: 18px;
  font-size: 14px;
  font-weight: 650;
  color: var(--rs-text-2) !important;
  text-decoration: none !important;
  transition: color 0.25s ease;
}

.rs-hof__more svg,
.rs-hof__all svg {
  width: 15px;
  height: 15px;
  transition: transform 0.25s var(--rs-ease);
}

.rs-hof__others ul {
  flex: 1;
  margin: 14px 0 0;
  padding: 0;
  list-style: none;
}

.rs-hof__others li {
  font-size: 15px;
  line-height: 1.45;
}

.rs-hof__others li + li {
  margin-top: 10px;
}

.rs-hof__others li a {
  font-weight: 650;
  line-height: 1.35;
  color: var(--rs-text) !important;
  text-decoration: none !important;
  transition: color 0.25s ease;
}

.rs-hof__others li span {
  margin-left: 8px;
  font-size: 13px;
  white-space: nowrap;
  color: var(--rs-text-3);
}

@media (hover: hover) {
  .rs-hof__featured:hover,
  a.rs-hof__card:hover {
    transform: translateY(-6px);
  }

  .rs-hof__featured:hover .rs-hof__more,
  .rs-hof__all:hover {
    color: var(--rs-text) !important;
  }

  .rs-hof__featured:hover .rs-hof__more svg {
    transform: translate(2px, -2px);
  }

  .rs-hof__all:hover svg {
    transform: translateX(4px);
  }

  .rs-hof__others li a:hover {
    color: var(--rs-pink) !important;
  }
}

@media (max-width: 1023px) {
  .rs-hof__grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 767px) {
  .rs-hof__featured {
    grid-template-columns: minmax(0, 1fr);
    gap: 18px;
    margin-top: 40px;
    padding: 26px 22px;
  }

  .rs-hof__journal {
    flex-direction: row;
    align-items: baseline;
    gap: 12px;
  }
}

@media (max-width: 600px) {
  .rs-hof__grid {
    grid-template-columns: minmax(0, 1fr);
    gap: 14px;
  }

  .rs-hof__card {
    padding: 20px 18px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .rs-hof__featured:hover,
  a.rs-hof__card:hover {
    transform: none;
  }
}
</style>

<script setup lang="ts">
import { papers } from '../data/papers'

// The full Hall of Fame list for a docs page: <PaperList />. Each paper is cited
// with all of its authors, grouped by venue, newest first.
const groups = [
  { title: 'Nature', kind: 'nature' },
  { title: 'Science Robotics', kind: 'science-robotics' },
  { title: 'Other venues', kind: 'other' },
].map((group) => ({
  ...group,
  papers: papers.filter((paper) => paper.kind === group.kind).sort((a, b) => b.year - a.year),
}))
</script>

<template>
  <div class="rs-papers">
    <section v-for="group in groups" :key="group.kind">
      <h2 :id="group.kind" tabindex="-1">
        {{ group.title }} <a class="header-anchor" :href="`#${group.kind}`" :aria-label="`Permalink to ${group.title}`">&#8203;</a>
      </h2>
      <ol class="rs-papers__list">
        <li v-for="paper in group.papers" :key="paper.url" class="rs-papers__item">
          <span class="rs-papers__year">{{ paper.year }}</span>
          <div>
            <a class="rs-papers__title" :href="paper.url" target="_blank" rel="noreferrer">{{ paper.title }}</a>
            <p class="rs-papers__cite">
              {{ paper.authors.join(', ') }}. <em>{{ paper.venue }}</em>{{ paper.citation ? ` ${paper.citation}` : '' }},
              {{ paper.year }}. {{ paper.institutions }}.
            </p>
            <p class="rs-papers__highlight">{{ paper.highlight }}</p>
          </div>
        </li>
      </ol>
    </section>
  </div>
</template>

<style scoped>
.rs-papers__list {
  margin: 16px 0 8px;
  padding: 0;
  list-style: none;
}

.vp-doc .rs-papers__item {
  display: grid;
  grid-template-columns: 56px minmax(0, 1fr);
  gap: 16px;
  margin: 0;
  padding: 18px 0;
  border-top: 1px solid var(--vp-c-divider);
}

.rs-papers__year {
  font-size: 14px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  line-height: 26px;
  color: var(--vp-c-brand-1);
}

.vp-doc a.rs-papers__title {
  font-size: 17px;
  font-weight: 650;
  line-height: 26px;
  color: var(--vp-c-text-1);
  text-decoration: none;
}

.vp-doc a.rs-papers__title:hover {
  color: var(--vp-c-brand-1);
  text-decoration: underline;
}

.vp-doc .rs-papers__cite {
  margin: 4px 0 0;
  font-size: 14px;
  line-height: 22px;
  color: var(--vp-c-text-2);
}

.vp-doc .rs-papers__highlight {
  margin: 6px 0 0;
  font-size: 14px;
  line-height: 22px;
  color: var(--vp-c-text-3);
}

@media (max-width: 480px) {
  .vp-doc .rs-papers__item {
    grid-template-columns: minmax(0, 1fr);
    gap: 2px;
  }
}
</style>

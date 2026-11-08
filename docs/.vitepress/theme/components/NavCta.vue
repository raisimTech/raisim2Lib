<script setup lang="ts">
import { withBase } from 'vitepress'
</script>

<template>
  <a class="rs-nav-cta" :href="withBase('/sections/QuickStart.html')">Get started</a>
</template>

<style scoped>
.rs-nav-cta {
  display: none;
}

@media (min-width: 1280px) {
  .rs-nav-cta {
    display: inline-flex;
    align-items: center;
    height: 34px;
    margin-left: 16px;
    padding: 0 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 650;
    color: #0b0b10;
    background: var(--rs-grad-warm);
    background-size: 160% 100%;
    background-position: 0 0;
    transition: background-position 0.5s ease, transform 0.25s ease, box-shadow 0.25s ease;
  }

  .rs-nav-cta:hover {
    background-position: 100% 0;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px -6px rgba(242, 39, 127, 0.7);
  }
}
</style>

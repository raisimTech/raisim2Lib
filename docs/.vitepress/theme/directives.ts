import type { Directive } from 'vue'

// Elements marked with v-reveal fade in when they first scroll into view. The
// hidden starting state needs the `rs-js` class that config.mts sets on <html>
// before the page renders, so the content stays visible without JavaScript.
// A number value delays the entrance by that many milliseconds (for staggering).
let revealObserver: IntersectionObserver | undefined

function getRevealObserver(): IntersectionObserver {
  revealObserver ??= new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue
        entry.target.classList.add('is-visible')
        revealObserver!.unobserve(entry.target)
      }
    },
    { rootMargin: '0px 0px -8% 0px', threshold: 0.12 },
  )
  return revealObserver
}

export const reveal: Directive<HTMLElement, number | undefined> = {
  getSSRProps(binding) {
    return {
      class: 'rs-reveal',
      style: binding.value ? { '--rs-reveal-delay': `${binding.value}ms` } : undefined,
    }
  },
  mounted(el, binding) {
    el.classList.add('rs-reveal')
    if (binding.value) el.style.setProperty('--rs-reveal-delay', `${binding.value}ms`)
    if (typeof IntersectionObserver === 'undefined') {
      el.classList.add('is-visible')
      return
    }
    getRevealObserver().observe(el)
  },
  unmounted(el) {
    revealObserver?.unobserve(el)
  },
}

// v-in-view keeps the `is-in-view` class on an element while it is on screen, so
// looping CSS animations inside it can run only while they can be seen.
let inViewObserver: IntersectionObserver | undefined

function getInViewObserver(): IntersectionObserver {
  inViewObserver ??= new IntersectionObserver((entries) => {
    for (const entry of entries) entry.target.classList.toggle('is-in-view', entry.isIntersecting)
  })
  return inViewObserver
}

export const inView: Directive<HTMLElement> = {
  mounted(el) {
    getInViewObserver().observe(el)
  },
  unmounted(el) {
    inViewObserver?.unobserve(el)
  },
}

// v-glow tracks the pointer inside a card so CSS can draw a light that follows
// it (--rs-mx / --rs-my, in pixels from the card's top-left corner).
type GlowElement = HTMLElement & { __rsGlow?: (event: PointerEvent) => void }

export const glow: Directive<GlowElement> = {
  mounted(el) {
    const onMove = (event: PointerEvent) => {
      const rect = el.getBoundingClientRect()
      el.style.setProperty('--rs-mx', `${event.clientX - rect.left}px`)
      el.style.setProperty('--rs-my', `${event.clientY - rect.top}px`)
    }
    el.__rsGlow = onMove
    el.addEventListener('pointermove', onMove, { passive: true })
  },
  unmounted(el) {
    if (el.__rsGlow) el.removeEventListener('pointermove', el.__rsGlow)
  },
}

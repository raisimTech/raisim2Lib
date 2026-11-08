import type { Router } from 'vitepress'

// Markdown pages show short clips (former GIFs) as
// <video class="doc-video" loop muted playsinline preload="metadata">. They have
// no `autoplay`, so a clip downloads only when it plays: the observer plays a
// clip while it is on screen and pauses it when it leaves. With reduced motion,
// clips never play by themselves and show their controls instead.
let observer: IntersectionObserver | undefined

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function onIntersect(entries: IntersectionObserverEntry[]): void {
  for (const entry of entries) {
    const video = entry.target as HTMLVideoElement
    if (entry.isIntersecting && !reducedMotion()) {
      video.muted = true
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }
}

function attach(): void {
  observer?.disconnect()
  const videos = document.querySelectorAll<HTMLVideoElement>('.vp-doc video.doc-video')
  if (videos.length === 0) return
  observer = new IntersectionObserver(onIntersect, { threshold: 0.25 })
  for (const video of videos) {
    if (reducedMotion()) video.controls = true
    observer.observe(video)
  }
}

export function setupDocVideos(router: Router): void {
  const previous = router.onAfterRouteChange
  router.onAfterRouteChange = async (to) => {
    await previous?.(to)
    // The new page renders on the next frame.
    requestAnimationFrame(() => requestAnimationFrame(attach))
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach, { once: true })
  } else {
    requestAnimationFrame(attach)
  }
}

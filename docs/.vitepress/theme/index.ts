import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { defineComponent, h, onMounted } from 'vue'

import WebTools from './components/WebTools.vue'
import BrowserViewer from './components/BrowserViewer.vue'
import BrowserSceneEditor from './components/BrowserSceneEditor.vue'
import ExampleGallery from './components/ExampleGallery.vue'
import HomePage from './components/home/HomePage.vue'
import NavCta from './components/NavCta.vue'
import PaperList from './components/PaperList.vue'
import PlatformTabs from './components/PlatformTabs.vue'
import RunOnWeb from './components/RunOnWeb.vue'
import { glow, inView, reveal } from './directives'
import { setupDocVideos } from './docVideos'
import './styles/vars.css'
import './styles/docs.css'

const Layout = defineComponent({
  name: 'RaisimLayout',
  setup() {
    // The app is running, so content hidden behind `rs-js` will be revealed
    // (see the inline script in config.mts).
    onMounted(() => document.documentElement.classList.add('rs-ready'))
    return () =>
      h(DefaultTheme.Layout, null, {
        'nav-bar-content-after': () => h(NavCta),
        'doc-before': () => h(RunOnWeb),
      })
  },
})

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app, router }) {
    app.component('WebTools', WebTools)
    app.component('BrowserViewer', BrowserViewer)
    app.component('BrowserSceneEditor', BrowserSceneEditor)
    app.component('HomePage', HomePage)
    app.component('ExampleGallery', ExampleGallery)
    app.component('PlatformTabs', PlatformTabs)
    app.component('PaperList', PaperList)
    app.directive('reveal', reveal)
    app.directive('in-view', inView)
    app.directive('glow', glow)
    // Markdown pages write the release version as {{ $raisimVersion }}.
    app.config.globalProperties.$raisimVersion = __RAISIM_VERSION__
    if (!import.meta.env.SSR) setupDocVideos(router)
  },
} satisfies Theme

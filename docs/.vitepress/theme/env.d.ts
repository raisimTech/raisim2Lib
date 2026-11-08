/// <reference types="vite/client" />

// Defined by `vite.define` in .vitepress/config.mts: the release version from
// raisim2Lib/CMakeLists.txt and the year the site was built.
declare const __RAISIM_VERSION__: string
declare const __BUILD_YEAR__: number

declare module 'vue' {
  interface ComponentCustomProperties {
    $raisimVersion: string
  }
}

export {}

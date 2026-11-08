import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type DefaultTheme, type HeadConfig } from 'vitepress'
import { docsSidebar } from './sidebar.mts'
import { firstContentImage, imageInfo } from './socialImage.mjs'

const docsDir = fileURLToPath(new URL('..', import.meta.url))

// The released package version comes from the root CMake project, like the
// version string the binary packages are downloaded with.
function readRaisimVersion(): string {
  const cmake = readFileSync(new URL('../../CMakeLists.txt', import.meta.url), 'utf8')
  const patterns = [
    /set\(\s*_raisim_expected_version\s+(\d+\.\d+\.\d+)/,
    /set\(\s*RAISIM_VERSION\s+(\d+\.\d+\.\d+)/,
    /project\(\s*raisim\s+VERSION\s+(\d+\.\d+\.\d+)/,
  ]
  for (const pattern of patterns) {
    const match = cmake.match(pattern)
    if (match) return match[1]
  }
  throw new Error('Could not parse the RaiSim version from raisim2Lib/CMakeLists.txt')
}

// scripts/generate_api_docs.py writes the API pages and their sidebar from the
// Doxygen XML. Without them the API sidebar is empty and links to /api/ fail
// the dead-link check, so a build never ships with silently missing API pages.
//
// Every page renders its whole sidebar into its HTML, and the full API tree has
// about 480 entries. Pages of one namespace (/api/raisim/...) therefore get only
// that namespace's group, plus the groups without children (the landing link).
function readApiSidebars(): DefaultTheme.SidebarMulti {
  const path = `${docsDir}api/sidebar.json`
  if (!existsSync(path)) return { '/api/': [] }
  const groups: DefaultTheme.SidebarItem[] = JSON.parse(readFileSync(path, 'utf8'))
  const links = (item: DefaultTheme.SidebarItem): string[] => [
    ...(item.link ? [item.link] : []),
    ...(item.items ?? []).flatMap(links),
  ]
  const landing = groups.filter((group) => !group.items?.length)
  const sidebars: DefaultTheme.SidebarMulti = { '/api/': groups }
  for (const group of groups) {
    const dirs = new Set(links(group).map((link) => link.match(/^\/api\/([^/]+)\//)?.[1]))
    const [dir] = dirs
    if (dirs.size === 1 && dir) sidebars[`/api/${dir}/`] = [...landing, group]
  }
  return sidebars
}

// VitePress's heading slugify (from @mdit-vue/shared), with en and em dashes
// (\u2013, \u2014) added to the separators. A heading such as
// "v0.4.x (2019-08-23 – 2019-09-06)" then keeps the anchor the Sphinx site
// published, v0-4-x-2019-08-23-2019-09-06.
const rControl = /[\u0000-\u001f]/g
const rSpecial = /[\s~`!@#$%^&*()\-_+=[\]{}|\\;:"'“”‘’<>,.?/\u2013\u2014]+/g
const rCombining = /[\u0300-\u036F]/g

function slugify(text: string): string {
  return text
    .normalize('NFKD')
    .replace(rCombining, '')
    .replace(rControl, '')
    .replace(rSpecial, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/^(\d)/, '_$1')
    .toLowerCase()
}

const raisimVersion = readRaisimVersion()
const buildYear = new Date().getFullYear()
const repoUrl = 'https://github.com/raisimTech/raisim2Lib'

export default defineConfig({
  lang: 'en-US',
  title: 'RaiSim',
  description:
    'RaiSim is a fast, cross-platform physics engine for robot learning, with photoreal cameras and LiDAR through rayrai.',
  cleanUrls: false,
  // Site data (including both sidebars) and the page hash map go to one cached
  // script instead of being inlined into each of the ~640 pages.
  metaChunk: true,
  lastUpdated: false,
  appearance: 'dark',
  sitemap: { hostname: 'https://raisim.com' },
  // Files starting with an underscore are partials pulled in with <!--@include: ...-->.
  srcExclude: ['**/node_modules/**', 'image_generators/**', 'code/**', '_build/**', '.venv/**', '**/_*.md'],

  head: [
    // Runs before the first paint: `rs-js` lets scroll-reveal content start hidden
    // (it stays visible without JavaScript), and `rs-home-route` gives the home page's
    // navigation bar its over-the-video style without a flash. The theme adds
    // `rs-ready` once the app has mounted. If the site's scripts fail to load or
    // run, `rs-js` is removed shortly after the page loads, so nothing stays hidden.
    [
      'script',
      {},
      '(function () {' +
        ' var root = document.documentElement;' +
        " root.classList.add('rs-js');" +
        " if (/^\\/(index\\.html)?$/.test(location.pathname)) root.classList.add('rs-home-route');" +
        " addEventListener('load', function () {" +
        ' setTimeout(function () {' +
        " if (!root.classList.contains('rs-ready')) root.classList.remove('rs-js');" +
        ' }, 1500);' +
        ' });' +
        '})();',
    ],
    ['link', { rel: 'icon', type: 'image/png', href: '/favicon.png' }],
    ['meta', { name: 'theme-color', content: '#07070b' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:site_name', content: 'RaiSim' }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
  ],

  // The card a shared link shows on social networks and messengers: the page's title, description,
  // address and picture. The picture is the page's frontmatter `image`, else the first picture of its
  // text, else the site's. Its size lets Facebook show it on the first share of a link.
  transformHead({ pageData, title, description, content, siteConfig }) {
    const path = pageData.relativePath.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '.html')
    const url = `https://raisim.com/${path}`
    const picture = pageData.frontmatter.image
      ? undefined
      : firstContentImage(content, siteConfig.outDir, siteConfig.site.base) ??
        (() => {
          const info = imageInfo(`${siteConfig.srcDir}/public/og-image.jpg`)
          return info && { path: '/og-image.jpg', ...info, alt: 'RaiSim' }
        })()
    const image = new URL(String(pageData.frontmatter.image ?? picture?.path ?? '/og-image.jpg'), url).href
    const head: HeadConfig[] = [
      ['meta', { property: 'og:title', content: title }],
      ['meta', { property: 'og:description', content: description }],
      ['meta', { property: 'og:url', content: url }],
      ['meta', { property: 'og:image', content: image }],
      ['meta', { name: 'twitter:title', content: title }],
      ['meta', { name: 'twitter:description', content: description }],
      ['meta', { name: 'twitter:image', content: image }],
    ]
    if (picture) {
      head.push(
        ['meta', { property: 'og:image:type', content: picture.type }],
        ['meta', { property: 'og:image:width', content: String(picture.width) }],
        ['meta', { property: 'og:image:height', content: String(picture.height) }],
      )
      if (picture.alt) {
        head.push(['meta', { property: 'og:image:alt', content: picture.alt }],
                  ['meta', { name: 'twitter:image:alt', content: picture.alt }])
      }
    }
    return head
  },

  markdown: {
    math: true,
    theme: { light: 'github-light', dark: 'github-dark' },
    anchor: { slugify },
    config(md) {
      // No :shortcode: emoji: C++ names such as std::thread::join must stay text.
      md.disable('emoji')
    },
  },

  vite: {
    server: { fs: { allow: ['..'] } },
    define: {
      __RAISIM_VERSION__: JSON.stringify(raisimVersion),
      __BUILD_YEAR__: JSON.stringify(buildYear),
    },
  },

  themeConfig: {
    logo: { src: '/logo-mark.svg', alt: 'RaiSim' },
    siteTitle: 'RaiSim',

    nav: [
      {
        text: 'Docs',
        link: '/sections/Overview.html',
        activeMatch: '^/sections/(?!(?:Web|WebAssembly|BrowserSceneEditor)(?:\\.html)?$|Benchmark$|Examples$|examples/|Rayrai$|rayrai/)',
      },
      { text: 'Examples', link: '/sections/Examples.html', activeMatch: '^/sections/(Examples$|examples/)' },
      { text: 'Rendering', link: '/sections/Rayrai.html', activeMatch: '^/sections/(Rayrai$|rayrai/)' },
      { text: 'Web', link: '/sections/Web.html', activeMatch: '^/sections/(?:Web|WebAssembly|BrowserSceneEditor)(?:\\.html)?$' },
      { text: 'Benchmark', link: '/sections/Benchmark.html', activeMatch: '^/sections/Benchmark$' },
      { text: 'API', link: '/api/', activeMatch: '^/api/' },
      {
        text: `v${raisimVersion}`,
        items: [
          { text: 'Changelog', link: '/sections/Changelog.html' },
          { text: 'Installation', link: '/sections/Installation.html' },
          { text: 'License', link: '/sections/License.html' },
        ],
      },
    ],

    sidebar: {
      '/sections/': docsSidebar,
      ...readApiSidebars(),
    },

    outline: { level: [2, 3], label: 'On this page' },
    notFound: {
      title: 'Page not found',
      quote: 'The documentation moved to a new site. The page may have a new name; try the search.',
      linkText: 'Go to the home page',
    },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: repoUrl }],
    editLink: {
      pattern: `${repoUrl}/edit/main/docs/:path`,
      text: 'Edit this page on GitHub',
    },
    footer: {
      message: 'RaiSim — physics for robotics and AI.',
      copyright: `© ${buildYear} RaiSim Tech Inc.`,
    },
  },
})

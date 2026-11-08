import { readFileSync } from 'node:fs'

// Build-time data for <ExampleGallery>: the categories of sections/examples/gallery.json
// and the title of every example page (sections/examples/**/*.md), keyed by the
// page's path below sections/examples without the extension.

export interface ExampleCategory {
  id: string
  title: string
  description: string
  /** Example pages, as paths below sections/examples without the extension. */
  examples: string[]
}

export interface ExampleGalleryData {
  categories: ExampleCategory[]
  /** Examples without a window, with the command their tile shows. */
  console: Record<string, string>
  titles: Record<string, string>
}

declare const data: ExampleGalleryData
export { data }

export default {
  watch: ['../../../sections/examples/**/*.md', '../../../sections/examples/gallery.json'],
  load(files: string[]): ExampleGalleryData {
    const titles: Record<string, string> = {}
    let gallery: { categories: ExampleCategory[]; console: Record<string, string> } | undefined
    for (const file of files) {
      const path = file.replace(/\\/g, '/')
      if (path.endsWith('/gallery.json')) {
        gallery = JSON.parse(readFileSync(file, 'utf8'))
        continue
      }
      const heading = readFileSync(file, 'utf8').match(/^# (.+)$/m)
      const key = path.replace(/^.*\/sections\/examples\//, '').replace(/\.md$/, '')
      if (heading) titles[key] = heading[1].trim()
    }
    if (!gallery) throw new Error('sections/examples/gallery.json is missing')
    return { categories: gallery.categories, console: gallery.console, titles }
  },
}

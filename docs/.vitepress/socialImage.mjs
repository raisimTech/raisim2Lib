import { existsSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'

/**
 * The picture of the card a shared link shows on Facebook, X and messengers.
 * @typedef {{ path: string, width: number, height: number, type: string, alt: string }} SocialImage
 */

// Formats Facebook and X both show; neither shows SVG.
const RASTER = /\.(png|jpe?g|gif)$/i
// X shows no card for a larger image; Facebook takes up to 8 MB.
const MAX_BYTES = 5 * 1024 * 1024
// Smaller pictures are icons and badges, too small for the card (X's large card needs 300 x 157,
// Facebook 200 x 200).
const MIN_WIDTH = 300
const MIN_HEIGHT = 200

/** @param {Buffer} bytes */
function jpegSize(bytes) {
  let offset = 2
  while (offset + 9 < bytes.length) {
    if (bytes[offset] !== 0xff) return undefined
    const marker = bytes[offset + 1]
    if (marker === 0xff) {
      offset += 1
      continue
    }
    // Start-of-frame markers, except DHT (C4), JPG (C8) and DAC (CC), hold the image size.
    if (marker >= 0xc0 && marker <= 0xcf && marker !== 0xc4 && marker !== 0xc8 && marker !== 0xcc)
      return { height: bytes.readUInt16BE(offset + 5), width: bytes.readUInt16BE(offset + 7) }
    offset += 2 + bytes.readUInt16BE(offset + 2)
  }
  return undefined
}

/**
 * Size and media type of a PNG, JPEG or GIF file, or undefined for anything else.
 * @param {string} file
 * @returns {{ width: number, height: number, type: string } | undefined}
 */
export function imageInfo(file) {
  if (!existsSync(file) || statSync(file).size > MAX_BYTES) return undefined
  const bytes = readFileSync(file)
  if (bytes.length >= 24 && bytes.readUInt32BE(0) === 0x89504e47 && bytes.toString('latin1', 12, 16) === 'IHDR')
    return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20), type: 'image/png' }
  if (bytes.length >= 10 && bytes.toString('latin1', 0, 4) === 'GIF8')
    return { width: bytes.readUInt16LE(6), height: bytes.readUInt16LE(8), type: 'image/gif' }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    const size = jpegSize(bytes)
    return size && { ...size, type: 'image/jpeg' }
  }
  return undefined
}

/** @param {string} text */
function decodeEntities(text) {
  return text
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCodePoint(parseInt(code, 16)))
    .replace(/&quot;/g, '"')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
}

/**
 * The first picture of a page's text that a card can show: a local PNG, JPEG or GIF of at most 5 MB
 * and at least 300 x 200 pixels. The navigation bar's logo comes before the text and is skipped.
 * @param {string} content the rendered page
 * @param {string} outDir the built site
 * @param {string} base the site's root path
 * @returns {SocialImage | undefined}
 */
export function firstContentImage(content, outDir, base) {
  const text = content.search(/class="vp-doc[\s"]/)
  if (text < 0) return undefined
  for (const [tag] of content.slice(text).matchAll(/<img\b[^>]*>/g)) {
    const src = tag.match(/\ssrc="([^"]+)"/)?.[1]
    if (!src || !src.startsWith(base)) continue
    const path = src.split(/[?#]/)[0]
    if (!RASTER.test(path)) continue
    const info = imageInfo(join(outDir, decodeURI(path.slice(base.length))))
    if (!info || info.width < MIN_WIDTH || info.height < MIN_HEIGHT) continue
    return { path, ...info, alt: decodeEntities(tag.match(/\salt="([^"]*)"/)?.[1] ?? '') }
  }
  return undefined
}

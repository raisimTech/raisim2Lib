// Line icons of the example gallery's categories (sections/examples/gallery.json),
// keyed by category id: the path data of a 24x24 icon drawn with a round 1.8 px
// stroke in the current text color.

function circle(cx: number, cy: number, r: number): string {
  return `M${cx - r} ${cy}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`
}

export const exampleCategoryIcons: Record<string, string[]> = {
  // A landscape with the sun.
  scenes: ['M5.5 4h13a2.5 2.5 0 0 1 2.5 2.5v11a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5v-11A2.5 2.5 0 0 1 5.5 4z', 'M3.5 17l5-6 4 4.5 2.5-3 5.5 5.5', circle(16, 8.5, 1.6)],
  // A robot's head.
  robots: ['M7.5 8h9a3 3 0 0 1 3 3v6a3 3 0 0 1-3 3h-9a3 3 0 0 1-3-3v-6a3 3 0 0 1 3-3z', 'M12 8V5', circle(12, 3.8, 1.2), circle(9.5, 13, 1.1), circle(14.5, 13, 1.1), 'M10 16.5h4', 'M2.5 12.5v3', 'M21.5 12.5v3'],
  // A glossy sphere and its shadow.
  rendering: [circle(12, 10.5, 7), 'M8.6 8.6a4.2 4.2 0 0 1 3.4-2.4', 'M6 20.5h12'],
  // A scanner and the arcs of its rays.
  sensors: [circle(5, 12, 1.8), 'M8.5 8.5a5 5 0 0 1 0 7', 'M11.4 5.6a9 9 0 0 1 0 12.8', 'M14.2 2.8a13 13 0 0 1 0 18.4'],
  // A solid box and a sphere.
  bodies: ['M3 7l5-3 5 3v6l-5 3-5-3V7z', 'M3 7l5 3 5-3', 'M8 10v6', circle(16.5, 16, 4.5)],
  // A rotating body and an orbital arrow.
  dynamics: ['M8 9l4-2 4 2v6l-4 2-4-2V9z', 'M8 9l4 2 4-2', 'M12 11v6', 'M18 4.5c2.5 1.7 4 4.4 4 7.5 0 5.5-4.5 10-10 10S2 17.5 2 12c0-3.1 1.5-5.8 4-7.5', 'M18 8V4.5h3.5'],
  // A stack of boxes resting on each other.
  contact: ['M3 20h18', 'M6 14h6v6H6z', 'M13.5 14h5v6h-5z', 'M9 8h6v6H9z'],
  // Contour lines of a height map.
  terrain: ['M3 9.5C6 6.5 8.5 10.5 12 7.5S17.5 5.5 21 7.5', 'M3 14.5C6.5 11.5 9 15.5 12.5 12.5S18 10.5 21 12.5', 'M3 19.5C7 16.5 9.5 20.5 13 17.5S18.5 15.5 21 17.5'],
  // A cloth draped between its corners.
  deformable: ['M4 5c4 4 12 4 16 0l-2 13c-4 3-8 3-12 0L4 5z', 'M9 8l1 12', 'M15 8l-1 12', 'M5 12c4 3 10 3 14 0'],
  // A float bobbing on waves.
  fluids: [circle(12, 8, 3.5), 'M3 14c3-2.5 6 2.5 9 0s6 2.5 9 0', 'M3 19c3-2.5 6 2.5 9 0s6 2.5 9 0'],
  // A four-bar linkage, a closed loop.
  joints: ['M5 18L8.5 7.5 17.5 9.5 19.5 18', 'M3 20.5h18', circle(5, 18, 1.6), circle(8.5, 7.5, 1.6), circle(17.5, 9.5, 1.6), circle(19.5, 18, 1.6)],
  // A terminal window.
  tools: ['M5.5 4.5h13A2.5 2.5 0 0 1 21 7v10a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17V7a2.5 2.5 0 0 1 2.5-2.5z', 'M7 9.5l3 2.5-3 2.5', 'M12.5 15H17'],
  // A speedometer.
  benchmarks: ['M4 16.9A8.5 8.5 0 1 1 20 16.9', 'M12 14l4.5-5.5', circle(12, 14, 1.5), 'M6.5 19.5h11'],
}

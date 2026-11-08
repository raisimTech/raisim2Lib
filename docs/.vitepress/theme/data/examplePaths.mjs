export function examplePageKey(file) {
  return file.replace(/\\/g, '/').replace(/^(?:.*\/)?sections\/examples\//, '').replace(/\.md$/, '');
}

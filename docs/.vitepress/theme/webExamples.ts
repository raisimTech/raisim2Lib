import { withBase } from 'vitepress'

// The examples a browser runs, read once from the deployed browser tools: each name maps to the
// page that runs it. RaisimServer examples run in the TCP viewer, rayrai examples on a page of
// their own. The map is empty when the site was built without the browser tools.
let listed: Promise<Map<string, string>> | undefined

function names(file: string): Promise<string[]> {
  return fetch(withBase(`/web-examples/${file}`), { cache: 'no-cache' })
    .then((response) => (response.ok ? response.text() : ''))
    .then((text) => text.split('\n').filter((line) => line && !line.startsWith('#')).map((line) => line.split('\t')[0]))
    .catch(() => [])
}

export function webExamples(): Promise<Map<string, string>> {
  listed ??= Promise.all([names('server-examples.tsv'), names('rayrai-examples.tsv')]).then(
    ([server, rayrai]) =>
      new Map([
        ...server.map((name): [string, string] => [name, withBase(`/web-examples/index.html?example=${name}`)]),
        ...rayrai.map((name): [string, string] => [name, withBase(`/web-examples/example.html?example=${name}`)]),
      ]),
  )
  return listed
}

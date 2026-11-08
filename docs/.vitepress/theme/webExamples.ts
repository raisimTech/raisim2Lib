import { withBase } from 'vitepress'

// The examples a browser runs, read once from the deployed browser tools: each name maps to the
// page that runs it. RaisimServer examples run in the TCP viewer, rayrai examples on a page of
// their own, and games such as Tank Survival on the pages games/games.tsv lists. The map is empty
// when the site was built without the browser tools.
let listed: Promise<Map<string, string>> | undefined

function rows(path: string): Promise<string[][]> {
  return fetch(withBase(path), { cache: 'no-cache' })
    .then((response) => (response.ok ? response.text() : ''))
    .then((text) => text.split(/\r?\n/).filter((line) => line && !line.startsWith('#')).map((line) => line.split('\t')))
    .catch(() => [])
}

// A games.tsv page is a folder of games/ (letters, digits and dashes): nothing else is linked.
const gamePage = /^[a-z0-9-]+\/$/

export function webExamples(): Promise<Map<string, string>> {
  listed ??= Promise.all([
    rows('/web-examples/server-examples.tsv'),
    rows('/web-examples/rayrai-examples.tsv'),
    rows('/games/games.tsv'),
  ]).then(
    ([server, rayrai, games]) =>
      new Map([
        ...server.map(([name]): [string, string] => [name, withBase(`/web-examples/index.html?example=${name}`)]),
        ...rayrai.map(([name]): [string, string] => [name, withBase(`/web-examples/example.html?example=${name}`)]),
        ...games
          .filter(([, page]) => page !== undefined && gamePage.test(page))
          .map(([name, page]): [string, string] => [name, withBase(`/games/${page}`)]),
      ]),
  )
  return listed
}

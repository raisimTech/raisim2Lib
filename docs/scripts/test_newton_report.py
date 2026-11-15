"""Keep the Newton report aligned with its data and GPU-only scaling rows."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

DOCS = Path(__file__).resolve().parents[1]
REPORT = DOCS / "public" / "raisim_newton_ten_step_value.html"
DATA = DOCS / "public" / "raisim_newton_ten_step_value.json"


class NewtonReportTest(unittest.TestCase):
    def test_report_embeds_the_downloadable_results(self):
        match = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                          REPORT.read_text(), re.S)
        self.assertIsNotNone(match)
        self.assertEqual(json.loads(match.group(1)), json.loads(DATA.read_text()))

    @unittest.skipUnless(shutil.which("node"), "Node.js is unavailable")
    def test_price_controls_and_gpu_only_batches(self):
        script = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const html = fs.readFileSync(process.argv[2], 'utf8');
const raw = html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const data = JSON.parse(raw);
const source = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].at(-1)[1];
const elements = new Map();
const get = id => {
  if (!elements.has(id)) elements.set(id, {
    value: '', textContent: '', innerHTML: '', listeners: {},
    addEventListener(type, fn) { this.listeners[type] = fn; },
  });
  return elements.get(id);
};
get('data').textContent = raw;
get('cpu').value = String(data.cpu_price_usd);
get('gpu').value = String(data.gpu_price_usd);
let csv;
const context = vm.createContext({
  document: {getElementById: get, createElement: () => ({click() {}})},
  Blob, setTimeout: fn => fn(),
  URL: {createObjectURL: blob => { csv = blob; return 'blob:test'; }, revokeObjectURL() {}},
});
vm.runInContext(source, context);
const original = get('results').innerHTML;
assert.doesNotMatch(original, /NaN|Infinity/);
const paired = data.rows.filter(row => row.cpu_fps != null);
const wins = paired.filter(row => row.cpu_fps / data.cpu_price_usd > row.gpu_fps / data.gpu_price_usd);
assert.ok(get('value-summary').textContent.includes(`${wins.length} of ${paired.length} paired comparisons`));
assert.equal((original.match(/<rect /g) || []).length, data.rows.length + paired.length);
assert.equal((original.match(/Not measured/g) || []).length, 2 * (data.rows.length - paired.length));
for (const [, x, width] of original.matchAll(/<rect x="([^"]+)" y="[^"]+" width="([^"]+)"/g)) {
  assert.ok(Number(x) >= 0 && Number(x) + Number(width) <= 1000);
}
const values = vm.runInContext('values()', context);
for (const row of values.filter(row => row.cpu_fps == null)) {
  assert.equal(row.cpu_value, null);
  assert.equal(row.speed_ratio, null);
  assert.equal(row.ratio, null);
}
get('gpu').value = '0';
get('gpu').listeners.input();
assert.match(get('error').textContent, /positive, finite prices/);
assert.equal(get('results').innerHTML, original);
get('reset').onclick();
assert.equal(get('error').textContent, '');
assert.equal(get('results').innerHTML, original);
get('gpu').value = String(data.gpu_price_usd * 2);
get('gpu').listeners.input();
assert.notEqual(get('results').innerHTML, original);
assert.doesNotMatch(get('results').innerHTML, /NaN|Infinity/);
get('download').onclick();
const lines = (await csv.text()).split('\r\n');
assert.equal(lines.length, data.rows.length + 1);
assert.ok(lines[0].includes('cpu_fps') && lines[0].includes('speed_ratio'));
"""
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-", str(REPORT)],
                                input=script, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

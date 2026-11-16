#!/usr/bin/env python3
"""Render the manipulation FPS/USD chart from recorded measurements.

An optional output path allows previewing without changing the checked-in chart.
"""
import argparse
from html import escape
import json
from pathlib import Path


def chart(data):
    prices = data['value_comparison']
    rows = data['rows']
    values = [(r['cpu_fps'] / prices['cpu_price_usd']
               if r['cpu_fps'] is not None else None,
               r['gpu_fps'] / prices['gpu_price_usd']) for r in rows]
    ceiling = max(v for pair in values for v in pair if v is not None) * 1.12
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1152 560" role="img" aria-labelledby="title desc">',
             '<title id="title">Robotiq manipulation: performance per USD</title>',
             '<desc id="desc">Aggregate world physics steps per second divided by recorded component price. Hatched Newton bars failed mimic validation; CPU at 8096 worlds is unmeasured.</desc>',
             '<defs><pattern id="failed" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#e7daf0"/><path d="M-2 2L2-2M0 8L8 0M6 10L10 6" stroke="#8c49ba" stroke-width="2"/></pattern></defs>',
             '<rect width="1152" height="560" fill="white"/>',
             '<g font-family="Arial, Helvetica, sans-serif" fill="#24313d">',
             '<text x="90" y="38" font-size="26" font-weight="bold">Robotiq manipulation · performance per USD</text>',
             f'<text x="90" y="68" font-size="17">Recorded prices: CPU ${prices["cpu_price_usd"]:.2f} / GPU ${prices["gpu_price_usd"]:.2f} · higher is better</text>',
             '<text x="90" y="98" font-size="14">Historical scenarios · checked 15 Nov 2026 · CPU out of stock · GPU board is a pricing proxy</text>']
    for k in range(5):
        y = 420 - k * 70
        parts.append(f'<path d="M90 {y}H1120" stroke="#e1e4e5"/><text x="80" y="{y+5}" text-anchor="end" font-size="14">{ceiling*k/4:,.0f}</text>')
    for i, (row, pair) in enumerate(zip(rows, values)):
        x = 105 + i * 127
        for j, value in enumerate(pair):
            if value is None:
                parts.append(f'<text x="{x+18}" y="408" text-anchor="middle" font-size="11">N/A</text>')
                continue
            failed = j == 1 and not row['gpu_quality_pass']
            color = 'url(#failed)' if failed else ('#00846e' if j == 0 else '#8c49ba')
            h = value / ceiling * 280
            label = f'{"RaiSim" if j == 0 else "Newton"}: {value:,.2f} FPS/USD at {row["worlds"]:,} worlds' + ('; FAILED mimic validation' if failed else '; passed validation')
            parts.append(f'<rect x="{x+j*42}" y="{420-h:.3f}" width="36" height="{h:.3f}" fill="{color}"><title>{escape(label)}</title></rect>')
            if failed:
                parts.append(f'<text x="{x+j*42+18}" y="{412-h:.3f}" text-anchor="middle" font-size="18">†</text>')
        parts.append(f'<text x="{x+39}" y="447" text-anchor="middle" font-size="15">{row["worlds"]:,}</text>')
    parts += ['<text x="605" y="475" text-anchor="middle" font-size="16">Independent worlds</text>',
              '<rect x="90" y="497" width="20" height="16" fill="#00846e"/><text x="118" y="511" font-size="15">RaiSim · up to 30 workers</text>',
              '<rect x="390" y="497" width="20" height="16" fill="#8c49ba"/><text x="418" y="511" font-size="15">Newton · RTX 5070</text>',
              '<rect x="660" y="497" width="20" height="16" fill="url(#failed)"/><text x="688" y="511" font-size="15">† Newton failed validation</text>',
              '<text x="90" y="543" font-size="14">FPS/USD = aggregate world-steps/s ÷ full component price. N/A = unmeasured; failed cases have no ratios.</text>',
              '</g></svg>']
    return '\n'.join(parts) + '\n'


def main():
    docs = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=docs.parent / 'rsc/docs/image/raisim_newton_manipulation_fps_per_usd.svg')
    args = parser.parse_args()
    data = json.loads((docs / 'public/raisim_newton_manipulation.json').read_text(encoding='utf-8'))
    args.output.write_text(chart(data), encoding='utf-8')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render the manipulation FPS/USD chart from recorded measurements.

An optional output path allows previewing without changing the checked-in chart.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from newton_charts import value_chart  # noqa: E402


def chart(data):
    prices = data['value_comparison']
    rows = data['rows']
    return value_chart(
        rows, prices['cpu_price_usd'], prices['gpu_price_usd'], 'Robotiq manipulation',
        'random shapes, gripper torque noise, staggered resets',
        ['Historical scenarios: CPU out of stock, the GPU board is a pricing proxy (checked 15 Nov 2026).',
         'Hatched Newton columns failed mimic validation; they are timings, not accepted scores.'],
        failed=[not row['gpu_quality_pass'] for row in rows])


def main():
    docs = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=docs.parent / 'rsc/docs/image/raisim_newton_manipulation_fps_per_usd.svg')
    args = parser.parse_args()
    data = json.loads((docs / 'public/raisim_newton_manipulation.json').read_text(encoding='utf-8'))
    args.output.write_text(chart(data), encoding='utf-8')


if __name__ == '__main__':
    main()

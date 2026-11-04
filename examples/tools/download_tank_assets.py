#!/usr/bin/env python3
"""Download the sources of the tank survival example's assets (raisim2Lib/wasm) into a cache folder:
the armored vehicles from Sketchfab (CC BY 4.0, credited in rsc/tank_survival/ATTRIBUTION.md) and props
and ground textures from Poly Haven (CC0). prepare_tank_assets.py turns them into rsc/tank_survival.

The Sketchfab downloads need the API token of a free Sketchfab account: set SKETCHFAB_API_TOKEN or
write the token to ~/.sketchfab_api_token.
"""
import argparse
import hashlib
import json
import os
import pathlib
import urllib.error
import urllib.request

# name: (Sketchfab model uid, title, author, author's profile)
VEHICLES = {
    'leopard': ('80b589fc9c0b4720888b40d01d6e5153', 'Leopard 2A4M Main Battle Tank', 'MirzaArrafiERV_45'),
    't72': ('dfe130dfce724c30958cec51f2bd86b7', 'T-72B3 Main Battle Tank', 'MirzaArrafiERV_45'),
    't90': ('619de9f0908e4c8395fe036ffe6e3884', 'Uralvagonzavod T-90AM Main Battle Tank', 'MirzaArrafiERV_45'),
    'bmd2': ('16c4113b6592474baa897902e8fbc637', 'Russian BMD-2M', '42manako'),
    'sprut': ('bca9950218c545ecbd70b15704a35bf6', '2S25 Sprut-SD (Armored Warfare)', 'KojfDiscord'),
    'msta': ('c12432ada9a84859bc74c3d5ef7046f8', '2S19 MSTA-S Self-Propelled Artillery', 'MirzaArrafiERV_45'),
    'armata': ('24c9a06973a64651a8b0baedee29fd57', 'T-14 Armata', '42manako'),
}
PROPS = ['barrel_03', 'wooden_military_crate', 'old_military_crate', 'ammo_box', 'metal_jerrycan_green',
         'concrete_road_barrier', 'old_tyre', 'shrub_02', 'shrub_04', 'wild_rooibos_bush', 'tree_stump_01',
         'namaqualand_boulder_04', 'namaqualand_boulder_05']
GROUNDS = ['sparse_grass', 'brown_mud_rocks_01', 'aerial_sand', 'snow_02', 'burned_ground_01', 'rubble',
           'cracked_concrete', 'concrete_wall_006']


def read(url, headers=None, attempts=4):
    request = urllib.request.Request(url, headers={'User-Agent': 'raisim2Lib asset downloader', **(headers or {})})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise


def sketchfab_token():
    token = os.environ.get('SKETCHFAB_API_TOKEN', '').strip()
    path = pathlib.Path.home() / '.sketchfab_api_token'
    if not token and path.exists():
        token = path.read_text().strip()
    if not token:
        raise SystemExit('The vehicles come from Sketchfab, which needs an API token: set SKETCHFAB_API_TOKEN or '
                         'write the token to ~/.sketchfab_api_token')
    return token


def save(cache, relative, fetch, url, sources):
    """Downloads url to cache/relative unless an earlier run did, and records its source."""
    target = cache / relative
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fetch())
    data = target.read_bytes()
    sources.append(dict(path=relative, url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=pathlib.Path, required=True, help='Download folder (outside the checkout)')
    parser.add_argument('--sources', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2] /
                        'rsc' / 'tank_survival' / 'sources.json')
    args = parser.parse_args()
    sources = []
    token = sketchfab_token()
    for name, (uid, title, author) in VEHICLES.items():
        glb = lambda: read(json.loads(read(f'https://api.sketchfab.com/v3/models/{uid}/download',
                                           {'Authorization': f'Token {token}'}))['glb']['url'])
        save(args.cache, f'vehicles/{name}.glb', glb, f'https://sketchfab.com/3d-models/{uid}', sources)
        print('vehicle', name, title, 'by', author)
    for name in PROPS:
        info = json.loads(read('https://api.polyhaven.com/files/' + name))['gltf']['1k']['gltf']
        for relative, entry in {f'{name}_1k.gltf': info, **info['include']}.items():
            save(args.cache, f'models/{name}/{relative}', lambda url=entry['url']: read(url), entry['url'], sources)
        print('prop', name)
    for name in GROUNDS:
        files = json.loads(read('https://api.polyhaven.com/files/' + name))
        for key in ('Diffuse', 'nor_gl'):
            entry = files[key]['1k']['jpg']
            save(args.cache, f'grounds/{name}_{key.lower()}_1k.jpg', lambda url=entry['url']: read(url), entry['url'],
                 sources)
        print('ground', name)
    args.sources.parent.mkdir(parents=True, exist_ok=True)
    args.sources.write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()

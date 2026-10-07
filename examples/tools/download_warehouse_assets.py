#!/usr/bin/env python3
"""Download the source assets of the rayrai warehouse example into DIR/assets.

The props and the floor, wall and roof textures come from Poly Haven (CC0).
The forklift, the pallet jack, the pallets and the cartons are Sketchfab
models under CC-BY-4.0, credited in rsc/warehouse/ATTRIBUTION.md. Sketchfab
downloads need the API token of a free account
(https://sketchfab.com/settings/password), read from the SKETCHFAB_API_TOKEN
environment variable or ~/.sketchfab_api_token. The sky and the yard asphalt
are reused from rsc/city, and the pallet racking, the building and the light
fixtures are generated. Only Python's standard library is used.
"""
import hashlib
import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile

USER_AGENT = {'User-Agent': 'RaisimWarehouseExample/1.0'}
# Poly Haven model id -> glTF resolution.
MODELS = {
    'rollershutter_door': '1k',
    'hand_truck': '1k',
    'cardboard_box_01': '1k',
    'Barrel_01': '1k',
    'barrel_03': '1k',
    'korean_fire_extinguisher_01': '1k',
    'plastic_crate_03': '1k',
    'propane_tank': '1k',
    'WetFloorSign_01': '1k',
    'steel_frame_shelves_01': '1k',
    'security_camera_01': '1k',
    'metal_office_desk': '1k',
    'tool_cart': '1k',
}
# Poly Haven texture id -> resolution; colour, OpenGL normal and packed
# AO/roughness/metalness maps are downloaded for each.
TEXTURES = {'concrete_floor_worn_001': '2k', 'corrugated_iron': '2k',
            'concrete_wall_004': '2k', 'corrugated_iron_02': '1k'}
TEXTURE_MAPS = ['Diffuse', 'nor_gl', 'arm']
# Sketchfab models, CC-BY-4.0: local name -> model uid.
SKETCHFAB = {
    'forklift': '060f3f8bc7de4e6ca2f348d414702e9d',         # Forklift Truck, louis-muir
    'pallet_jack': '55397e1bb66a48bd9d6e1ef93d8456cc',      # Pallet Jack (Low Poly), Berk Gedik
    'pallets': 'dba5c00928cd400796d9f6fffdd724b3',          # Wooden Pallets, YadroGames
    'cardboard_boxes': '8986ba512f704ac5b253286a0d1ad8bb',  # Set of Cardboard Boxes
}


def read(url, headers=None):
    request = urllib.request.Request(url, headers={**USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=300).read()


def sketchfab_token():
    token = os.environ.get('SKETCHFAB_API_TOKEN', '').strip()
    path = pathlib.Path.home() / '.sketchfab_api_token'
    if not token and path.exists():
        token = path.read_text().strip()
    if not token:
        raise SystemExit('The forklift, pallets and cartons come from Sketchfab, which needs an API '
                         'token: set SKETCHFAB_API_TOKEN or write the token to ~/.sketchfab_api_token '
                         '(https://sketchfab.com/settings/password).')
    return token


def fetch(root, manifest, relative, url, md5=None):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(read(url))
    data = path.read_bytes()
    if md5 is not None and hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError('Checksum mismatch: ' + str(path))
    manifest.append(dict(path=relative, url=url, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: download_warehouse_assets.py DIR')
    root = pathlib.Path(sys.argv[1]) / 'assets'
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    print('Powered by Poly Haven: https://polyhaven.com', flush=True)
    for name, resolution in MODELS.items():
        files = json.loads(read('https://api.polyhaven.com/files/' + name))
        gltf = files['gltf'][resolution]['gltf']
        entries = {f'{name}_{resolution}.gltf': gltf, **gltf['include']}
        print(name, '%.1f MiB' % (sum(f['size'] for f in entries.values()) / 2**20), flush=True)
        for relative, entry in entries.items():
            fetch(root, manifest, f'{name}/{relative}', entry['url'], entry['md5'])
    for name, resolution in TEXTURES.items():
        files = json.loads(read('https://api.polyhaven.com/files/' + name))
        for key in TEXTURE_MAPS:
            entry = files[key][resolution]['jpg']
            fetch(root, manifest, f'textures/{name}/' + entry['url'].rsplit('/', 1)[1],
                  entry['url'], entry['md5'])
        print(name, flush=True)

    # Sketchfab hands out short-lived links, so the manifest records the model page.
    token = None
    for name, uid in SKETCHFAB.items():
        target = root / 'sketchfab' / name
        if not (target / 'scene.gltf').exists():
            token = token or sketchfab_token()
            links = json.loads(read(f'https://api.sketchfab.com/v3/models/{uid}/download',
                                    {'Authorization': 'Token ' + token}))
            zipfile.ZipFile(io.BytesIO(read(links['gltf']['url']))).extractall(target)
        page = f'https://sketchfab.com/3d-models/{uid}'
        for path in sorted(target.rglob('*')):
            if path.is_file():
                data = path.read_bytes()
                manifest.append(dict(path=str(path.relative_to(root)), url=page,
                                     sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
        print('Sketchfab:', name, flush=True)
    (root / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()

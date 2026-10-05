#!/usr/bin/env python3
"""Download the source assets of the rayrai city example into DIR/assets.

Building kits, street props, ground textures and the sky come from Poly Haven
(CC0); the traffic cone comes from BlendKit (CC0), whose free asset files
download without an account. The parked cars are Daniel Zhabotinsky's
fictional-brand models on Sketchfab (CC-BY-4.0, credited in
rsc/city/ATTRIBUTION.md). Sketchfab downloads need the API token of a free
account (https://sketchfab.com/settings/password), read from the
SKETCHFAB_API_TOKEN environment variable or ~/.sketchfab_api_token.
Only Python's standard library is used.
"""
import hashlib
import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile

USER_AGENT = {'User-Agent': 'RaisimCityExample/1.0'}
# Poly Haven model id -> glTF resolution. The two facade kits cover most of
# the screen, so they use 2K textures; small props use 1K.
MODELS = {
    'modular_urban_apartments_facade': '2k',
    'modular_factory_facade': '2k',
    'street_lamp_01': '1k',
    'fire_hydrant': '1k',
    'metal_trash_can': '1k',
    'modular_street_seating': '1k',
    'water_manhole_cover': '1k',
    'utility_box_02': '1k',
    'concrete_road_barrier': '1k',
}
# Poly Haven texture id -> resolution; colour, OpenGL normal and packed
# AO/roughness/metalness maps are downloaded for each.
TEXTURES = {'asphalt_02': '2k', 'concrete_pavement_02': '2k',
            'concrete_floor_01': '1k', 'bitumen': '1k'}
TEXTURE_MAPS = ['Diffuse', 'nor_gl', 'arm']
SKY = ('kloofendal_48d_partly_cloudy_puresky', '4k')
# BlendKit "Traffic Cone (Photoscanned)" by Nik Kottmann, CC0.
BLENDKIT_CONE = '4aec6ae7-3b6f-4888-90ec-2e77f890c223'
# Sketchfab models by Daniel Zhabotinsky, CC-BY-4.0: local name -> model uid.
SKETCHFAB_CARS = {
    'fairheaven_lt80': 'e2678da920cc4be68dbc193727919ffb',  # Fairheaven LT '80 sedan
    'fairheaven_sw84': 'aa0becb6e854422596cac6b21bf79787',  # Fairheaven SW '84 wagon
    'kiri86': '5c17c582ef9549798f694b660feea42a',           # Kiri '86 compact
    'canyon75_taxi': '9e8f92a215784f3d8aaaaeab1bef54c4',    # Canyon '75 taxi
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
        raise SystemExit('The parked cars come from Sketchfab, which needs an API token: set '
                         'SKETCHFAB_API_TOKEN or write the token to ~/.sketchfab_api_token '
                         '(https://sketchfab.com/settings/password).')
    return token


def fetch(root, manifest, relative, url, md5=None, source=None):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(read(url))
    data = path.read_bytes()
    if md5 is not None and hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError('Checksum mismatch: ' + str(path))
    manifest.append(dict(path=relative, url=source or url,
                         sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))


def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: download_city_assets.py DIR')
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
    name, resolution = SKY
    entry = json.loads(read('https://api.polyhaven.com/files/' + name))['hdri'][resolution]['hdr']
    fetch(root, manifest, 'sky/' + entry['url'].rsplit('/', 1)[1], entry['url'], entry['md5'])
    print(name, flush=True)

    # BlendKit hands out short-lived signed links, so the manifest records the
    # asset page instead of the link that was used.
    asset = json.loads(read(f'https://www.blendkit.com/api/v1/assets/{BLENDKIT_CONE}/'))
    if asset['license'] != 'cc_zero' or not asset['isFree']:
        raise RuntimeError('The BlendKit traffic cone is no longer a free CC0 asset')
    # The Godot export is the plain glTF (the other one is Draco-compressed).
    download = next(f['downloadUrl'] for f in asset['files'] if f['fileType'] == 'gltf_godot')
    signed = json.loads(read(download + '?scene_uuid=00000000-0000-0000-0000-000000000000'))
    fetch(root, manifest, 'traffic_cone/traffic_cone.glb', signed['filePath'],
          source=f'https://www.blendkit.com/asset-gallery-detail/{BLENDKIT_CONE}/')
    print('BlendKit:', asset['name'], 'by', asset['author']['fullName'], flush=True)

    # Sketchfab also hands out short-lived links: the manifest records the model page.
    token = None
    for name, uid in SKETCHFAB_CARS.items():
        target = root / 'cars' / name
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

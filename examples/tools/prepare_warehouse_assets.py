#!/usr/bin/env python3
"""Prepare the rayrai warehouse's runtime assets from download_warehouse_assets.py's DIR/assets.

Usage: prepare_warehouse_assets.py DIR [OUTPUT]   (OUTPUT defaults to DIR/prepared)

* Models: every prop, the forklift, the pallet jack, the pallets and the
  cartons are baked into Z-up glTF files in metres: node transforms go into
  the vertices (RaiSim reads a dynamic prop's collision hull from the same
  file), parts are merged into one primitive per material, the model is
  centred on its footprint with its base at z = 0, and vehicles face +x.
  Static props get a .rasset descriptor with simple colliders.
* Sketchfab textures are reduced to 1K or 2K JPEG files; the pallets'
  specular-glossiness materials become metallic-roughness ones.
* The floor's colour map is brightened. Painted steel, the rack uprights'
  slot pattern, worn floor paint and hazard stripes are generated, tileable
  textures.

The sky and the yard asphalt are reused from rsc/city. Finally
generate_warehouse_rscene.py builds the warehouse and writes the scene.
Requires NumPy, SciPy and Pillow.
"""
import copy
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

import generate_warehouse_rscene
from prepare_city_assets import dump, flatten_scene, load, node_matrix, write_rasset

TURN = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # glTF +y up to +z up
# Poly Haven props: scale to metres, yaw (degrees) applied after the Z-up turn,
# nodes to keep (default all) and colliders in the prepared frame. A prop
# without colliders is dynamic or visual only.
PROPS = {
    'rollershutter_door': dict(nodes=['rollershutter_door'], colliders=[]),
    'hand_truck': dict(colliders=[('box', dict(size=(0.59, 0.69, 1.4)), (0, 0, 0.7))]),
    'cardboard_box_01': dict(colliders=[]),
    'Barrel_01': dict(colliders=[('cylinder', dict(radius=0.28, height=0.88), (0, 0, 0.44))]),
    'barrel_03': dict(colliders=[('cylinder', dict(radius=0.317, height=0.93), (0, 0, 0.465))]),
    'korean_fire_extinguisher_01': dict(colliders=[('box', dict(size=(0.28, 0.36, 0.66)), (0, 0, 0.33))]),
    'plastic_crate_03': dict(colliders=[]),
    'propane_tank': dict(colliders=[('cylinder', dict(radius=0.169, height=0.554), (0, 0, 0.277))]),
    'WetFloorSign_01': dict(colliders=[]),
    'steel_frame_shelves_01': dict(scale=0.1, colliders=[('box', dict(size=(1.1, 0.5, 2.14)), (0, 0, 1.07))]),
    'security_camera_01': dict(yaw=90, colliders=[]),
    'metal_office_desk': dict(colliders=[('box', dict(size=(2.0, 0.95, 0.79)), (0, 0, 0.395))]),
    'tool_cart': dict(colliders=[('box', dict(size=(1.27, 0.75, 0.96)), (0, 0, 0.48))]),
}
# Sketchfab models: uniform or per-axis scale (after the turn), yaw, and the
# texture size limit.
VEHICLES = {
    'forklift': dict(scale=0.008, yaw=-90, texture=2048),
    'pallet_jack': dict(scale=1.0, yaw=90, texture=1024),
}
# Pallets are scaled to 1.2 m x 1.0 m x 0.144 m; the model's two pallets are 2 m squares.
PALLETS = {'pallet_plywood': 'Pallet_1', 'pallet_slatted': 'Pallet_2'}
PALLET_SIZE = (1.2, 1.0, 0.144)
# Carton variants of the Sketchfab set, named by their size in millimetres.
CARTONS = ['430x210x270_1', '430x210x270_2', '290x170x190_1', '290x290x290_1',
           '290x290x400_1', '290x290x400_2']
TEXTURES = ['concrete_floor_worn_001', 'corrugated_iron', 'concrete_wall_004', 'corrugated_iron_02']
# The worn concrete is a dark slab (about 9 % albedo); a warehouse floor is
# lighter, so its colour map is scaled in linear light.
FLOOR_GAIN = 2.4


def matrix(scale=1.0, yaw=0.0):
    """4x4 root transform: the Z-up turn, then a yaw about +z, scaled."""
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    root = np.eye(4)
    root[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag(np.broadcast_to(scale, 3)) @ TURN
    return root


def ancestors(document, index):
    """Product of the transforms of a node's ancestors, from the scene root down."""
    parents = {child: i for i, node in enumerate(document['nodes']) for child in node.get('children', [])}
    result = np.eye(4)
    while index in parents:
        index = parents[index]
        result = node_matrix(document['nodes'][index]) @ result
    return result


def flatten_node(document, binary, name, root):
    """flatten_scene() of the named node and its children, under its ancestors' transforms."""
    index = next(i for i, node in enumerate(document['nodes']) if node.get('name') == name)
    scene = dict(document, scenes=[dict(nodes=[index])], scene=0)
    return flatten_scene(scene, binary, root @ ancestors(document, index))


def merge_by_material(parts):
    merged = {}
    for part in parts:
        group = merged.setdefault(part['material'], dict(material=part['material'], POSITION=[], NORMAL=[],
                                                         TEXCOORD_0=[], indices=[], count=0))
        for key in ('POSITION', 'NORMAL', 'TEXCOORD_0'):
            group[key].append(part[key])
        group['indices'].append(part['indices'] + group['count'])
        group['count'] += len(part['POSITION'])
    return [dict(material=g['material'], indices=np.vstack(g['indices']),
                 **{key: np.vstack(g[key]) for key in ('POSITION', 'NORMAL', 'TEXCOORD_0')})
            for g in merged.values()]


def recentre(parts):
    """Centre the parts on their footprint with the lowest point at z = 0; returns the size."""
    points = np.vstack([part['POSITION'] for part in parts])
    low, high = points.min(0), points.max(0)
    shift = np.array([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
    for part in parts:
        part['POSITION'] = part['POSITION'] + shift
    return high - low


def write_mesh(target, stem, parts, document, images, generator='prepare_warehouse_assets.py'):
    """One glTF file and buffer holding parts; materials, textures and samplers come from document."""
    data = bytearray()
    views, accessors, primitives = [], [], []

    def add(array, kind, target_kind, bounds=False):
        while len(data) % 4:
            data.append(0)
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=array.nbytes, target=target_kind))
        data.extend(array.tobytes())
        accessor = dict(bufferView=len(views) - 1, componentType=5125 if kind == 'SCALAR' else 5126,
                        count=int(array.size if kind == 'SCALAR' else len(array)), type=kind)
        if bounds:
            accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
        accessors.append(accessor)
        return len(accessors) - 1

    for part in parts:
        attributes = {key: add(part[key].astype(np.float32), kind, 34962, key == 'POSITION')
                      for key, kind in (('POSITION', 'VEC3'), ('NORMAL', 'VEC3'), ('TEXCOORD_0', 'VEC2'))}
        primitive = dict(attributes=attributes, indices=add(part['indices'].ravel().astype(np.uint32),
                                                            'SCALAR', 34963))
        if part['material'] is not None:
            primitive['material'] = part['material']
        primitives.append(primitive)
    model = dict(asset=dict(version='2.0', generator=generator), scene=0, scenes=[dict(nodes=[0])],
                 nodes=[dict(name=stem, mesh=0)], meshes=[dict(name=stem, primitives=primitives)],
                 buffers=[dict(uri=f'{stem}.bin', byteLength=len(data))], bufferViews=views,
                 accessors=accessors)
    for key in ('materials', 'textures', 'samplers'):
        if document.get(key):
            model[key] = copy.deepcopy(document[key])
    if images:
        model['images'] = images
    used = {e for m in model.get('materials', []) for e in m.get('extensions', {})}
    if used:
        model['extensionsUsed'] = sorted(used)
    target.mkdir(parents=True, exist_ok=True)
    (target / f'{stem}.bin').write_bytes(bytes(data))
    dump(model, target / f'{stem}.gltf')


def convert_images(document, source, target, limit, alpha=()):
    """Reduce each image to limit pixels; JPEG unless its index is in alpha. Returns glTF images."""
    from PIL import Image
    (target / 'textures').mkdir(parents=True, exist_ok=True)
    images = []
    for index, image in enumerate(document.get('images', [])):
        picture = Image.open(source / image['uri'])
        picture.thumbnail((limit, limit), Image.LANCZOS)
        stem = Path(image['uri']).stem
        if index in alpha:
            images.append(dict(uri=f'textures/{stem}.png'))
            picture.save(target / images[-1]['uri'], optimize=True)
        else:
            images.append(dict(uri=f'textures/{stem}.jpg'))
            picture.convert('RGB').save(target / images[-1]['uri'], quality=90)
    return images


def prepare_prop(source, output, name, spec):
    document = load(source / name / f'{name}_1k.gltf')
    binary = (source / name / document['buffers'][0]['uri']).read_bytes()
    root = matrix(spec.get('scale', 1.0), spec.get('yaw', 0.0))
    if 'nodes' in spec:
        parts = [part for name in spec['nodes'] for part in flatten_node(document, binary, name, root)]
    else:
        parts = flatten_scene(document, binary, root)
    parts = merge_by_material(parts)
    size = recentre(parts)
    target = output / 'props' / name
    for image in document['images']:
        (target / image['uri']).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name / image['uri'], target / image['uri'])
    write_mesh(target, 'model', parts, document, document['images'])
    if spec['colliders']:
        write_rasset(target / 'model.rasset', 'model.gltf', spec['colliders'])
    print(f'{name}: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, '
          f'{sum(len(p["indices"]) for p in parts)} triangles')


def prepare_vehicle(source, output, name, spec):
    folder = source / 'sketchfab' / name
    document = load(folder / 'scene.gltf')
    binary = (folder / document['buffers'][0]['uri']).read_bytes()
    parts = merge_by_material(flatten_scene(document, binary, matrix(spec['scale'], spec['yaw'])))
    size = recentre(parts)
    target = output / name
    images = convert_images(document, folder, target, spec['texture'])
    write_mesh(target, 'model', parts, document, images)
    shutil.copyfile(folder / 'license.txt', target / 'license.txt')
    print(f'{name}: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} m, '
          f'{sum(len(p["indices"]) for p in parts)} triangles')
    return size


def prepare_pallets(source, output):
    folder = source / 'sketchfab' / 'pallets'
    document = load(folder / 'scene.gltf')
    binary = (folder / document['buffers'][0]['uri']).read_bytes()
    target = output / 'pallets'
    images = convert_images(document, folder, target, 1024)
    # Specular-glossiness to metallic-roughness: diffuse colour, rough wood,
    # and the occlusion map (its red channel) for ambient occlusion.
    textures = document['textures']
    for material in document['materials']:
        gloss = material.pop('extensions')['KHR_materials_pbrSpecularGlossiness']
        material['pbrMetallicRoughness'] = dict(baseColorTexture=gloss['diffuseTexture'],
                                                metallicFactor=0.0, roughnessFactor=0.86)
        stem = Path(images[textures[gloss['diffuseTexture']['index']]['source']]['uri']).stem
        occlusion = stem.replace('_diffuse', '_occlusion')
        material['occlusionTexture'] = dict(index=next(
            i for i, t in enumerate(textures) if Path(images[t['source']]['uri']).stem == occlusion))
    document.pop('extensionsUsed', None)
    for stem, node_name in PALLETS.items():
        # The node keeps its offset in the lineup, which recentre() removes.
        parts = merge_by_material(flatten_node(document, binary, node_name, matrix()))
        size = recentre(parts)
        factor = np.array(PALLET_SIZE) / size
        for part in parts:
            part['POSITION'] = part['POSITION'] * factor
            normals = part['NORMAL'] / factor
            part['NORMAL'] = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        write_mesh(target, stem, parts, document, images)
        print(f'{stem}: {sum(len(p["indices"]) for p in parts)} triangles, scaled by {np.round(factor, 3)}')
    shutil.copyfile(folder / 'license.txt', target / 'license.txt')


def prepare_cartons(source, output):
    """Each carton of the set as its own model: footprint axis-aligned and centred, base at z = 0."""
    folder = source / 'sketchfab' / 'cardboard_boxes'
    document = load(folder / 'scene.gltf')
    binary = (folder / document['buffers'][0]['uri']).read_bytes()
    target = output / 'cartons'
    images = convert_images(document, folder, target, 2048)
    sizes = {}
    for name in CARTONS:
        parts = merge_by_material(flatten_node(document, binary, name, matrix()))
        # Turn the footprint onto the axes: the smallest bounding rectangle's angle.
        points = np.vstack([part['POSITION'][:, :2] for part in parts])
        angles = np.radians(np.arange(0, 90, 0.05))
        areas = [np.ptp(points @ [math.cos(a), math.sin(a)]) * np.ptp(points @ [-math.sin(a), math.cos(a)])
                 for a in angles]
        angle = angles[int(np.argmin(areas))]
        c, s = math.cos(-angle), math.sin(-angle)
        turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        for part in parts:
            part['POSITION'] = part['POSITION'] @ turn.T
            part['NORMAL'] = part['NORMAL'] @ turn.T
        size = recentre(parts)
        sizes[name] = [round(float(v), 4) for v in size]
        write_mesh(target, name, parts, document, images)
    (target / 'sizes.json').write_text(json.dumps(sizes, indent=1) + '\n')
    shutil.copyfile(folder / 'license.txt', target / 'license.txt')
    print('cartons:', sizes)


# ---------------------------------------------------------------- generated textures
def tileable_noise(rng, size, sigma):
    from scipy.ndimage import gaussian_filter
    field = gaussian_filter(rng.standard_normal((size, size)), sigma, mode='wrap')
    return (field - field.mean()) / (field.std() + 1e-12)


def save_rgb(path, rgb):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)).save(path, quality=92)


def generate_textures(output):
    """Tileable textures multiplied by a material colour: painted steel, floor paint, stripes."""
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(11)
    target = output / 'textures' / 'generated'
    size = 1024
    # Painted steel: faint blotches and fine scratches lighten the colour a little
    # and roughen the surface.
    blotch = tileable_noise(rng, size, 24) * 0.6 + tileable_noise(rng, size, 6) * 0.4
    scratches = np.zeros((size, size))
    for _ in range(260):
        x, y = rng.uniform(0, size, 2)
        angle, length = rng.uniform(0, math.pi), rng.uniform(8, 60)
        steps = np.linspace(0, length, int(length * 2))
        xs = ((x + np.cos(angle) * steps) % size).astype(int)
        ys = ((y + np.sin(angle) * steps) % size).astype(int)
        scratches[ys, xs] = rng.uniform(0.3, 1.0)
    scratches = gaussian_filter(scratches, 0.7, mode='wrap')
    grunge = np.clip(0.93 + 0.035 * blotch + 0.25 * scratches, 0, 1)
    save_rgb(target / 'painted_steel_diff.jpg', np.repeat(grunge[..., None], 3, 2))
    rough = np.clip(0.42 + 0.06 * blotch + 0.3 * scratches, 0.2, 0.9)
    save_rgb(target / 'painted_steel_arm.jpg', np.dstack([np.ones_like(rough), rough, np.zeros_like(rough)]))

    # Rack upright face: 90 mm wide, one 50 mm slot pitch per 1/8 of the
    # texture height (0.4 m), two columns of teardrop slots.
    width, height = 256, 1024
    v, u = np.mgrid[0:height, 0:width]
    u = (u + 0.5) / width * 0.09
    v = (v + 0.5) / height * 0.4
    face = np.ones((height, width))
    for column in (0.028, 0.062):
        local_v = (v % 0.05) - 0.025
        du = u - column
        # A slot is a 9 mm x 30 mm rounded rectangle, wider at the top.
        half_width = 0.0045 + 0.0018 * (local_v / 0.015)
        inside = (np.abs(local_v) < 0.015) & (np.abs(du) < np.maximum(half_width, 0.0025))
        face[inside] = 0.04
    face = gaussian_filter(face, 0.8)
    grain = gaussian_filter(rng.standard_normal((height, width)), 3, mode='wrap')
    face *= 0.96 + 0.02 * grain / grain.std()
    save_rgb(target / 'rack_upright_diff.jpg', np.repeat(face[..., None], 3, 2))

    # Floor paint: worn yellow; alpha (stored in a PNG) removes the worn spots so
    # the concrete shows through.
    wear = tileable_noise(rng, size, 14) * 0.7 + tileable_noise(rng, size, 3) * 0.3
    alpha = np.clip((wear + 1.55) * 2.5, 0, 1)
    shade = 0.92 + 0.06 * tileable_noise(rng, size, 30)
    rgba = np.dstack([np.repeat(shade[..., None], 3, 2), alpha])
    Image.fromarray(np.clip(rgba * 255 + 0.5, 0, 255).astype(np.uint8), 'RGBA').save(
        target / 'floor_paint_diff.png', optimize=True)

    # Hazard stripes: 45 degree yellow and black bands, 4 per texture, worn.
    y, x = np.mgrid[0:512, 0:512]
    band = (((x + y) // 64) % 2 == 0)
    yellow, black = np.array([0.95, 0.72, 0.05]), np.array([0.03, 0.03, 0.03])
    stripes = np.where(band[..., None], yellow, black)
    dirt = 0.9 + 0.08 * tileable_noise(rng, 512, 10)[..., None]
    save_rgb(target / 'hazard_stripes_diff.jpg', np.clip(stripes * dirt, 0, 1) ** (1 / 2.2))
    generate_signs(target)
    print('generated textures in', target)


def generate_signs(target):
    """An emergency exit sign and aisle signs AISLE 01 to AISLE 05 (rows of one image)."""
    from PIL import Image, ImageDraw, ImageFont
    sign = Image.new('RGB', (512, 192), (0, 132, 61))
    draw = ImageDraw.Draw(sign)
    draw.rectangle((6, 6, 505, 185), outline=(235, 245, 238), width=6)
    font = ImageFont.load_default(size=118)
    draw.text((50, 96), 'EXIT', font=font, fill=(245, 250, 246), anchor='lm', stroke_width=3,
              stroke_fill=(245, 250, 246))
    draw.polygon([(372, 76), (430, 76), (430, 52), (478, 96), (430, 140), (430, 116), (372, 116)],
                 fill=(245, 250, 246))
    sign.save(target / 'exit_sign.png', optimize=True)
    rows = 5
    aisles = Image.new('RGB', (1024, 256 * rows), (12, 52, 112))
    draw = ImageDraw.Draw(aisles)
    font = ImageFont.load_default(size=150)
    for row in range(rows):
        top = row * 256
        draw.rectangle((10, top + 10, 1013, top + 245), outline=(240, 240, 236), width=10)
        draw.rectangle((20, top + 205, 1003, top + 235), fill=(245, 184, 0))
        draw.text((512, top + 112), f'AISLE {row + 1:02d}', font=font, fill=(245, 245, 240), anchor='mm',
                  stroke_width=2, stroke_fill=(245, 245, 240))
    aisles.save(target / 'aisle_signs.png', optimize=True)


def brighten(path, gain):
    """Scale a colour map in linear light and save it in place."""
    from PIL import Image
    srgb = np.asarray(Image.open(path).convert('RGB')).astype(np.float64) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4) * gain
    srgb = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055)
    Image.fromarray(np.clip(srgb * 255 + 0.5, 0, 255).astype(np.uint8)).save(path, quality=92)


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: prepare_warehouse_assets.py DIR [OUTPUT]')
    source = Path(sys.argv[1]) / 'assets'
    output = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(sys.argv[1]) / 'prepared'
    output.mkdir(parents=True, exist_ok=True)
    for name, spec in PROPS.items():
        prepare_prop(source, output, name, spec)
    for name, spec in VEHICLES.items():
        prepare_vehicle(source, output, name, spec)
    prepare_pallets(source, output)
    prepare_cartons(source, output)
    for name in TEXTURES:
        shutil.copytree(source / 'textures' / name, output / 'textures' / name, dirs_exist_ok=True)
    brighten(output / 'textures/concrete_floor_worn_001/concrete_floor_worn_001_diff_2k.jpg', FLOOR_GAIN)
    generate_textures(output)
    shutil.copyfile(source / 'sources.json', output / 'sources.json')
    notices = Path(__file__).resolve().parents[2] / 'rsc/warehouse'
    for notice in ['LICENSE-CC0-1.0.txt', 'ATTRIBUTION.md']:
        if (notices / notice).exists() and (notices / notice).resolve() != (output / notice).resolve():
            shutil.copyfile(notices / notice, output / notice)
    generate_warehouse_rscene.generate(output)
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'
             and not p.name.endswith('.lods') and not p.name.endswith('.rmpc')]
    (output / 'manifest.json').write_text(json.dumps(dict(files=files), indent=2) + '\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Prepare the armored vehicles of the tank survival example (raisim2Lib/wasm) from their Sketchfab glTF
downloads (download_tank_assets.py).

Each model is turned to face +x with z up, scaled to the real vehicle's width, set on the ground and
centred on its hull, and split into the three parts the game moves: the hull, the turret (turning
about z at the turret ring) and the gun (elevating about y at its trunnions). Connected pieces of
geometry stay whole: the gun is the barrel (the longest thin piece) and the pieces strung along it;
the turret is the pieces resting above the hull deck inside the turret ring's footprint. Textures are
reduced to 1K. Each part is written as a glTF file sharing its vehicle's textures, with vehicle.json
giving the pivots and sizes the game builds the vehicle's URDF from.

Requires numpy and Pillow.
"""
import argparse
import hashlib
import io
import json
import math
import struct
from pathlib import Path

import numpy as np

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)

# Real widths (m) set each model's scale. The turret footprint is a box in the normalized frame
# (x forward, y left, metres from the hull centre) and deck the height of the turret ring above the
# ground; pieces lower than it stay on the hull.
VEHICLES = {
    'leopard_2a4': dict(source='leopard', forward='+x', width=3.75, gun=dict(x=1.0, y=0.0, z=1.9, radius=0.15),
                        turret=dict(x=(-3.4, 2.9), y=(-1.62, 1.62)), deck=1.48, ring=0.2),
    't72b3': dict(source='t72', forward='+y', width=3.6, gun=dict(x=0.95, y=0.0, z=1.6, radius=0.15),
                  turret=dict(x=(-2.7, 2.0), y=(-1.75, 1.75)), deck=1.4, ring=-0.4),
    't90am': dict(source='t90', forward='+x', width=3.78, gun=dict(x=0.85, y=0.0, z=1.7, radius=0.15),
                  turret=dict(x=(-2.2, 2.3), y=(-1.7, 1.7)), deck=1.4, ring=0.0),
    'bmd2': dict(source='bmd2', forward='-x', width=2.63, gun=dict(x=1.1, y=0.0, z=1.62, radius=0.08),
                 turret=dict(x=(-0.7, 1.1), y=(-0.8, 0.8)), deck=1.5, ring=0.1),
    'sprut_sd': dict(source='sprut', forward='-y', width=3.15, gun=dict(x=0.9, y=0.0, z=1.9, radius=0.15),
                     turret=dict(x=(-1.9, 1.9), y=(-1.5, 1.5)), deck=1.4, ring=-0.3),
    'msta_s': dict(source='msta', forward='+y', width=3.58, gun=dict(x=1.9, y=0.0, z=2.38, radius=0.22, pitch=7.06),
                   turret=dict(x=(-3.7, 2.0), y=(-1.75, 1.75)), deck=1.25, ring=-0.3),
    't14_armata': dict(source='armata', forward='-y', width=3.5, gun=dict(x=0.95, y=0.0, z=2.0, radius=0.15),
                       turret=dict(x=(-2.6, 2.4), y=(-1.45, 1.45)), deck=1.62, ring=-0.5),
}


def node_matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    m = np.eye(4)
    m[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    m[:3, :3] *= np.array(node.get('scale', [1, 1, 1]))
    m[:3, 3] = node.get('translation', [0, 0, 0])
    return m


def read_glb(path):
    data = Path(path).read_bytes()
    if data[:4] != b'glTF':
        raise ValueError(f'{path} is not a binary glTF file')
    length = struct.unpack('<I', data[12:16])[0]
    document = json.loads(data[20:20 + length])
    binary = data[20 + length + 8:]
    return document, binary


def accessor(document, binary, index):
    item = document['accessors'][index]
    view = document['bufferViews'][item['bufferView']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[item['type']]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[item['componentType']]
    size = np.dtype(dtype).itemsize
    stride = view.get('byteStride', width * size)
    start = view.get('byteOffset', 0) + item.get('byteOffset', 0)
    values = np.ndarray((item['count'], width), dtype, binary, start, (stride, size)).astype(np.float64)
    if item.get('normalized') and dtype != np.float32:
        values /= np.iinfo(dtype).max
    return values


def flatten(document, binary):
    """Every primitive with its node transforms baked in, in a z-up frame."""
    primitives = []

    def visit(index, parent):
        node = document['nodes'][index]
        world = parent @ node_matrix(node)
        if 'mesh' in node:
            linear = world[:3, :3]
            normal_matrix = np.linalg.inv(linear).T
            mirrored = np.linalg.det(linear) < 0
            for primitive in document['meshes'][node['mesh']]['primitives']:
                attributes = primitive['attributes']
                positions = accessor(document, binary, attributes['POSITION']) @ linear.T + world[:3, 3]
                normals = (accessor(document, binary, attributes['NORMAL']) @ normal_matrix.T
                           if 'NORMAL' in attributes else np.zeros_like(positions))
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                uvs = (accessor(document, binary, attributes['TEXCOORD_0']) if 'TEXCOORD_0' in attributes
                       else np.zeros((len(positions), 2)))
                triangles = accessor(document, binary, primitive['indices']).reshape(-1, 3).astype(np.int64)
                if mirrored:
                    triangles = triangles[:, ::-1]
                primitives.append(dict(material=primitive.get('material'), positions=positions, normals=normals,
                                       uvs=uvs, triangles=triangles))
        for child in node.get('children', []):
            visit(child, world)

    for root in document['scenes'][document.get('scene', 0)]['nodes']:
        visit(root, Y_UP_TO_Z_UP)
    return primitives


def pieces(primitive):
    """Label of the connected piece of each triangle (vertices at the same position are joined)."""
    key = np.round(primitive['positions'] / (1e-5 * np.ptp(primitive['positions'], axis=0).max())).astype(np.int64)
    _, unique = np.unique(key, axis=0, return_inverse=True)
    unique = unique.reshape(-1)
    parent = np.arange(unique.max() + 1)

    def find(a):
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for a, b, c in unique[primitive['triangles']]:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[rc] = ra
    return np.array([find(u) for u in unique[primitive['triangles'][:, 0]]])


def normalize(primitives, forward, width):
    """Turns the model to face +x with z up, scales it to width and sets it on the ground, centred on the
    lower half of its hull (the tracks and hull sides, below the turret and gun)."""
    for primitive in primitives:
        primitive['pieces'] = pieces(primitive)
    yaw = {'+x': 0.0, '+y': math.pi / 2, '-x': math.pi, '-y': -math.pi / 2}[forward]
    c, s = math.cos(-yaw), math.sin(-yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for primitive in primitives:
        primitive['positions'] = primitive['positions'] @ rotation.T
        primitive['normals'] = primitive['normals'] @ rotation.T
    everything = np.concatenate([p['positions'] for p in primitives])
    lo, hi = everything.min(0), everything.max(0)
    scale = width / (hi[1] - lo[1])
    lower = everything[everything[:, 2] < lo[2] + 0.4 * (hi[2] - lo[2])]
    offset = np.array([0.5 * (lower[:, 0].min() + lower[:, 0].max()), 0.5 * (lo[1] + hi[1]), lo[2]])
    for primitive in primitives:
        primitive['positions'] = (primitive['positions'] - offset) * scale


def split(primitives, gun, turret, deck):
    """Part of each triangle: 0 hull, 1 turret, 2 gun. The gun is the pieces strung along its axis ahead of
    its trunnions (and, of a piece welded to the turret, the triangles on that axis); the turret is the
    pieces resting at or above the deck inside its footprint. A gun modelled elevated is turned level
    about its trunnions."""
    slope = math.tan(math.radians(gun.get('pitch', 0.0)))
    axis_z = lambda x: gun['z'] + (x - gun['x']) * slope
    for primitive in primitives:
        part = np.zeros(len(primitive['triangles']), np.int64)
        centres = primitive['positions'][primitive['triangles']].mean(axis=1)
        for label in np.unique(primitive['pieces']):
            mask = primitive['pieces'] == label
            q = primitive['positions'][primitive['triangles'][mask].reshape(-1)]
            lo, hi, centre = q.min(0), q.max(0), q.mean(0)
            on_barrel = (lo[0] >= gun['x'] - 0.05 and abs(centre[1] - gun['y']) < 2.5 * gun['radius'] and
                         abs(centre[2] - axis_z(centre[0])) < 2.5 * gun['radius'] and
                         hi[2] - lo[2] < 6 * gun['radius'] + (hi[0] - lo[0]) * abs(slope))
            in_ring = (lo[2] >= deck - 0.08 and turret['x'][0] <= centre[0] <= turret['x'][1] and
                       turret['y'][0] <= centre[1] <= turret['y'][1])
            part[mask] = 2 if on_barrel else 1 if in_ring else 0
        c = centres
        on_axis = ((c[:, 0] > gun['x'] + 0.3) & (np.abs(c[:, 1] - gun['y']) < 1.3 * gun['radius']) &
                   (np.abs(c[:, 2] - axis_z(c[:, 0])) < 1.3 * gun['radius']))
        part[on_axis] = 2
        primitive['part'] = part
    if slope:
        angle = math.radians(gun['pitch'])
        c, s = math.cos(angle), math.sin(angle)
        level = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        pivot = np.array([gun['x'], gun['y'], gun['z']])
        for primitive in primitives:
            moved = np.unique(primitive['triangles'][primitive['part'] == 2].reshape(-1))
            primitive['positions'][moved] = (primitive['positions'][moved] - pivot) @ level.T + pivot
            primitive['normals'][moved] = primitive['normals'][moved] @ level.T


def texture_bytes(document, binary, image_index, folder):
    image = document['images'][image_index]
    if 'bufferView' in image:
        view = document['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        return binary[start:start + view['byteLength']]
    return (folder / image['uri']).read_bytes()


def write_textures(document, binary, folder, output):
    """Each image at 1K: JPEG, or PNG where it has transparency. Returns the file of each image."""
    from PIL import Image
    files = {}
    for index in range(len(document.get('images', []))):
        picture = Image.open(io.BytesIO(texture_bytes(document, binary, index, folder)))
        picture.load()
        picture.thumbnail((1024, 1024), Image.LANCZOS)
        transparent = picture.mode in ('RGBA', 'LA') and picture.getchannel('A').getextrema()[0] < 250
        name = f'texture_{index}.' + ('png' if transparent else 'jpg')
        target = output / 'textures' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if transparent:
            picture.save(target, optimize=True)
        else:
            picture.convert('RGB').save(target, quality=88, optimize=True)
        files[index] = 'textures/' + name
    return files


def write_part(document, primitives, part, images, path):
    """One part of the vehicle as a glTF file with one primitive per material."""
    gltf = dict(asset=dict(version='2.0', generator='raisim2Lib prepare_tank_assets.py'), scene=0,
                scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name=path.stem)], meshes=[dict(primitives=[])],
                accessors=[], bufferViews=[], materials=[], textures=[], images=[], samplers=[dict()])
    blob = bytearray()
    texture_of = {}

    def view(data, target):
        while len(blob) % 4:
            blob.append(0)
        gltf['bufferViews'].append(dict(buffer=0, byteOffset=len(blob), byteLength=len(data), target=target))
        blob.extend(data)
        return len(gltf['bufferViews']) - 1

    def add_accessor(values, kind, component, target, bounds=False):
        data = np.ascontiguousarray(values).tobytes()
        entry = dict(bufferView=view(data, target), componentType=component, count=len(values), type=kind)
        if bounds:
            entry['min'] = values.min(0).tolist()
            entry['max'] = values.max(0).tolist()
        gltf['accessors'].append(entry)
        return len(gltf['accessors']) - 1

    def texture(source):
        index = source['index']
        image = document['textures'][index]['source']
        if image not in texture_of:
            gltf['images'].append(dict(uri=images[image]))
            gltf['textures'].append(dict(source=len(gltf['images']) - 1, sampler=0))
            texture_of[image] = len(gltf['textures']) - 1
        return dict(source, index=texture_of[image])

    materials = {}
    for primitive in primitives:
        selected = primitive['triangles'][primitive['part'] == part]
        if not len(selected):
            continue
        used, triangles = np.unique(selected.reshape(-1), return_inverse=True)
        material = primitive['material']
        if material not in materials:
            source = dict(document['materials'][material]) if material is not None else dict(name='default')
            for key in ('normalTexture', 'occlusionTexture', 'emissiveTexture'):
                if key in source:
                    source[key] = texture(source[key])
            pbr = dict(source.get('pbrMetallicRoughness', {}))
            for key in ('baseColorTexture', 'metallicRoughnessTexture'):
                if key in pbr:
                    pbr[key] = texture(pbr[key])
            source['pbrMetallicRoughness'] = pbr
            source.pop('extensions', None)
            gltf['materials'].append(source)
            materials[material] = len(gltf['materials']) - 1
        attributes = dict(
            POSITION=add_accessor(primitive['positions'][used].astype(np.float32), 'VEC3', 5126, 34962, True),
            NORMAL=add_accessor(primitive['normals'][used].astype(np.float32), 'VEC3', 5126, 34962),
            TEXCOORD_0=add_accessor(primitive['uvs'][used].astype(np.float32), 'VEC2', 5126, 34962))
        indices = add_accessor(triangles.reshape(-1).astype(np.uint32), 'SCALAR', 5125, 34963)
        gltf['meshes'][0]['primitives'].append(dict(attributes=attributes, indices=indices,
                                                    material=materials[material]))
    if not gltf['meshes'][0]['primitives']:
        raise ValueError(f'{path.name} is empty')
    if not gltf['textures']:
        for key in ('textures', 'images', 'samplers'):
            gltf.pop(key)
    binary_name = path.with_suffix('.bin').name
    gltf['buffers'] = [dict(uri=binary_name, byteLength=len(blob))]
    path.with_suffix('.bin').write_bytes(bytes(blob))
    path.write_text(json.dumps(gltf, separators=(',', ':')))


def bounds(primitives, part):
    points = [p['positions'][p['triangles'][p['part'] == part].reshape(-1)] for p in primitives]
    points = np.concatenate([q for q in points if len(q)])
    return points.min(0), points.max(0)


def prepare(name, spec, downloads, output):
    document, binary = read_glb(downloads / f"{spec['source']}.glb")
    primitives = flatten(document, binary)
    normalize(primitives, spec['forward'], spec['width'])
    split(primitives, spec['gun'], spec['turret'], spec['deck'])
    gun = spec['gun']
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    images = write_textures(document, binary, downloads, folder)
    parts = {}
    for part, part_name in enumerate(('hull', 'turret', 'gun')):
        write_part(document, primitives, part, images, folder / f'{part_name}.gltf')
        lo, hi = bounds(primitives, part)
        parts[part_name] = dict(min=lo.round(4).tolist(), max=hi.round(4).tolist())
    description = dict(
        hull=parts['hull'], turret=parts['turret'], gun=parts['gun'],
        turretPivot=[round(spec['ring'], 4), 0.0, round(spec['deck'], 4)],
        gunPivot=[round(gun['x'], 4), round(gun['y'], 4), round(gun['z'], 4)], gunRadius=round(gun['radius'], 4),
        muzzle=round(parts['gun']['max'][0], 4))
    (folder / 'vehicle.json').write_text(json.dumps(description, indent=2) + '\n')
    return description


# Poly Haven props (prepare_offroad_venue.prepare_polyhaven bakes them z-up): the game gives barrels and
# crates bodies of its own, and the pickups and plants need none; rocks, stumps and barriers are solid.
PROPS = {
    'barrel_03': dict(colliders=[]),
    'wooden_military_crate': dict(colliders=[], keep=0.5),
    'old_military_crate': dict(colliders=[]),
    'ammo_box': dict(colliders=[]),
    'metal_jerrycan_green': dict(colliders=[]),
    'old_tyre': dict(colliders=[]),
    'shrub_02': dict(colliders=[], keep=0.4),
    'shrub_04': dict(colliders=[], keep=0.4),
    'wild_rooibos_bush': dict(colliders=[], keep=0.3),
    'concrete_road_barrier': dict(colliders='bounds', keep=0.2),
    'tree_stump_01': dict(colliders='hull', keep=0.3),
    'namaqualand_boulder_04': dict(colliders='hull', keep=0.12),
    'namaqualand_boulder_05': dict(colliders='hull', keep=0.12),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True, help='Download folder of download_tank_assets.py')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'rsc' / 'tank_survival')
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()
    for name, spec in VEHICLES.items():
        if args.only and name not in args.only:
            continue
        description = prepare(name, spec, args.cache / 'vehicles', args.output / 'vehicles')
        print(name, json.dumps(description))
    from prepare_offroad_venue import prepare_polyhaven
    for name, spec in PROPS.items():
        if args.only and name not in args.only:
            continue
        prepare_polyhaven(args.cache, args.output / 'props', name, spec)
    if not args.only:
        grounds = args.output / 'ground'
        grounds.mkdir(parents=True, exist_ok=True)
        for image in sorted((args.cache / 'grounds').glob('*.jpg')):
            (grounds / image.name).write_bytes(image.read_bytes())


if __name__ == '__main__':
    main()

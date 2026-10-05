#!/usr/bin/env python3
"""Prepare the rayrai city's runtime assets from download_city_assets.py's DIR/assets.

Usage: prepare_city_assets.py DIR [OUTPUT]   (OUTPUT defaults to DIR/prepared)

* Facade kits: every module the city uses becomes its own Z-up glTF file in
  <kit>/modules, sharing the kit's buffer and textures.
* Props: one variant of each Poly Haven lineup, recentred, rotated to Z-up,
  with a .rasset of simple colliders.
* Cars: the Sketchfab models are centred and rotated to Z-up, their PNG
  textures reduced to 1K (JPEG unless blended), and two extra paint colours
  are written as glTF files that share each car's buffer and textures.
* Traffic cone: BlendKit's WebP textures become JPEG, and the Z-up rotation
  is baked into the vertices, because RaiSim reads the dynamic cone's
  collision hull from the same file.
* Sky: the sun disk of the HDR is clamped. The scene's directional light
  provides the sun, with shadows, so image-based lighting must not add it a
  second time to the shadowed side of every building. The sky is also turned
  by 180 degrees so that the sun lights the street view from behind.

Finally generate_city_rscene.py lays out the city and writes the scene.
Requires NumPy, SciPy and Pillow.
"""
import copy
import hashlib
import io
import json
import math
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

import generate_city_rscene

Y_TO_Z = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]  # glTF xyzw: turns the +y up axis to +z
KITS = {'apartments': ('modular_urban_apartments_facade', '2k'),
        'factory': ('modular_factory_facade', '2k')}
# Poly Haven lineup nodes of one variant, the lineup offset of that variant
# (glTF metres, Y-up), node overrides, and colliders in the Z-up model frame.
PROPS = {
    'street_lamp_01': dict(colliders=[('cylinder', dict(radius=0.1, height=3.87), (0, 0, 1.935))]),
    'fire_hydrant': dict(nodes=['fire_hydrant', 'fire_hydrant_cap_01', 'fire_hydrant_cap_02',
                                'fire_hydrant_cap_03', 'fire_hydrant_chain'], offset=(-0.3, 0, 0),
                         colliders=[('cylinder', dict(radius=0.14, height=0.8), (0, 0, 0.4))]),
    'metal_trash_can': dict(nodes=['metal_trash_can', 'metal_trash_can_lid', 'metal_trash_can_handle_left',
                                   'metal_trash_can_handle_right'], offset=(0.5, 0, 0),
                            # The lineup leans the lid against the can; put it on top.
                            overrides={'metal_trash_can_lid': dict(translation=[0.5, 0.906, 0],
                                                                   rotation=[0, 0, 0, 1])},
                            colliders=[('cylinder', dict(radius=0.3, height=0.95), (0, 0, 0.475))]),
    'modular_street_seating': dict(nodes=['crossbar', 'legs_double', 'legs_single', 'suspended_support_01',
                                          'back_support_r', 'back_support_l', 'arm_rest_01', 'arm_rest_02',
                                          'seat', 'seat_back'], offset=(-1.16, 0, 0),
                                   colliders=[('box', dict(size=(2.42, 0.62, 0.48)), (0, 0, 0.24)),
                                              ('box', dict(size=(1.7, 0.2, 0.42)), (0, 0.3, 0.66))]),
    'water_manhole_cover': dict(colliders=[]),
    'utility_box_02': dict(colliders=[('box', dict(size=(0.92, 0.44, 1.12)), (0, 0, 0.56))]),
    'concrete_road_barrier': dict(colliders=[('box', dict(size=(1.55, 0.64, 0.82)), (0, -0.025, 0.41))]),
}
TEXTURES = ['asphalt_02', 'concrete_pavement_02', 'concrete_floor_01', 'bitumen']
# Sketchfab cars and extra linear-RGB paint colours of their body material; each
# colour is a glTF file that shares the car's buffer and textures.
CARS = {'fairheaven_lt80': [(0.55, 0.56, 0.58), (0.42, 0.35, 0.25)],
        'fairheaven_sw84': [(0.72, 0.72, 0.7), (0.02, 0.035, 0.11)],
        'kiri86': [(0.015, 0.015, 0.017), (0.24, 0.29, 0.35)],
        'canyon75_taxi': []}
SKY = 'kloofendal_48d_partly_cloudy_puresky_4k.hdr'
SUN_CLAMP = 24.0  # HDR radiance above which the sun disk and its glare are clamped


def load(path):
    return json.loads(Path(path).read_text())


def dump(document, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, separators=(',', ':')) + '\n')


def extract(document, mesh_indices):
    """A copy that keeps only the given meshes and the accessors they read."""
    result = copy.deepcopy(document)
    accessors, views = {}, {}

    def accessor(index):
        if index not in accessors:
            item = copy.deepcopy(document['accessors'][index])
            view = item['bufferView']
            if view not in views:
                views[view] = len(views)
            item['bufferView'] = views[view]
            accessors[index] = (len(accessors), item)
        return accessors[index][0]

    meshes = []
    for mesh_index in mesh_indices:
        mesh = copy.deepcopy(document['meshes'][mesh_index])
        for primitive in mesh['primitives']:
            primitive['indices'] = accessor(primitive['indices'])
            primitive['attributes'] = {k: accessor(v) for k, v in primitive['attributes'].items()}
        meshes.append(mesh)
    result['meshes'] = meshes
    result['accessors'] = [item for _, item in sorted(accessors.values(), key=lambda a: a[0])]
    result['bufferViews'] = [document['bufferViews'][old] for old, _ in sorted(views.items(), key=lambda v: v[1])]
    return result


def read_accessor(document, binary, index):
    item = document['accessors'][index]
    view = document['bufferViews'][item['bufferView']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[item['type']]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[item['componentType']]
    'byteStride' not in view or sys.exit('interleaved buffers are not supported')
    offset = view.get('byteOffset', 0) + item.get('byteOffset', 0)
    return np.frombuffer(binary, dtype, item['count'] * width, offset).reshape(item['count'], width)


def simplify_planar(positions, normals, uvs, triangles):
    """Replace flat, gridded regions of a mesh by as few rectangles as cover them exactly.

    The kits model each flat wall as a dense grid (a plain 3 m panel has 4,608
    triangles). Triangles are grouped by plane; a plane whose vertex normals
    match it, whose texture coordinates are an affine function of position and
    whose triangles tile whole cells of the grid of their vertex coordinates is
    rebuilt from maximal rectangles of those cells. Anything else is kept.
    Returns new (positions, normals, uvs, triangles), or None if nothing changed.
    """
    corners = positions[triangles]
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(face, axis=1)
    keep = np.ones(len(triangles), bool)
    unit = face / np.maximum(area, 1e-12)[:, None]
    key = np.round(np.c_[unit, (unit * corners[:, 0]).sum(1)], 4)
    planes, plane_of = np.unique(key, axis=0, return_inverse=True)
    new_p, new_n, new_uv, new_i = [], [], [], []
    for plane_index, plane in enumerate(planes):
        members = np.flatnonzero(plane_of.ravel() == plane_index)
        if len(members) < 16:
            continue
        normal, distance = plane[:3] / np.linalg.norm(plane[:3]), plane[3]
        used = np.unique(triangles[members])
        if np.abs(normals[used] @ normal - 1).max() > 1e-3:
            continue
        fit = np.c_[positions[used], np.ones(len(used))]
        uv_map = np.linalg.lstsq(fit, uvs[used], rcond=None)[0]
        if np.abs(fit @ uv_map - uvs[used]).max() > 1e-5:
            continue
        axis_u = np.cross(normal, [0, 0, 1] if abs(normal[2]) < 0.9 else [1, 0, 0])
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        flat = np.c_[positions @ axis_u, positions @ axis_v]
        a_values = np.unique(np.round(flat[used, 0], 5))
        b_values = np.unique(np.round(flat[used, 1], 5))
        # A sample point off the cell's diagonals lies in exactly one covering triangle.
        centres_a = a_values[:-1] + 0.37 * np.diff(a_values)
        centres_b = b_values[:-1] + 0.29 * np.diff(b_values)
        grid_a, grid_b = np.meshgrid(centres_a, centres_b, indexing='ij')
        points = np.c_[grid_a.ravel(), grid_b.ravel()]
        covered = np.zeros(len(points), np.int32)
        for chunk in range(0, len(members), 256):
            t = flat[triangles[members[chunk:chunk + 256]]]          # (k, 3, 2)
            edge0, edge1 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
            det = edge0[:, 0] * edge1[:, 1] - edge0[:, 1] * edge1[:, 0]
            rel = points[None] - t[:, None, 0]                       # (k, cells, 2)
            s = (rel[..., 0] * edge1[:, None, 1] - rel[..., 1] * edge1[:, None, 0]) / det[:, None]
            r = (edge0[:, None, 0] * rel[..., 1] - edge0[:, None, 1] * rel[..., 0]) / det[:, None]
            covered += ((s >= 0) & (r >= 0) & (s + r <= 1)).sum(0)
        cells = covered.reshape(len(centres_a), len(centres_b))
        cell_area = np.outer(np.diff(a_values), np.diff(b_values))
        triangle_area = area[members].sum() / 2
        if cells.max() > 1 or abs((cell_area * cells).sum() - triangle_area) > 1e-5 * triangle_area:
            continue  # the triangles do not tile whole grid cells
        open_runs, rectangles = {}, []
        for j in range(cells.shape[1] + 1):
            runs = set()
            if j < cells.shape[1]:
                column = np.r_[0, cells[:, j], 0]
                edges = np.flatnonzero(np.diff(column))
                runs = set(zip(edges[::2], edges[1::2]))
            for run in list(open_runs):
                if run not in runs:
                    rectangles.append((*run, open_runs.pop(run), j))
            for run in runs:
                open_runs.setdefault(run, j)
        for a0, a1, b0, b1 in rectangles:
            base = len(new_p)
            for a, b in ((a0, b0), (a1, b0), (a1, b1), (a0, b1)):
                point = a_values[a] * axis_u + b_values[b] * axis_v + distance * normal
                new_p.append(point)
                new_n.append(normal)
                new_uv.append(np.r_[point, 1] @ uv_map)
            new_i += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        keep[members] = False
    if keep.all():
        return None
    kept = triangles[keep]
    vertices, remap = np.unique(kept, return_inverse=True)
    offset = len(vertices)
    out_p = np.vstack([positions[vertices]] + ([np.array(new_p)] if new_p else []))
    out_n = np.vstack([normals[vertices]] + ([np.array(new_n)] if new_n else []))
    out_uv = np.vstack([uvs[vertices]] + ([np.array(new_uv)] if new_uv else []))
    out_i = np.vstack([remap.reshape(-1, 3)] + ([np.array(new_i) + offset] if new_i else []))
    return (out_p.astype(np.float32), out_n.astype(np.float32), out_uv.astype(np.float32),
            out_i.astype(np.uint32))


def simplify_module(module, binary, name, target):
    """Rewrite the planar parts of a module into its own buffer; others keep the kit buffer."""
    data = bytearray()
    before = after = 0
    for primitive in module['meshes'][0]['primitives']:
        attributes = primitive['attributes']
        if set(attributes) != {'POSITION', 'NORMAL', 'TEXCOORD_0'}:
            continue
        arrays = [read_accessor(module, binary, attributes[k]).astype(np.float64)
                  for k in ('POSITION', 'NORMAL', 'TEXCOORD_0')]
        triangles = read_accessor(module, binary, primitive['indices']).reshape(-1, 3).astype(np.int64)
        before += len(triangles)
        result = simplify_planar(*arrays, triangles)
        if result is None:
            after += len(triangles)
            continue
        after += len(result[3])
        indices = {}
        for key, array, target_kind in zip(('POSITION', 'NORMAL', 'TEXCOORD_0', 'indices'), result,
                                           (34962, 34962, 34962, 34963)):
            while len(data) % 4:
                data.append(0)
            module['bufferViews'].append(dict(buffer=1, byteOffset=len(data), byteLength=array.nbytes,
                                              target=target_kind))
            data.extend(array.tobytes())
            accessor = dict(bufferView=len(module['bufferViews']) - 1,
                            componentType=5125 if key == 'indices' else 5126, count=int(array.size if key == 'indices' else len(array)),
                            type='SCALAR' if key == 'indices' else 'VEC%d' % array.shape[1])
            if key == 'POSITION':
                accessor.update(min=array.min(0).tolist(), max=array.max(0).tolist())
            module['accessors'].append(accessor)
            indices[key] = len(module['accessors']) - 1
        primitive['indices'] = indices.pop('indices')
        primitive['attributes'] = indices
    if data:
        module['buffers'].append(dict(uri=f'{name}.bin', byteLength=len(data)))
        (target / f'{name}.bin').write_bytes(bytes(data))
    return before, after


def split_kit(source, output, folder, modules):
    stem, resolution = KITS[folder]
    document = load(source / stem / f'{stem}_{resolution}.gltf')
    target = output / folder
    (target / 'textures').mkdir(parents=True, exist_ok=True)
    (len(document['buffers']) == 1) or sys.exit('unexpected buffer layout in ' + stem)
    shutil.copyfile(source / stem / document['buffers'][0]['uri'], target / f'{folder}.bin')
    for image in document['images']:
        shutil.copyfile(source / stem / image['uri'], target / image['uri'])
    binary = (target / f'{folder}.bin').read_bytes()
    nodes = {node['name']: node for node in document['nodes']}
    before = after = 0
    (target / 'modules').mkdir(exist_ok=True)
    for name in modules:
        module = extract(document, [nodes[name]['mesh']])
        # Modules keep their kit pivot: x in [-w, 0], the outer side towards -y after rotation.
        module.update(nodes=[dict(name=name, mesh=0, rotation=Y_TO_Z)], scenes=[dict(nodes=[0])], scene=0)
        counts = simplify_module(module, binary, name, target / 'modules')
        before, after = before + counts[0], after + counts[1]
        module['buffers'] = [dict(uri=f'../{folder}.bin', byteLength=document['buffers'][0]['byteLength'])] + \
            module['buffers'][1:]
        for image in module['images']:
            image['uri'] = '../' + image['uri']
        dump(module, target / 'modules' / f'{name}.gltf')
    print(f'{folder}: {len(modules)} modules, flat walls simplified from {before} to {after} triangles')


def write_rasset(path, visual, colliders):
    lines = ['<rasset version="1">', f'  <visual file="{visual}"/>']
    for kind, size, position in colliders:
        attributes = ' '.join(f'{key}="{value}"' if not isinstance(value, tuple)
                              else f'{key}="{" ".join(map(str, value))}"' for key, value in size.items())
        lines.append(f'  <collision type="{kind}" {attributes} position="{" ".join(map(str, position))}"/>')
    lines.append('</rasset>')
    path.write_text('\n'.join(lines) + '\n')


def prepare_prop(source, output, name, spec):
    document = load(source / name / f'{name}_1k.gltf')
    keep = spec.get('nodes') or [node['name'] for node in document['nodes'] if 'mesh' in node]
    offset = spec.get('offset', (0, 0, 0))
    by_name = {node['name']: node for node in document['nodes']}
    prop = extract(document, [by_name[n]['mesh'] for n in keep])
    children = []
    for i, node_name in enumerate(keep):
        node = {k: v for k, v in by_name[node_name].items() if k != 'children'}
        node.update(spec.get('overrides', {}).get(node_name, {}), mesh=i)
        node['translation'] = [a - b for a, b in zip(node.get('translation', [0, 0, 0]), offset)]
        children.append(node)
    prop.update(nodes=[dict(name=name, rotation=Y_TO_Z, children=list(range(1, len(keep) + 1)))] + children,
                scenes=[dict(nodes=[0])], scene=0)
    target = output / 'props' / name
    target.mkdir(parents=True, exist_ok=True)
    for buffer in prop['buffers']:
        shutil.copyfile(source / name / buffer['uri'], target / buffer['uri'])
    for image in prop['images']:
        (target / image['uri']).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name / image['uri'], target / image['uri'])
    dump(prop, target / 'model.gltf')
    if spec['colliders']:
        write_rasset(target / 'model.rasset', 'model.gltf', spec['colliders'])


def node_matrix(node):
    if 'matrix' in node:
        return np.array(node['matrix'], float).reshape(4, 4).T
    x, y, z, w = node.get('rotation', [0, 0, 0, 1])
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                               [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                               [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix[:3, :3] *= np.array(node.get('scale', [1, 1, 1]))
    matrix[:3, 3] = node.get('translation', [0, 0, 0])
    return matrix


def read_float_accessor(document, binary, index):
    item = document['accessors'][index]
    view = document['bufferViews'][item['bufferView']]
    width = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[item['type']]
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[item['componentType']]
    size = np.dtype(dtype).itemsize
    stride = view.get('byteStride', width * size)
    start = view.get('byteOffset', 0) + item.get('byteOffset', 0)
    return np.ndarray((item['count'], width), dtype, binary, start, (stride, size)).astype(np.float64)


def flatten_scene(document, binary, root):
    """Every mesh primitive with its node transforms (and root) baked into the vertices."""
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
                read = lambda key: read_float_accessor(document, binary, attributes[key])
                positions = read('POSITION') @ linear.T + world[:3, 3]
                normals = read('NORMAL') @ normal_matrix.T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                part = dict(material=primitive.get('material'), POSITION=positions, NORMAL=normals,
                            TEXCOORD_0=read('TEXCOORD_0'))
                triangles = read_float_accessor(document, binary, primitive['indices']).reshape(-1, 3)
                # A mirroring transform flips the winding, which must stay counter-clockwise.
                part['indices'] = (triangles[:, ::-1] if mirrored else triangles).astype(np.uint32)
                primitives.append(part)
        for child in node.get('children', []):
            visit(child, world)

    for scene_root in document['scenes'][document.get('scene', 0)]['nodes']:
        visit(scene_root, root)
    return primitives


def prepare_car(source, output, name, paints):
    """A Sketchfab car as one Z-up mesh: front at +x, centred, wheels at z = 0.

    The models nest FBX-converted nodes with mirroring matrices; their
    transforms are baked into the vertices so that the result does not depend
    on how an importer evaluates them, and the parts are merged into one
    primitive per material. Tangents are left to the importer. The body uses
    a tiled metallic-flake texture under a clear coat; it becomes a plain
    dielectric car paint.
    """
    from PIL import Image
    folder = source / 'cars' / name
    document = load(folder / 'scene.gltf')
    target = output / 'cars' / name
    (target / 'textures').mkdir(parents=True, exist_ok=True)
    binary = (folder / document['buffers'][0]['uri']).read_bytes()
    turn = np.eye(4)
    turn[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # Y-up to Z-up
    parts = flatten_scene(document, binary, turn)
    # One primitive per material: a car is drawn as a regular visual, one draw per primitive.
    merged = {}
    for part in parts:
        group = merged.setdefault(part['material'], dict(material=part['material'], POSITION=[], NORMAL=[],
                                                         TEXCOORD_0=[], indices=[], count=0))
        for key in ('POSITION', 'NORMAL', 'TEXCOORD_0'):
            group[key].append(part[key])
        group['indices'].append(part['indices'] + group['count'])
        group['count'] += len(part['POSITION'])
    parts = [dict(material=g['material'], indices=np.vstack(g['indices']),
                  **{key: np.vstack(g[key]) for key in ('POSITION', 'NORMAL', 'TEXCOORD_0')})
             for g in merged.values()]
    points = np.vstack([part['POSITION'] for part in parts])
    low, high = points.min(0), points.max(0)
    shift = np.array([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
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
        attributes = {}
        for key, kind in (('POSITION', 'VEC3'), ('NORMAL', 'VEC3'), ('TEXCOORD_0', 'VEC2')):
            array = part[key] + shift if key == 'POSITION' else part[key]
            attributes[key] = add(array.astype(np.float32), kind, 34962, key == 'POSITION')
        primitive = dict(attributes=attributes, indices=add(part['indices'].ravel(), 'SCALAR', 34963))
        if part['material'] is not None:
            primitive['material'] = part['material']
        primitives.append(primitive)
    # Colour textures of blended materials keep their alpha; the rest become JPEG.
    alpha = {m['pbrMetallicRoughness']['baseColorTexture']['index'] for m in document['materials']
             if m.get('alphaMode') in ('BLEND', 'MASK') and 'baseColorTexture' in m.get('pbrMetallicRoughness', {})}
    alpha = {document['textures'][t]['source'] for t in alpha}
    images = []
    for index, image in enumerate(document['images']):
        picture = Image.open(folder / image['uri'])
        picture.thumbnail((1024, 1024), Image.LANCZOS)
        stem = Path(image['uri']).stem
        if index in alpha:
            images.append(dict(uri=f'textures/{stem}.png'))
            picture.save(target / images[-1]['uri'], optimize=True)
        else:
            images.append(dict(uri=f'textures/{stem}.jpg'))
            picture.convert('RGB').save(target / images[-1]['uri'], quality=90)
    materials = copy.deepcopy(document['materials'])
    body = [m for m in materials if m['name'].endswith('_Bodymat')]
    len(body) == 1 or sys.exit('expected one body material in ' + name)
    paint = body[0]['pbrMetallicRoughness']
    paint.pop('metallicRoughnessTexture', None)
    paint.update(metallicFactor=0.15, roughnessFactor=0.3)
    body[0].pop('extensions', None)
    car = dict(asset=dict(version='2.0', generator='prepare_city_assets.py'), scene=0,
               scenes=[dict(nodes=[0])], nodes=[dict(name=name, mesh=0)],
               meshes=[dict(name=name, primitives=primitives)], materials=materials,
               textures=document['textures'], images=images, samplers=document.get('samplers', [{}]),
               buffers=[dict(uri='model.bin', byteLength=len(data))], bufferViews=views, accessors=accessors)
    used = {e for m in materials for e in m.get('extensions', {})}
    if used:
        car['extensionsUsed'] = sorted(used)
    (target / 'model.bin').write_bytes(bytes(data))
    length, width, height = high - low
    collider = [('box', dict(size=(round(float(length) - 0.1, 3), round(float(width) - 0.15, 3),
                                   round(float(height) - 0.12, 3))), (0, 0, round(float(height) / 2, 3)))]
    for i, colour in enumerate([None] + paints):
        if colour is not None:
            paint['baseColorFactor'] = list(colour) + [1]
        stem = 'model' if i == 0 else f'model_paint{i}'
        dump(car, target / f'{stem}.gltf')
        write_rasset(target / f'{stem}.rasset', f'{stem}.gltf', collider)
    shutil.copyfile(folder / 'license.txt', target / 'license.txt')
    print(f'{name}: {length:.2f} x {width:.2f} x {height:.2f} m, '
          f'{sum(len(p["indices"]) for p in parts)} triangles, {len(paints) + 1} paints')


def prepare_cone(source, output):
    from PIL import Image
    from scipy.spatial import ConvexHull
    data = (source / 'traffic_cone/traffic_cone.glb').read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + length])
    binary = data[20 + length + 8:]
    target = output / 'traffic_cone'
    (target / 'textures').mkdir(parents=True, exist_ok=True)

    def view_bytes(index):
        view = document['bufferViews'][index]
        start = view.get('byteOffset', 0)
        return binary[start:start + view['byteLength']]

    def accessor_array(index, width):
        item = document['accessors'][index]
        raw = view_bytes(item['bufferView'])[item.get('byteOffset', 0):]
        dtype = np.float32 if item['componentType'] == 5126 else np.uint32 if item['componentType'] == 5125 else np.uint16
        return np.frombuffer(raw, dtype, item['count'] * width).reshape(item['count'], width)

    primitive = document['meshes'][0]['primitives'][0]
    turn = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)  # Y-up to Z-up
    positions = accessor_array(primitive['attributes']['POSITION'], 3) @ turn.T
    normals = accessor_array(primitive['attributes']['NORMAL'], 3) @ turn.T
    uvs = accessor_array(primitive['attributes']['TEXCOORD_0'], 2)
    indices = accessor_array(primitive['indices'], 1).astype(np.uint32).ravel()
    positions[:, 2] -= positions[:, 2].min()
    names = ['normal', 'diffuse', 'roughness']
    for index, image in enumerate(document['images']):
        picture = Image.open(io.BytesIO(view_bytes(image['bufferView']))).convert('RGB')
        picture.resize((1024, 1024), Image.LANCZOS).save(target / f'textures/traffic_cone_{names[index]}.jpg',
                                                         quality=90)
    buffer = b''.join(a.astype(dtype).tobytes() for a, dtype in
                      ((positions, np.float32), (normals, np.float32), (uvs, np.float32), (indices, np.uint32)))
    sizes = [positions.nbytes, normals.nbytes, uvs.nbytes, indices.nbytes]
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    material = document['materials'][0]
    material['normalTexture'] = dict(index=0)
    material['pbrMetallicRoughness']['baseColorTexture'] = dict(index=1)
    material['pbrMetallicRoughness']['metallicRoughnessTexture'] = dict(index=2)
    cone = dict(asset=dict(version='2.0', generator='prepare_city_assets.py'), scene=0,
                scenes=[dict(nodes=[0])], nodes=[dict(name='traffic_cone', mesh=0)],
                meshes=[dict(name='traffic_cone', primitives=[dict(
                    material=0, indices=3, attributes=dict(POSITION=0, NORMAL=1, TEXCOORD_0=2))])],
                materials=[material], samplers=[dict(magFilter=9729, minFilter=9987)],
                images=[dict(uri=f'textures/traffic_cone_{n}.jpg') for n in names],
                textures=[dict(source=i, sampler=0) for i in range(3)],
                buffers=[dict(uri='model.bin', byteLength=len(buffer))],
                bufferViews=[dict(buffer=0, byteOffset=o, byteLength=s) for o, s in zip(offsets, sizes)],
                accessors=[dict(bufferView=0, componentType=5126, count=len(positions), type='VEC3',
                                min=positions.min(0).tolist(), max=positions.max(0).tolist()),
                           dict(bufferView=1, componentType=5126, count=len(normals), type='VEC3'),
                           dict(bufferView=2, componentType=5126, count=len(uvs), type='VEC2'),
                           dict(bufferView=3, componentType=5125, count=len(indices), type='SCALAR')])
    (target / 'model.bin').write_bytes(buffer)
    dump(cone, target / 'model.gltf')
    hull = ConvexHull(positions)
    print('traffic_cone', len(indices) // 3, 'triangles, hull', len(hull.vertices), 'vertices, height',
          float(positions[:, 2].max()))


def read_hdr(path):
    data = Path(path).read_bytes()
    header_end = data.index(b'\n\n') + 2
    line_end = data.index(b'\n', header_end)
    _, height, _, width = data[header_end:line_end].split()
    height, width = int(height), int(width)
    pixels = np.zeros((height, width, 4), np.uint8)
    position = line_end + 1
    for y in range(height):
        position += 4  # new-style run-length scanline header
        for channel in range(4):
            x = 0
            while x < width:
                count = data[position]
                position += 1
                if count > 128:
                    pixels[y, x:x + count - 128, channel] = data[position]
                    position += 1
                    x += count - 128
                else:
                    pixels[y, x:x + count, channel] = np.frombuffer(data, np.uint8, count, position)
                    position += count
                    x += count
    exponent = pixels[..., 3].astype(np.int32)
    scale = np.where(exponent > 0, np.ldexp(1.0, exponent - 136), 0.0)
    return data[:header_end], pixels[..., :3] * scale[..., None]


def write_hdr(path, header, radiance):
    height, width, _ = radiance.shape
    largest = radiance.max(axis=2)
    mantissa, exponent = np.frexp(largest)
    scale = np.where(largest > 1e-32, mantissa * 256.0 / np.maximum(largest, 1e-32), 0)
    rgbe = np.zeros((height, width, 4), np.uint8)
    rgbe[..., :3] = np.clip(radiance * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(largest > 1e-32, exponent + 128, 0).astype(np.uint8)
    out = bytearray(header + f'-Y {height} +X {width}\n'.encode())
    for y in range(height):
        out += bytes([2, 2, width >> 8, width & 255])
        for channel in range(4):
            row = rgbe[y, :, channel]
            # Runs of three or more equal bytes are run-length coded; the
            # bytes between them are copied in chunks of up to 128.
            change = np.flatnonzero(np.diff(row)) + 1
            starts = np.concatenate(([0], change))
            lengths = np.diff(np.concatenate((starts, [width])))
            done = 0
            for start, run in zip(starts[lengths >= 3].tolist(), lengths[lengths >= 3].tolist()):
                for chunk in range(done, start, 128):
                    piece = row[chunk:min(chunk + 128, start)]
                    out += bytes([len(piece)]) + piece.tobytes()
                while run > 0:
                    count = min(run, 127)
                    out += bytes([128 + count, int(row[start])])
                    start += count
                    run -= count
                done = start
            for chunk in range(done, width, 128):
                piece = row[chunk:min(chunk + 128, width)]
                out += bytes([len(piece)]) + piece.tobytes()
    Path(path).write_bytes(bytes(out))


def prepare_sky(source, output):
    header, radiance = read_hdr(source / 'sky' / SKY)
    luminance = radiance @ np.array([0.2126, 0.7152, 0.0722])
    height, width = luminance.shape
    y, x = np.unravel_index(np.argmax(luminance), luminance.shape)
    weight = np.cos((np.arange(height)[:, None] + 0.5) / height * math.pi - math.pi / 2)
    before = float((luminance * weight).sum())
    factor = np.minimum(1.0, SUN_CLAMP / np.maximum(luminance, 1e-9))
    radiance *= factor[..., None]
    # Turn the sky half a revolution about the vertical so that the sun stands
    # behind the street camera (see SUN_AZIMUTH_DEG in generate_city_rscene.py).
    radiance = np.roll(radiance, width // 2, axis=1)
    after = float(((radiance @ np.array([0.2126, 0.7152, 0.0722])) * weight).sum())
    (output / 'sky').mkdir(parents=True, exist_ok=True)
    write_hdr(output / 'sky' / SKY, header, radiance.astype(np.float32))
    print(f'sky: sun at u={(x + 0.5) / width:.4f}, elevation {90 - (y + 0.5) / height * 180:.1f} deg; '
          f'clamping keeps {after / before:.1%} of the radiance')


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit('Usage: prepare_city_assets.py DIR [OUTPUT]')
    source = Path(sys.argv[1]) / 'assets'
    output = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(sys.argv[1]) / 'prepared'
    output.mkdir(parents=True, exist_ok=True)
    for folder, modules in generate_city_rscene.used_modules().items():
        split_kit(source, output, folder, modules)
    for name, spec in PROPS.items():
        prepare_prop(source, output, name, spec)
    for name, paints in CARS.items():
        prepare_car(source, output, name, paints)
    prepare_cone(source, output)
    for name in TEXTURES:
        shutil.copytree(source / 'textures' / name, output / 'textures' / name, dirs_exist_ok=True)
    prepare_sky(source, output)
    shutil.copyfile(source / 'sources.json', output / 'sources.json')
    notices = Path(__file__).resolve().parents[2] / 'rsc/city'
    for notice in ['LICENSE-CC0-1.0.txt', 'ATTRIBUTION.md']:
        if (notices / notice).resolve() != (output / notice).resolve():
            shutil.copyfile(notices / notice, output / notice)
    generate_city_rscene.generate(output)
    files = [dict(path=str(p.relative_to(output)), bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'
             and not p.name.endswith('.lods') and not p.name.endswith('.rmpc')]
    (output / 'manifest.json').write_text(json.dumps(dict(license='CC0-1.0', files=files), indent=2) + '\n')


if __name__ == '__main__':
    main()

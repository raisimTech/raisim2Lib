#!/usr/bin/env python3
"""Lay out the rayrai city and write it as rayrai_city.rscene.

The city is a grid of blocks. Each block is split into two rows of lots, and
each lot becomes a building assembled from Poly Haven's modular apartment or
factory facade kit (3 m wall modules with matching window and door inserts,
plus base, dado, cornice, crown and corner trims). Streets get asphalt,
sidewalks with curbs, lane markings, crosswalks, lamps, trees and other
street furniture. Everything is deterministic for a fixed seed.

The scene stores, as records RaiSim and rayrai read:
  * static, hidden box colliders for the ground, the sidewalks and every
    building, and .rasset colliders for the street furniture and trees;
  * dynamic traffic cones (convex hull of their mesh);
  * one instanced batch per facade module and prop;
  * two baked visual-only meshes: streets (asphalt, sidewalks, curbs,
    markings, tree pits) and building shells (roofs and dark interiors seen
    through the windows);
  * the HDR sky, the sun, render settings and the street-level camera.

The quadruped is not part of the file: the .rscene reader builds no
articulated systems, so rayrai_city.cpp adds it after loading.

Usage: generate_city_rscene.py CITY_DIR   (a prepared rsc/city directory)
"""
import json
import math
import random
import struct
import sys
from pathlib import Path

SEED = 7
BLOCKS = 4                 # blocks per axis
BLOCK = 36.0               # building area of a block; a multiple of the 3 m module
SIDEWALK = 3.5             # curb to building line
ROAD_HALF = 6.0            # street centre to curb: two 3.5 m lanes and 2.5 m parking
PITCH = BLOCK + 2 * (SIDEWALK + ROAD_HALF)
CURB = 0.15                # sidewalk height above the road
FLOOR0 = 0.55              # ground-floor level; the base plinth shows above the sidewalk
FLOOR = 3.0
LOT_DEPTH = 18.0
GROUND_EXTENT = 450.0      # half size of the asphalt plane around the city
MASK = '18446744073709551615'
# Sun of the prepared sky HDR: the elevation is measured in the image, the
# azimuth (from +x towards +y) where rayrai shows the sun disk of the turned
# image, measured in a rendered view.
SUN_ELEVATION_DEG = 47.9
SUN_AZIMUTH_DEG = 215.8
SKY_HDR = 'sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr'
# Street-level hero view: the robot stands on the middle street, by roadworks.
ROBOT_XY = (-36.0, -1.6)
CAMERA = dict(position=(-41.5, -3.4, 1.05), yaw_deg=6.0, pitch_deg=-2.0, vfov=50.0)


def street_centres():
    return [(k - BLOCKS / 2) * PITCH for k in range(BLOCKS + 1)]


# ---------------------------------------------------------------- facade kits
class Kit:
    """Module names of one facade kit; all walls are 3 m wide except garages."""

    def __init__(self, folder, base, dado, cornice, crown, corner, crown_height):
        self.folder, self.base, self.dado = folder, base, dado
        self.cornice, self.crown, self.corner = cornice, crown, corner
        self.crown_height = crown_height
        self.plain = 'wall_standard_standard_01'

    def path(self, module):
        return f'{self.folder}/modules/{module}.gltf'


APARTMENTS = Kit('apartments', 'base_standard_01', 'dado_standard_standard_01',
                 'cornice_standard_standard_01', 'crown_standard_standard_01',
                 dict(wall='wall_standard_corner_large_01', base='base_corner_large_01',
                      dado='dado_standard_corner_large_01',
                      cornice='cornice_standard_corner_large_01',
                      crown='crown_standard_corner_large_01'), 0.75)
FACTORY = Kit('factory', 'base_standard_standard_01', 'dado_standard_standard_01',
              'cornice01_standard_standard_01', 'crown_standard_standard_01',
              dict(wall='wall_standard_corner_large_01', base='base_standard_corner_large_01',
                   dado='dado_standard_corner_large_01',
                   cornice='cornice01_standard_corner_large_01',
                   crown='crown_standard_corner_large_01'), 0.32)
APARTMENT_TYPES = ['large', 'small', 'double']
APARTMENT_PATTERNS = [['large'], ['large', 'small'], ['double', 'large', 'double'],
                      ['small', 'double'], ['large', 'large', 'small'], ['double']]
FACTORY_TYPES = ['large', 'medium', 'double', 'tall']


def window(lot, column, floor):
    """(wall, insert) modules of an upper-floor window, or a ground-floor window."""
    if lot['kit'] is APARTMENTS:
        # Tallest windows on the first floor, smaller trimmed ones on the top floor.
        variant = '01' if floor <= 1 else '03' if floor == lot['floors'] - 1 and lot['floors'] >= 4 else '02'
        name = f'window_centered_{column}_{variant}'
    elif floor == 0:
        name = 'window_centered_medium_01'
    elif column == 'tall':
        # Tall factory windows stack into one strip: bottom, middle and top pieces.
        variant = '01' if floor == 1 else '04' if floor == lot['floors'] - 1 else '02'
        name = f'window_tall_large_{variant}'
    else:
        name = f'window_centered_{column}_0{lot["variant"]}'
    return 'wall_' + name, name


def ground_door(lot, kind):
    """(wall, insert, dado, width) of a ground-floor door or garage."""
    if lot['kit'] is APARTMENTS:
        size = 'small' if kind == 'door_small' else 'large'
        return (f'wall_door_centered_{size}_01', f'door_centered_{size}_01',
                f'dado_door_centered_{size}_01', 3.0)
    if kind == 'garage':
        return 'wall_door_garage_door_01', 'door_garage_door_01', 'dado_garage_door_01', 6.0
    return 'wall_door_recessed_small_01', 'door_recessed_small_01', 'dado_door_recessed_small_01', 3.0


def used_modules():
    """Every module name each kit can use, for prepare_city_assets.py."""
    used = {APARTMENTS.folder: set(), FACTORY.folder: set()}
    for kit in (APARTMENTS, FACTORY):
        names = used[kit.folder]
        names.update([kit.plain, kit.base, kit.dado, kit.cornice, kit.crown, *kit.corner.values()])
        for floors in range(3, 8):
            for variant in range(1, 5):
                lot = dict(kit=kit, floors=floors, variant=variant)
                for column in (APARTMENT_TYPES if kit is APARTMENTS else FACTORY_TYPES):
                    for floor in range(floors):
                        names.update(window(lot, column, floor))
                for kind in ('door_small', 'door_large', 'garage', 'door'):
                    names.update(ground_door(lot, kind)[:3])
    return {kit: sorted(names) for kit, names in used.items()}


# ---------------------------------------------------------------- layout
def lots_of_block(rng, bx0, by0):
    """Two rows of lots, 18 m deep, with widths that sum to the block width."""
    splits = [[12, 12, 12], [9, 15, 12], [15, 9, 12], [12, 9, 15], [9, 9, 9, 9],
              [15, 12, 9], [9, 12, 15], [12, 15, 9], [18, 9, 9], [9, 18, 9]]
    lots = []
    for row in range(2):
        x = bx0
        for width in rng.choice(splits):
            apartments = rng.random() < 0.65
            lot = dict(x0=x, x1=x + width, y0=by0 + row * LOT_DEPTH, y1=by0 + (row + 1) * LOT_DEPTH,
                       kit=APARTMENTS if apartments else FACTORY,
                       floors=rng.randint(4, 6) if apartments else rng.randint(3, 5),
                       pattern=rng.choice(APARTMENT_PATTERNS),
                       column=rng.choice(FACTORY_TYPES), variant=rng.randint(1, 4),
                       block=(bx0, by0, bx0 + BLOCK, by0 + BLOCK))
            lots.append(lot)
            x += width
    return lots


def lot_faces(lot):
    """Faces in counter-clockwise order: start point, tangent, yaw and length.

    A module is modelled on x in [-w, 0] with its outer side towards -y, so at
    a face's arc length s it is placed at start + tangent * (s + w), turned by
    the face's yaw. Each face starts where the previous one ends.
    """
    x0, y0, x1, y1 = lot['x0'], lot['y0'], lot['x1'], lot['y1']
    bx0, by0, bx1, by1 = lot['block']
    return [dict(name='S', start=(x0, y0), t=(1, 0), yaw=0, length=x1 - x0, street=y0 == by0),
            dict(name='E', start=(x1, y0), t=(0, 1), yaw=90, length=y1 - y0, street=x1 == bx1),
            dict(name='N', start=(x1, y1), t=(-1, 0), yaw=180, length=x1 - x0, street=y1 == by1),
            dict(name='W', start=(x0, y1), t=(0, -1), yaw=270, length=y1 - y0, street=x0 == bx0)]


class Placements:
    """Instances of every batch: (x, y, z, yaw degrees, uniform scale)."""

    def __init__(self):
        self.batches = {}

    def add(self, mesh, x, y, z, yaw=0.0, scale=1.0):
        self.batches.setdefault(mesh, []).append((x, y, z, yaw, scale))


def place_on_face(out, kit, face, module, s, width, z):
    x = face['start'][0] + face['t'][0] * (s + width)
    y = face['start'][1] + face['t'][1] * (s + width)
    out.add(kit.path(module), x, y, z, face['yaw'])


def floors_at(lots, x, y):
    for lot in lots:
        if lot['x0'] < x < lot['x1'] and lot['y0'] < y < lot['y1']:
            return lot['floors']
    return 0


def build_facades(lot, lots, out):
    kit, floors = lot['kit'], lot['floors']
    top = FLOOR0 + FLOOR * floors
    faces = lot_faces(lot)
    for index, face in enumerate(faces):
        previous, following = faces[index - 1], faces[(index + 1) % 4]
        corner_start = face['street'] and previous['street']
        corner_end = face['street'] and following['street']
        if corner_start:
            for floor in range(floors):
                place_on_face(out, kit, face, kit.corner['wall'], 0, 3, FLOOR0 + FLOOR * floor)
            place_on_face(out, kit, face, kit.corner['base'], 0, 3, FLOOR0 - 0.75)
            place_on_face(out, kit, face, kit.corner['dado'], 0, 3, FLOOR0)
            place_on_face(out, kit, face, kit.corner['cornice'], 0, 3, FLOOR0 + FLOOR)
            place_on_face(out, kit, face, kit.corner['crown'], 0, 3, top)
        begin = 3 if corner_start else 0
        end = face['length'] - (3 if corner_end else 0)
        slots = [begin + 3 * i for i in range(int(round((end - begin) / 3)))]
        if not face['street']:
            # Party and back walls show only above the neighbouring building.
            nx, ny = face['t'][1], -face['t'][0]
            for s in slots:
                mx = face['start'][0] + face['t'][0] * (s + 1.5) + nx * 0.5
                my = face['start'][1] + face['t'][1] * (s + 1.5) + ny * 0.5
                neighbour = floors_at(lots, mx, my)
                for floor in range(neighbour, floors):
                    place_on_face(out, kit, face, kit.plain, s, 3, FLOOR0 + FLOOR * floor)
                if neighbour < floors:
                    place_on_face(out, kit, face, kit.crown, s, 3, top)
            continue
        if kit is APARTMENTS:
            columns = [lot['pattern'][i % len(lot['pattern'])] for i in range(len(slots))]
        else:
            columns = [lot['column']] * len(slots)
        # Ground floor: one entrance per street face, and a garage on long factory faces.
        ground = ['window'] * len(slots)
        if kit is FACTORY and len(slots) >= 4:
            ground[1], ground[2] = 'garage', 'skip'
            ground[len(slots) - 2 if len(slots) >= 5 else 3] = 'door'
        elif slots:
            middle = len(slots) // 2
            ground[middle] = 'door' if kit is FACTORY else (
                'door_small' if columns[middle] == 'small' else 'door_large')
        for i, s in enumerate(slots):
            place_on_face(out, kit, face, kit.base, s, 3, FLOOR0 - 0.75)
            place_on_face(out, kit, face, kit.cornice, s, 3, FLOOR0 + FLOOR)
            place_on_face(out, kit, face, kit.crown, s, 3, top)
            if ground[i] == 'window':
                wall, insert = window(lot, columns[i], 0)
                for module in (wall, insert):
                    place_on_face(out, kit, face, module, s, 3, FLOOR0)
                place_on_face(out, kit, face, kit.dado, s, 3, FLOOR0)
            elif ground[i] != 'skip':
                wall, insert, dado, width = ground_door(lot, ground[i])
                for module in (wall, insert, dado):
                    place_on_face(out, kit, face, module, s, width, FLOOR0)
            for floor in range(1, floors):
                for module in window(lot, columns[i], floor):
                    place_on_face(out, kit, face, module, s, 3, FLOOR0 + FLOOR * floor)


# ---------------------------------------------------------------- baked meshes
class MeshBuilder:
    """Triangles grouped by material, written as one glTF file and buffer.

    With max_cell set, quads are split into a grid of cells no larger than
    that. rayrai v2.8.0 interpolates a receiver's shadow-cascade depth from
    abs(view depth) per vertex, so a large triangle reaching behind the camera
    makes the ground just ahead sample a far cascade and lose its shadows.
    Small cells keep the receivers that the camera can approach correct.
    """

    def __init__(self, materials, images, max_cell=None):
        self.materials, self.images, self.max_cell = materials, images, max_cell
        self.parts = [dict(p=[], n=[], uv=[], i=[]) for _ in materials]

    def grid(self, material, origin, axis_u, axis_v, size_u, size_v, normal, uv, subdivide=True):
        """Planar patch origin + s * axis_u + t * axis_v; uv(point) gives texture coordinates."""
        cells_u = max(1, math.ceil(size_u / self.max_cell - 1e-9)) if subdivide and self.max_cell else 1
        cells_v = max(1, math.ceil(size_v / self.max_cell - 1e-9)) if subdivide and self.max_cell else 1
        part = self.parts[material]
        base = len(part['p'])
        for j in range(cells_v + 1):
            for i in range(cells_u + 1):
                s, t = size_u * i / cells_u, size_v * j / cells_v
                point = tuple(o + s * a + t * b for o, a, b in zip(origin, axis_u, axis_v))
                part['p'].append(point)
                part['n'].append(normal)
                part['uv'].append(uv(point))
        for j in range(cells_v):
            for i in range(cells_u):
                a = base + j * (cells_u + 1) + i
                b, c, d = a + 1, a + cells_u + 2, a + cells_u + 1
                part['i'].extend([a, b, c, a, c, d])

    def top(self, material, x0, y0, x1, y1, z, tile, subdivide=True):
        """Upward quad with world-space texture coordinates (tile metres per repeat)."""
        self.grid(material, (x0, y0, z), (1, 0, 0), (0, 1, 0), x1 - x0, y1 - y0, (0, 0, 1),
                  lambda p: (p[0] / tile, -p[1] / tile), subdivide)

    def wall(self, material, a, b, z0, z1, tile):
        """Vertical quad from a to b; its front faces right of the direction a->b."""
        (ax, ay), (bx, by) = a, b
        length = math.hypot(bx - ax, by - ay)
        direction = ((bx - ax) / length, (by - ay) / length, 0)
        normal = (direction[1], -direction[0], 0)
        self.grid(material, (ax, ay, z0), direction, (0, 0, 1), length, z1 - z0, normal,
                  lambda p: ((p[0] * direction[0] + p[1] * direction[1]) / tile, -p[2] / tile))

    def box_sides(self, material, x0, y0, x1, y1, z0, z1, tile):
        for a, b in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
            self.wall(material, a, b, z0, z1, tile)

    def write(self, path):
        path = Path(path)
        data = bytearray()
        views, accessors, primitives = [], [], []

        def add(values, fmt, kind, target, bounds=False):
            while len(data) % 4:
                data.append(0)
            offset = len(data)
            for value in values:
                data.extend(struct.pack(fmt, *value) if isinstance(value, tuple) else struct.pack(fmt, value))
            views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data) - offset, target=target))
            accessor = dict(bufferView=len(views) - 1, componentType=5125 if kind == 'SCALAR' else 5126,
                            count=len(values), type=kind)
            if bounds:
                accessor['min'] = [min(v[k] for v in values) for k in range(3)]
                accessor['max'] = [max(v[k] for v in values) for k in range(3)]
            accessors.append(accessor)
            return len(accessors) - 1

        for material, part in enumerate(self.parts):
            if not part['i']:
                continue
            primitives.append(dict(material=material, indices=add(part['i'], '<I', 'SCALAR', 34963),
                                   attributes=dict(POSITION=add(part['p'], '<3f', 'VEC3', 34962, True),
                                                   NORMAL=add(part['n'], '<3f', 'VEC3', 34962),
                                                   TEXCOORD_0=add(part['uv'], '<2f', 'VEC2', 34962))))
        document = dict(asset=dict(version='2.0', generator='generate_city_rscene.py'),
                        scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name=path.stem)],
                        meshes=[dict(name=path.stem, primitives=primitives)],
                        materials=self.materials, buffers=[dict(uri=path.stem + '.bin', byteLength=len(data))],
                        bufferViews=views, accessors=accessors)
        if self.images:
            document['images'] = [dict(uri=uri) for uri in self.images]
            document['textures'] = [dict(source=i, sampler=0) for i in range(len(self.images))]
            document['samplers'] = [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / (path.stem + '.bin')).write_bytes(data)
        path.write_text(json.dumps(document, indent=1) + '\n')


def textured_materials(specs, prefix):
    """glTF materials from (name, texture folder or None, colour, roughness)."""
    materials, images = [], []
    for name, folder, color, roughness in specs:
        material = dict(name=name, pbrMetallicRoughness=dict(baseColorFactor=list(color) + [1],
                                                             metallicFactor=0, roughnessFactor=roughness))
        if folder:
            stem, resolution = folder
            base = len(images)
            images += [f'{prefix}{stem}/{stem}_diff_{resolution}.jpg',
                       f'{prefix}{stem}/{stem}_nor_gl_{resolution}.jpg',
                       f'{prefix}{stem}/{stem}_arm_{resolution}.jpg']
            material['pbrMetallicRoughness'].update(baseColorTexture=dict(index=base),
                                                    metallicRoughnessTexture=dict(index=base + 2),
                                                    metallicFactor=1, roughnessFactor=1)
            material['normalTexture'] = dict(index=base + 1)
            material['occlusionTexture'] = dict(index=base + 2)
        materials.append(material)
    return materials, images


ASPHALT, SIDEWALK_MAT, CURB_MAT, PAINT, SOIL = range(5)
ROOF, INTERIOR = range(2)


def street_mesh(blocks, trees):
    materials, images = textured_materials([
        ('asphalt', ('asphalt_02', '2k'), (1, 1, 1), 1),
        ('sidewalk', ('concrete_pavement_02', '2k'), (1, 1, 1), 1),
        ('curb', ('concrete_floor_01', '1k'), (0.92, 0.92, 0.92), 1),
        ('road_paint', None, (0.78, 0.78, 0.74), 0.62),
        ('tree_pit', None, (0.16, 0.12, 0.09), 0.95)], '../textures/')
    mesh = MeshBuilder(materials, images, max_cell=3.0)
    e = GROUND_EXTENT
    # Asphalt in 50 m tiles keeps texture coordinates small; tiles that a camera
    # in the city can stand on are split into 3 m cells (see MeshBuilder).
    city = BLOCKS / 2 * PITCH + BLOCK + SIDEWALK + ROAD_HALF
    for x in range(-int(e), int(e), 50):
        for y in range(-int(e), int(e), 50):
            near = x < city and x + 50 > -city and y < city and y + 50 > -city
            mesh.top(ASPHALT, x, y, x + 50, y + 50, 0, 3.0, subdivide=near)
    curb_width = 0.2
    for bx0, by0, bx1, by1 in blocks:
        x0, y0, x1, y1 = bx0 - SIDEWALK, by0 - SIDEWALK, bx1 + SIDEWALK, by1 + SIDEWALK
        mesh.top(SIDEWALK_MAT, x0 + curb_width, y0 + curb_width, x1 - curb_width, y1 - curb_width, CURB, 1.8)
        for a in [(x0, y0, x1, y0 + curb_width), (x0, y1 - curb_width, x1, y1),
                  (x0, y0 + curb_width, x0 + curb_width, y1 - curb_width),
                  (x1 - curb_width, y0 + curb_width, x1, y1 - curb_width)]:
            mesh.top(CURB_MAT, *a, CURB, 2.0)
        mesh.box_sides(CURB_MAT, x0, y0, x1, y1, 0, CURB, 2.0)
    for x, y in trees:
        mesh.top(SOIL, x - 0.6, y - 0.6, x + 0.6, y + 0.6, CURB + 0.004, 1.2)
    paint = 0.012  # above the asphalt, enough for the depth buffer at 100 m
    centres = street_centres()
    stripe_from = ROAD_HALF + 0.8      # crosswalk start, from the crossing street's centre
    stripe_to = stripe_from + 3.0
    for c in centres:
        for a, b in zip(centres, centres[1:]):
            lo, hi = a + stripe_to + 1.5, b - stripe_to - 1.5
            for along in (True, False):  # streets along x (centre y=c) and along y (centre x=c)
                def rect(u0, u1, v0, v1):
                    if along:
                        mesh.top(PAINT, u0, c + v0, u1, c + v1, paint, 1)
                    else:
                        mesh.top(PAINT, c + v0, u0, c + v1, u1, paint, 1)
                u = lo
                while u + 3 <= hi:                       # dashed centre line
                    rect(u, u + 3, -0.06, 0.06)
                    u += 9
                for side in (-1, 1):                     # parking lane edges
                    v = side * (ROAD_HALF - 2.5)
                    rect(lo, hi, v - 0.05, v + 0.05)
                # Zebra crossings in front of both intersections, and stop lines.
                for start, direction in ((a, 1), (b, -1)):
                    near, far = start + direction * stripe_from, start + direction * stripe_to
                    v = -ROAD_HALF + 0.6
                    while v + 0.5 <= ROAD_HALF - 0.5:
                        rect(min(near, far), max(near, far), v, v + 0.5)
                        v += 1.0
                    stop = far + direction * 1.0
                    # The stop line covers the lane that drives towards the intersection
                    # (right-hand traffic).
                    towards_positive_v = (direction > 0) == along
                    lane = (0.0, ROAD_HALF - 0.4) if towards_positive_v else (-ROAD_HALF + 0.4, 0.0)
                    rect(min(stop, stop + direction * 0.4), max(stop, stop + direction * 0.4), *lane)
    return mesh


def building_mesh(lots):
    materials, images = textured_materials([
        ('roof', ('bitumen', '1k'), (0.85, 0.85, 0.85), 1),
        ('interior', None, (0.025, 0.024, 0.023), 0.9)], '../textures/')
    mesh = MeshBuilder(materials, images)
    inset = 0.3
    for lot in lots:
        top = FLOOR0 + FLOOR * lot['floors']
        mesh.top(ROOF, lot['x0'], lot['y0'], lot['x1'], lot['y1'], top + 0.02, 6.0)
        # Dark walls just behind the facades read as unlit rooms through the glass.
        mesh.box_sides(INTERIOR, lot['x0'] + inset, lot['y0'] + inset, lot['x1'] - inset,
                       lot['y1'] - inset, FLOOR0, top, 3.0)
    return mesh


# ---------------------------------------------------------------- street furniture
def sidewalk_frames(block):
    """Per block side: origin at the curb, along-street tangent, outward normal, yaw."""
    bx0, by0, bx1, by1 = block
    return [dict(o=(bx0, by0 - SIDEWALK), t=(1, 0), n=(0, -1), yaw=0),
            dict(o=(bx1 + SIDEWALK, by0), t=(0, 1), n=(1, 0), yaw=90),
            dict(o=(bx1, by1 + SIDEWALK), t=(-1, 0), n=(0, 1), yaw=180),
            dict(o=(bx0 - SIDEWALK, by1), t=(0, -1), n=(-1, 0), yaw=270)]


def at(frame, along, from_curb):
    """Point on a sidewalk: along the block side, and inward from the curb."""
    return (frame['o'][0] + frame['t'][0] * along - frame['n'][0] * from_curb,
            frame['o'][1] + frame['t'][1] * along - frame['n'][1] * from_curb)


def near_robot(x, y, radius):
    return math.hypot(x - ROBOT_XY[0], y - ROBOT_XY[1]) < radius


def furnish(rng, blocks, props, trees):
    for block in blocks:
        for side, frame in enumerate(sidewalk_frames(block)):
            for along in (4.5, 22.5):
                x, y = at(frame, along, 0.45)
                props.add('lamp', x, y, CURB, frame['yaw'])
            for along in (13.5, 31.5):
                x, y = at(frame, along, 1.0)
                trees.append((x, y))
                props.add('tree', x, y, CURB, rng.uniform(0, 360), rng.uniform(1.35, 1.7))
            if side % 2 == 0:
                x, y = at(frame, 1.6, 0.55)
                props.add('hydrant', x, y, CURB, rng.uniform(0, 360))
            x, y = at(frame, 34.2, 0.65)
            props.add('trash_can', x, y, CURB, rng.uniform(0, 360))
            if rng.random() < 0.5:
                x, y = at(frame, 18.0, 2.2)
                props.add('bench', x, y, CURB, frame['yaw'])
            if rng.random() < 0.35:
                x, y = at(frame, 9.0, SIDEWALK - 0.5)
                props.add('utility_box', x, y, CURB, frame['yaw'])


def park_cars(rng, props):
    centres = street_centres()
    clear = ROAD_HALF + 0.8 + 3.0 + 3.0
    for c in centres:
        for a, b in zip(centres, centres[1:]):
            for along_x in (True, False):
                for side in (-1, 1):
                    u = a + clear + rng.uniform(0, 3)
                    while u < b - clear:
                        v = c + side * (ROAD_HALF - 1.25)
                        x, y = (u, v) if along_x else (v, u)
                        # Keep the camera, the robot and the roadworks free.
                        hero = along_x and c == 0 and a < ROBOT_XY[0] < b and u < ROBOT_XY[0] + 14
                        if not hero and rng.random() < 0.4:
                            # Parked in the direction of travel of right-hand traffic.
                            yaw = (0 if side < 0 else 180) if along_x else (90 if side > 0 else 270)
                            props.add(rng.choices(CAR_MODELS, CAR_WEIGHTS)[0], x, y, 0,
                                      yaw + rng.uniform(-1.5, 1.5))
                        u += rng.uniform(6.0, 7.5)
            if c in (centres[0], centres[-1]):
                continue
            for along_x in (True, False):
                u = a + rng.uniform(12, 20)
                x, y = (u, c + 1.75) if along_x else (c + 1.75, u)
                props.add('manhole', x, y, -0.03, rng.uniform(0, 360))


def roadworks(rng, props):
    """Barriers closing the far lane ahead of the robot, and a taper of cones."""
    cones = []
    for i in range(3):
        props.add('barrier', -27.0 + i * 1.6, 2.4, 0, 0)
    for i in range(7):
        x = -38.5 + i * 1.5
        y = 4.6 - i * 0.33
        cones.append((x, y, rng.uniform(0, 360)))
    for x, y in [(-25.5, 1.2), (-22.0, 1.3), (-31.4, -2.8), (-33.0, 0.9)]:
        cones.append((x, y, rng.uniform(0, 360)))
    return cones


# ---------------------------------------------------------------- records
def fmt(value, digits=5):
    text = ('%.*f' % (digits, value)).rstrip('0').rstrip('.')
    return '0' if text in ('-0', '') else text


def fmt_quat(quat):
    # .rasset objects need unit quaternions to within 1e-5 of squared norm.
    return ' '.join(fmt(v, 9) for v in quat)


def quat_yaw(deg):
    half = math.radians(deg) / 2
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def quat_yaw_pitch(yaw_deg, pitch_deg):
    """Rotation by yaw about +z, then by pitch about the turned +y (positive looks down)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def object_record(path, primitive, position, quat, scale, mass, visible, mesh, mode, keys=''):
    numbers = f'{" ".join(fmt(v) for v in position)} {fmt_quat(quat)} {" ".join(fmt(v) for v in scale)}'
    return (f'object {path} {primitive} {numbers} 0.5 1 {fmt(mass)} default engine_default '
            f'{"true" if mode == "visual_only" else "false"} {"true" if visible else "false"} '
            f'{"true" if mode == "static" else "false"} {mesh} {mode} '
            f'{"false" if mode == "visual_only" else "true"} 1 {MASK}' + (' ' + keys if keys else ''))


PROP_MESHES = {
    'lamp': 'props/street_lamp_01/model.rasset',
    'tree': '../forest/tree_small_02/model.rasset',
    'hydrant': 'props/fire_hydrant/model.rasset',
    'trash_can': 'props/metal_trash_can/model.rasset',
    'bench': 'props/modular_street_seating/model.rasset',
    'utility_box': 'props/utility_box_02/model.rasset',
    **{f'car_{name}_{paint}': f'cars/{name}/model{"" if paint == 0 else f"_paint{paint}"}.rasset'
       for name in ('fairheaven_lt80', 'fairheaven_sw84', 'kiri86') for paint in range(3)},
    'car_canyon75_taxi_0': 'cars/canyon75_taxi/model.rasset',
    'barrier': 'props/concrete_road_barrier/model.rasset',
    'manhole': 'props/water_manhole_cover/model.gltf',
}


CAR_MODELS = [name for name in PROP_MESHES if name.startswith('car_')]
CAR_WEIGHTS = [0.5 if 'taxi' in name else 1.0 for name in CAR_MODELS]


def instanced_record(path, mesh, instances, keys=''):
    values = ';'.join(','.join(fmt(v) for v in (x, y, z, *quat_yaw(yaw), s, s, s))
                      for x, y, z, yaw, s in instances)
    return (f'instanced_visual {path} mesh meshPath={mesh} size=1,1,1 colorA=1,1,1,1 colorB=1,1,1,1 '
            f'instances={values} castShadows=true automaticMeshLod=true' + (' ' + keys if keys else ''))


def generate(city_dir):
    city_dir = Path(city_dir)
    rng = random.Random(SEED)
    centres = street_centres()
    margin = ROAD_HALF + SIDEWALK
    blocks = [(a + margin, b + margin, a + margin + BLOCK, b + margin + BLOCK)
              for a in centres[:-1] for b in centres[:-1]]
    # One more block beyond the last cross street ends the camera's street at a
    # T-junction, so the view down the street closes with buildings.
    end = centres[-1] + margin
    blocks.append((end, -BLOCK / 2, end + BLOCK, BLOCK / 2))
    lots = []
    for bx0, by0, _, _ in blocks:
        lots += lots_of_block(rng, bx0, by0)
    facades = Placements()
    for lot in lots:
        build_facades(lot, lots, facades)
    props, trees = Placements(), []
    furnish(rng, blocks, props, trees)
    park_cars(rng, props)
    cones = roadworks(rng, props)

    street_mesh(blocks, trees).write(city_dir / 'ground/city_streets.gltf')
    building_mesh(lots).write(city_dir / 'ground/city_buildings.gltf')

    sun_dir = (-math.cos(math.radians(SUN_ELEVATION_DEG)) * math.cos(math.radians(SUN_AZIMUTH_DEG)),
               -math.cos(math.radians(SUN_ELEVATION_DEG)) * math.sin(math.radians(SUN_AZIMUTH_DEG)),
               -math.sin(math.radians(SUN_ELEVATION_DEG)))
    cam = CAMERA
    lines = [
        'raisim_engine_scene 2',
        '# rayrai city: Poly Haven modular facades and street props, a BlendKit traffic cone, all CC0.',
        '# Generated by examples/tools/generate_city_rscene.py; the quadruped is added by rayrai_city.cpp.',
        'time_step 0.002',
        'gravity 0 0 -9.81',
        # raisim::World's own solver defaults (see the .rscene documentation).
        'solver 150 1e-08 1.5 accurate erp2=0.002 defaultRestitutionThreshold=0.01 '
        'defaultStaticFrictionVelocityThreshold=1 fixedContactSolverIterationOrder=true',
        'asset_root .',
        f'environment 0.62 0.7 0.82 0.1 0.1 0.11 0.0018 true false 10 {SKY_HDR} '
        'backgroundMode=hdr fogColor=0.66,0.72,0.8 exposure=1 gamma=2.2 bloomIntensity=0.05 '
        'bloomThreshold=1.6 bloomRadius=4 shadowMapSize=4096 shadowedLightBudget=1 colorMode=aces_approx '
        'pbrEnvironmentIntensity=1 fxaa=true ssao=true',
        'rayrai_render preset=high custom=true colorMode=aces_approx viewerMsaaSamples=4 '
        'shadowResolution=4096 shadowBias=0.0005 shadowStrength=1 shadowPcfRadius=1.5 '
        'directionalShadowCascadeCount=2 directionalShadowCascadeLambda=0.7 '
        'directionalShadowCascadeMaxDistance=100 highFidelityPbr=true pbrToneMapping=true '
        'pbrExposure=1 pbrEnvironmentIntensity=1 addViewerFillLights=false reflectiveGround=false '
        'contactShadows=true',
        f'light /World/Lights/Sun {" ".join(fmt(v) for v in sun_dir)} 5 type=directional shadows=true '
        'color=1,0.96,0.9 ambientColor=0,0,0 shadowResolution=4096 shadowBias=0.0005 shadowStrength=1 '
        'shadowPcfRadius=1.5 shadowOrthoHalfSize=80 shadowNear=0.1 shadowFar=400',
        f'camera /World/Cameras/Street {" ".join(fmt(v) for v in (*cam["position"], *quat_yaw_pitch(cam["yaw_deg"], cam["pitch_deg"])))} '
        f'{fmt(cam["vfov"])} 0.1 2000 1280 800 rgb true projection=perspective',
        object_record('/World/Physics/Ground', 'ground', (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 1, False, '-', 'static'),
        object_record('/World/Streets', 'mesh', (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 1, True,
                      'ground/city_streets.gltf', 'visual_only',
                      'renderMeshPath=ground/city_streets.gltf visualUseMeshColor=false'),
        object_record('/World/BuildingShells', 'mesh', (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 1, True,
                      'ground/city_buildings.gltf', 'visual_only',
                      'renderMeshPath=ground/city_buildings.gltf visualUseMeshColor=false'),
    ]
    for i, (bx0, by0, bx1, by1) in enumerate(blocks):
        size = BLOCK + 2 * SIDEWALK
        lines.append(object_record(f'/World/Physics/Sidewalk_{i}', 'box',
                                   ((bx0 + bx1) / 2, (by0 + by1) / 2, CURB / 2), (1, 0, 0, 0),
                                   (size, size, CURB), 1, False, '-', 'static'))
    for i, lot in enumerate(lots):
        height = FLOOR0 + FLOOR * lot['floors']
        lines.append(object_record(f'/World/Physics/Building_{i}', 'box',
                                   ((lot['x0'] + lot['x1']) / 2, (lot['y0'] + lot['y1']) / 2, height / 2),
                                   (1, 0, 0, 0), (lot['x1'] - lot['x0'], lot['y1'] - lot['y0'], height),
                                   1, False, '-', 'static'))
    for name, instances in sorted(props.batches.items()):
        mesh = PROP_MESHES[name]
        # Cars have glass and plain-coloured paint, which instanced batches do not
        # draw correctly; each car is a visible .rasset object, a regular visual.
        car = name.startswith('car_')
        if not car:
            keys = ('foliageWindEnabled=true grassBladeRootHeight=0 grassBladeTipHeight=1.3 '
                    'grassWindStrength=0.025 grassStiffness=0.7 grassFlutterWeight=0.2 '
                    'shadowFoliageLod=true projectedLod=true') if name == 'tree' else ''
            lines.append(instanced_record(f'/World/Street/{name}', mesh, instances, keys))
        if mesh.endswith('.rasset'):
            group = 'Cars' if car else 'Collision'
            for i, (x, y, z, yaw, s) in enumerate(instances):
                lines.append(object_record(f'/World/{group}/{name}_{i}', 'mesh', (x, y, z), quat_yaw(yaw),
                                           (s, s, s), 1, car, mesh, 'static',
                                           f'renderMeshPath={mesh}' + (' visualUseMeshColor=false' if car else '')))
    for mesh, instances in sorted(facades.batches.items()):
        name = mesh.replace('/modules/', '/').replace('.gltf', '')
        lines.append(instanced_record(f'/World/Buildings/{name}', mesh, instances))
    for i, (x, y, yaw) in enumerate(cones):
        lines.append(object_record(f'/World/Props/Cone_{i}', 'mesh', (x, y, 0.002), quat_yaw(yaw), (1, 1, 1),
                                   1.5, True, 'traffic_cone/model.gltf', 'dynamic',
                                   'collisionMode=convex_hull renderMeshPath=traffic_cone/model.gltf'))
    (city_dir / 'rayrai_city.rscene').write_text('\n'.join(lines) + '\n')
    modules = sum(len(v) for v in facades.batches.values())
    print(f'{len(lots)} buildings, {modules} facade modules in {len(facades.batches)} batches, '
          f'{sum(len(v) for v in props.batches.values())} props, {len(cones)} cones')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: generate_city_rscene.py CITY_DIR')
    generate(sys.argv[1])

#!/usr/bin/env python3
"""Lay out the rayrai warehouse and write it as rayrai_warehouse.rscene.

The warehouse is a 60 m x 36 m steel portal-frame hall, 10 m to the eaves,
with precast concrete panels below corrugated metal cladding, a pitched roof
with skylights, and four dock doors on its east wall, two of them open to a
sunlit yard. Ten lines of blue and orange selective pallet racking (12 bays,
four beam levels) fill the west of the hall; the east is a dock area with
staging lanes, a forklift, a pallet jack and the usual clutter. Everything is
deterministic for a fixed seed.

The scene stores, as records RaiSim and rayrai read:
  * static, hidden box colliders for the walls, columns, rack frames, beams,
    floor loads and vehicles, and .rasset props with their own colliders;
  * dynamic cartons, crates, a wet-floor sign and traffic cones (convex hulls
    of their meshes);
  * instanced batches for the pallets, cartons and drums;
  * baked visual-only meshes: floor with markings, building shell, steel
    structure, skylights, racking with doors, signs and sprinklers, light
    fixtures and the yard; the forklifts, the pallet jack and closed doors;
  * the HDR sky, the sun, area lights over the aisles and the dock, render
    settings and the aisle camera.

The quadruped is not part of the file: the .rscene reader builds no
articulated systems, so rayrai_warehouse.cpp adds it after loading.

Usage: generate_warehouse_rscene.py WAREHOUSE_DIR   (a prepared rsc/warehouse directory)
"""
import json
import math
import random
import struct
import sys
from pathlib import Path

from generate_city_rscene import fmt, object_record, quat_yaw, quat_yaw_pitch

SEED = 5
# Hall: inner wall faces, eave height, roof rise at the ridge (y = 0).
HALF_X, HALF_Y = 30.0, 18.0
EAVE, RISE = 10.0, 1.6
WALL = 0.25                 # wall thickness
DADO = 3.0                  # height of the precast concrete panels
ROOF = 0.15                 # roof slab thickness
FRAMES = [-HALF_X + 7.5 * i for i in range(9)]
# Dock doors on the east wall: centre y, open.
DOORS = [(-13.0, False), (-4.5, True), (4.5, True), (13.0, False)]
DOOR_W, DOOR_H = 4.0, 4.6
# Skylight bands on each roof slope (distance from the ridge along y), and
# their length along x within each 7.5 m frame bay.
SKYLIGHT_BANDS = [(4.6, 5.8), (11.2, 12.4)]
SKYLIGHT_X = (1.5, 6.0)
# Selective pallet racking.
POST, POST_D = 0.09, 0.075  # upright post section along x and y
BEAM_LEN = 2.70             # clear span of a bay
PITCH = BEAM_LEN + POST
BAYS = 12
RACK_X0 = -27.0
RACK_X1 = RACK_X0 + BAYS * PITCH + POST
FRAME_D = 1.1               # upright frame depth
FLUE = 0.3                  # gap between back-to-back lines
UPRIGHT_H = 8.2
LEVELS = [1.8, 3.6, 5.4, 7.2]  # beam tops
BEAM_H, BEAM_W = 0.13, 0.05
PALLET = (1.2, 1.0, 0.144)
SLOTS = [RACK_X0 + POST + 0.1 + 0.6, RACK_X0 + POST + BEAM_LEN - 0.1 - 0.6]  # pallet centres in bay 0
FIXTURE_Z = 9.2
GROUND_EXTENT = 160.0
# Sun: elevation, and the direction it shines towards (azimuth from +x towards +y).
SUN_ELEVATION_DEG = 50.0
SUN_TOWARDS_DEG = 146.8
SKY_HDR = '../city/sky/kloofendal_48d_partly_cloudy_puresky_4k.hdr'
ASPHALT = '../city/textures/asphalt_02/asphalt_02'
# Aisle hero view: the robot stands in the middle aisle, the camera looks west down it.
ROBOT_XY = (4.4, 0.3)
CAMERA = dict(position=(9.6, -1.0, 1.3), yaw_deg=183.0, pitch_deg=2.0, vfov=55.0)


def rack_lines():
    """Rack lines south to north: (y of the south face, facing): facing is +1 for the aisle to the north."""
    aisle = (2 * HALF_Y - 0.9 - 2 * FRAME_D - 4 * (2 * FRAME_D + FLUE)) / 5
    y = -HALF_Y + 0.45
    lines = [(y, +1)]
    y += FRAME_D
    for _ in range(4):
        y += aisle
        lines.append((y, -1))
        y += FRAME_D + FLUE
        lines.append((y, +1))
        y += FRAME_D
    lines.append((y + aisle, -1))
    return lines, aisle


def aisle_centres():
    lines, aisle = rack_lines()
    return [y0 + FRAME_D + aisle / 2 for (y0, facing) in lines if facing > 0]


# ---------------------------------------------------------------- mesh building
def add(a, b, s=1.0):
    return tuple(x + s * y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def normalize(a):
    length = math.sqrt(sum(x * x for x in a))
    return tuple(x / length for x in a)


class Mesh:
    """Triangles grouped by material, written as one glTF file and buffer.

    Patches larger than max_cell are split into cells: rayrai v2.8.0
    interpolates a receiver's shadow-cascade depth per vertex, so a large
    triangle reaching behind the camera loses the shadows in front of it.
    """

    def __init__(self, materials, images, max_cell=3.0):
        self.materials, self.images, self.max_cell = materials, images, max_cell
        self.index = {m['name']: i for i, m in enumerate(materials)}
        self.parts = [dict(p=[], n=[], uv=[], i=[]) for _ in materials]

    def patch(self, material, origin, axis_u, axis_v, size_u, size_v, tile=(1.0, 1.0), uv0=(0.0, 0.0),
              subdivide=True):
        """Planar patch origin + s * axis_u + t * axis_v facing axis_u x axis_v.

        Texture coordinates are uv0 + (s / tile_u, -t / tile_v), so the top of
        an image is up on a wall.
        """
        cells_u = max(1, math.ceil(size_u / self.max_cell - 1e-9)) if subdivide and self.max_cell else 1
        cells_v = max(1, math.ceil(size_v / self.max_cell - 1e-9)) if subdivide and self.max_cell else 1
        part = self.parts[self.index[material]]
        normal = normalize(cross(axis_u, axis_v))
        base = len(part['p'])
        for j in range(cells_v + 1):
            for i in range(cells_u + 1):
                s, t = size_u * i / cells_u, size_v * j / cells_v
                part['p'].append(add(add(origin, axis_u, s), axis_v, t))
                part['n'].append(normal)
                part['uv'].append((uv0[0] + s / tile[0], uv0[1] - t / tile[1]))
        for j in range(cells_v):
            for i in range(cells_u):
                a = base + j * (cells_u + 1) + i
                part['i'].extend([a, a + 1, a + cells_u + 2, a, a + cells_u + 2, a + cells_u + 1])

    def triangle(self, material, a, b, c, uv):
        part = self.parts[self.index[material]]
        normal = normalize(cross(tuple(y - x for x, y in zip(a, b)), tuple(y - x for x, y in zip(a, c))))
        base = len(part['p'])
        for point in (a, b, c):
            part['p'].append(point)
            part['n'].append(normal)
            part['uv'].append(uv(point))
        part['i'].extend([base, base + 1, base + 2])

    def box_axes(self, material, centre, axes, half, tile=1.0, faces='xXyYzZ', fit_u=False):
        """Box with right-handed unit axes (ex, ey, ez) and half sizes; world-metre texture
        coordinates, or with fit_u the side faces span the texture's width once."""
        ex, ey, ez = axes
        hx, hy, hz = half

        def corner(sx, sy, sz):
            return add(add(add(centre, ex, sx * hx), ey, sy * hy), ez, sz * hz)

        neg = lambda v: tuple(-x for x in v)
        specs = {'X': (corner(1, -1, -1), ey, ez, 2 * hy, 2 * hz),
                 'x': (corner(-1, 1, -1), neg(ey), ez, 2 * hy, 2 * hz),
                 'Y': (corner(1, 1, -1), neg(ex), ez, 2 * hx, 2 * hz),
                 'y': (corner(-1, -1, -1), ex, ez, 2 * hx, 2 * hz),
                 'Z': (corner(-1, -1, 1), ex, ey, 2 * hx, 2 * hy),
                 'z': (corner(-1, 1, -1), ex, neg(ey), 2 * hx, 2 * hy)}
        for face in faces:
            origin, u, v, size_u, size_v = specs[face]
            if fit_u and face in 'xXyY':
                self.patch(material, origin, u, v, size_u, size_v, (size_u, tile), subdivide=False)
            else:
                self.patch(material, origin, u, v, size_u, size_v, (tile, tile))

    def box(self, material, centre, size, yaw=0.0, tile=1.0, faces='xXyYzZ', fit_u=False):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        self.box_axes(material, centre, ((c, s, 0), (-s, c, 0), (0, 0, 1)),
                      tuple(v / 2 for v in size), tile, faces, fit_u)

    def bar(self, material, start, end, width, height, up=(0, 0, 1), tile=1.0):
        """Box of the given cross-section from start to end; height is measured along up."""
        axis = tuple(e - s for s, e in zip(start, end))
        length = math.sqrt(sum(x * x for x in axis))
        ex = normalize(axis)
        ey = normalize(cross(up, ex))
        ez = cross(ex, ey)
        centre = tuple((s + e) / 2 for s, e in zip(start, end))
        self.box_axes(material, centre, (ex, ey, ez), (length / 2, width / 2, height / 2), tile)

    def cylinder(self, material, start, end, radius, sides=8):
        """Open prism of the given radius from start to end, with smooth normals."""
        axis = normalize(tuple(e - b for b, e in zip(start, end)))
        length = math.sqrt(sum((e - b) ** 2 for b, e in zip(start, end)))
        side = normalize(cross(axis, (0, 0, 1) if abs(axis[2]) < 0.9 else (1, 0, 0)))
        other = cross(axis, side)
        part = self.parts[self.index[material]]
        base = len(part['p'])
        for i in range(sides + 1):
            angle = 2 * math.pi * i / sides
            normal = tuple(math.cos(angle) * a + math.sin(angle) * b for a, b in zip(side, other))
            for end_point, v in ((start, 0.0), (end, length)):
                part['p'].append(add(end_point, normal, radius))
                part['n'].append(normal)
                part['uv'].append((i / sides, v))
        for i in range(sides):
            a = base + 2 * i
            part['i'].extend([a, a + 2, a + 3, a, a + 3, a + 1])

    def sign(self, material, centre, normal_x, width, height, cell):
        """A textured rectangle in a vertical plane facing +x (normal_x = 1) or -x, showing the
        image rows [cell[0], cell[1]] (texture v, top to bottom) the right way round."""
        x, y, z = centre
        direction = (0, normal_x, 0)
        origin = (x, y - normal_x * width / 2, z - height / 2)
        v0, v1 = cell
        self.patch(material, origin, direction, (0, 0, 1), width, height, (width, height / (v1 - v0)),
                   uv0=(0.0, v1), subdivide=False)

    def write(self, path):
        path = Path(path)
        data = bytearray()
        views, accessors, primitives = [], [], []

        def put(values, fmt_code, kind, target, bounds=False):
            while len(data) % 4:
                data.append(0)
            offset = len(data)
            packer = struct.Struct(fmt_code)
            for value in values:
                data.extend(packer.pack(*value) if isinstance(value, tuple) else packer.pack(value))
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
            primitives.append(dict(material=material, indices=put(part['i'], '<I', 'SCALAR', 34963),
                                   attributes=dict(POSITION=put(part['p'], '<3f', 'VEC3', 34962, True),
                                                   NORMAL=put(part['n'], '<3f', 'VEC3', 34962),
                                                   TEXCOORD_0=put(part['uv'], '<2f', 'VEC2', 34962))))
        document = dict(asset=dict(version='2.0', generator='generate_warehouse_rscene.py'),
                        scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0, name=path.stem)],
                        meshes=[dict(name=path.stem, primitives=primitives)],
                        materials=self.materials, buffers=[dict(uri=path.stem + '.bin', byteLength=len(data))],
                        bufferViews=views, accessors=accessors)
        used = {e for m in self.materials for e in m.get('extensions', {})}
        if used:
            document['extensionsUsed'] = sorted(used)
        if self.images:
            document['images'] = [dict(uri=uri) for uri in self.images]
            document['textures'] = [dict(source=i, sampler=0) for i in range(len(self.images))]
            document['samplers'] = [dict(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)]
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / (path.stem + '.bin')).write_bytes(data)
        path.write_text(json.dumps(document, indent=1) + '\n')
        return sum(len(part['i']) // 3 for part in self.parts)


def materials(specs):
    """glTF materials and images from (name, dict(color, roughness, metallic, maps, emissive, ...)).

    maps is (diffuse, normal, arm) image paths, any of them None; the arm image
    provides occlusion, roughness and metalness, scaled by the factors.
    """
    result, images = [], []

    def image(uri):
        if uri not in images:
            images.append(uri)
        return images.index(uri)

    for name, spec in specs:
        pbr = dict(baseColorFactor=list(spec.get('color', (1, 1, 1))) + [spec.get('alpha', 1.0)],
                   metallicFactor=spec.get('metallic', 0.0), roughnessFactor=spec.get('roughness', 0.8))
        material = dict(name=name, pbrMetallicRoughness=pbr)
        diffuse, normal, arm = (spec.get('maps') or (None, None, None))
        if diffuse:
            pbr['baseColorTexture'] = dict(index=image(diffuse))
        if normal:
            material['normalTexture'] = dict(index=image(normal), scale=spec.get('normal_scale', 1.0))
        if arm:
            pbr['metallicRoughnessTexture'] = dict(index=image(arm))
            material['occlusionTexture'] = dict(index=image(arm))
        if 'emissive' in spec:
            material['emissiveFactor'] = list(spec['emissive'])
            if spec.get('emissive_map'):
                material['emissiveTexture'] = dict(index=image(spec['emissive_map']))
            material['extensions'] = dict(KHR_materials_emissive_strength=dict(
                emissiveStrength=spec.get('emissive_strength', 1.0)))
        if spec.get('mask'):
            material.update(alphaMode='MASK', alphaCutoff=0.5)
        result.append(material)
    return result, images


def polyhaven(stem, resolution, prefix='../textures/'):
    return (f'{prefix}{stem}/{stem}_diff_{resolution}.jpg', f'{prefix}{stem}/{stem}_nor_gl_{resolution}.jpg',
            f'{prefix}{stem}/{stem}_arm_{resolution}.jpg')


PAINT = ('../textures/generated/painted_steel_diff.jpg', None, '../textures/generated/painted_steel_arm.jpg')
SPECS = {
    'floor': dict(maps=polyhaven('concrete_floor_worn_001', '2k'), roughness=1.0, normal_scale=0.6),
    'joint': dict(color=(0.05, 0.05, 0.05), roughness=0.95),
    'paint_yellow': dict(color=(0.86, 0.56, 0.02), roughness=0.5, mask=True,
                         maps=('../textures/generated/floor_paint_diff.png', None, None)),
    'paint_white': dict(color=(0.8, 0.8, 0.78), roughness=0.5, mask=True,
                        maps=('../textures/generated/floor_paint_diff.png', None, None)),
    'hazard': dict(maps=('../textures/generated/hazard_stripes_diff.jpg', None, PAINT[2])),
    'concrete_wall': dict(maps=polyhaven('concrete_wall_004', '2k'), color=(0.86, 0.9, 1.0)),
    'cladding': dict(maps=polyhaven('corrugated_iron', '2k'), color=(0.95, 0.95, 0.95)),
    'roof': dict(maps=polyhaven('corrugated_iron_02', '1k'), color=(0.85, 0.86, 0.88)),
    'steel': dict(maps=PAINT, color=(0.32, 0.34, 0.36)),
    'galvanized': dict(maps=PAINT, color=(0.62, 0.63, 0.64), metallic=0.85, roughness=0.9),
    'upright': dict(maps=('../textures/generated/rack_upright_diff.jpg', None, PAINT[2]),
                    color=(0.015, 0.085, 0.3)),
    'brace': dict(maps=PAINT, color=(0.015, 0.085, 0.3)),
    'beam': dict(maps=PAINT, color=(0.85, 0.2, 0.012)),
    'yellow_steel': dict(maps=PAINT, color=(0.92, 0.6, 0.02)),
    'label': dict(color=(0.85, 0.85, 0.82), roughness=0.6),
    'rubber': dict(color=(0.025, 0.025, 0.025), roughness=0.8),
    'fixture': dict(maps=PAINT, color=(0.7, 0.71, 0.72), metallic=0.8, roughness=0.8),
    'diffuser': dict(color=(1, 1, 1), roughness=0.4, emissive=(1.0, 0.97, 0.92), emissive_strength=12.0),
    'skylight': dict(color=(0.9, 0.92, 0.95), roughness=0.5, emissive=(0.92, 0.95, 1.0), emissive_strength=6.0),
    'exit_sign': dict(maps=('../textures/generated/exit_sign.png', None, None), roughness=0.3,
                      emissive=(1, 1, 1), emissive_map='../textures/generated/exit_sign.png', emissive_strength=3.0),
    'aisle_sign': dict(maps=('../textures/generated/aisle_signs.png', None, None), roughness=0.5),
    'door_steel': dict(maps=PAINT, color=(0.22, 0.3, 0.38)),
    'sprinkler': dict(maps=PAINT, color=(0.55, 0.03, 0.02)),
    'asphalt': dict(maps=(f'../{ASPHALT}_diff_2k.jpg', f'../{ASPHALT}_nor_gl_2k.jpg', f'../{ASPHALT}_arm_2k.jpg')),
}


def mesh(names, max_cell=3.0):
    specs, images = materials([(name, SPECS[name]) for name in names])
    return Mesh(specs, images, max_cell)


# ---------------------------------------------------------------- building
def roof_z(y):
    """Underside of the roof slab; beyond the walls the slope continues to the eave overhang."""
    return EAVE + RISE * (1 - abs(y) / HALF_Y)


def is_skylight(x, d):
    """Whether the roof point at x and distance d from the ridge is in a skylight opening."""
    if not -HALF_X <= x < HALF_X:
        return False
    bay = FRAMES[int((x + HALF_X) // 7.5)]
    return (any(a <= d <= b for a, b in SKYLIGHT_BANDS) and
            bay + SKYLIGHT_X[0] <= x <= bay + SKYLIGHT_X[1])


def floor_mesh(rng):
    floor = mesh(['floor', 'joint', 'paint_yellow', 'paint_white', 'hazard'])
    floor.patch('floor', (-HALF_X, -HALF_Y, 0), (1, 0, 0), (0, 1, 0), 2 * HALF_X, 2 * HALF_Y, (4.0, 4.0))
    # Saw-cut joints every 6 m.
    for x in range(-24, 30, 6):
        floor.patch('joint', (x - 0.004, -HALF_Y, 0.001), (1, 0, 0), (0, 1, 0), 0.008, 2 * HALF_Y,
                    subdivide=False)
    for y in range(-12, 18, 6):
        floor.patch('joint', (-HALF_X, y - 0.004, 0.001), (1, 0, 0), (0, 1, 0), 2 * HALF_X, 0.008,
                    subdivide=False)
    z = 0.003

    def line(material, x0, y0, x1, y1):
        floor.patch(material, (x0, y0, z), (1, 0, 0), (0, 1, 0), x1 - x0, y1 - y0, (1.0, 1.0),
                    uv0=(rng.random(), rng.random()))

    lines, _ = rack_lines()
    for y0, facing in lines:
        edge = y0 + FRAME_D + 0.15 if facing > 0 else y0 - 0.15
        line('paint_yellow', RACK_X0 - 0.2, edge - 0.05, RACK_X1 + 0.2, edge + 0.05)
    # Staging lanes in front of the dock doors, and a walkway along the dock wall.
    for door_y, _ in DOORS:
        for side in (-1, 1):
            y = door_y + side * 1.45
            line('paint_yellow', 14.0, y - 0.05, 27.0, y + 0.05)
        line('paint_yellow', 14.0, door_y - 1.5, 14.1, door_y + 1.5)
    line('paint_white', 27.8, -HALF_Y + 0.4, 27.9, HALF_Y - 0.4)
    # Hazard strips at the dock door thresholds.
    for door_y, _ in DOORS:
        floor.patch('hazard', (HALF_X - 0.35, door_y - DOOR_W / 2, z), (1, 0, 0), (0, 1, 0), 0.35, DOOR_W,
                    (0.5, 0.5), subdivide=False)
    # Hazardous-goods bay in the south-east corner.
    x0, y0, x1, y1 = 21.0, -HALF_Y + 0.4, 27.0, -HALF_Y + 3.4
    for a, b, c, d in [(x0, y0, x1, y0 + 0.1), (x0, y1 - 0.1, x1, y1), (x0, y0, x0 + 0.1, y1)]:
        line('paint_yellow', a, b, c, d)
    return floor


def wall_with_openings(shell, x_wall, openings):
    """East wall at x = x_wall (inner face): concrete panels, cladding above, door openings."""
    edges = sorted({-HALF_Y - WALL, HALF_Y + WALL} | {y + s * DOOR_W / 2 for y, _ in openings for s in (-1, 1)})
    for y0, y1 in zip(edges, edges[1:]):
        door = any(abs((y0 + y1) / 2 - y) < DOOR_W / 2 for y, _ in openings)
        z0 = DOOR_H if door else 0.0
        for material, za, zb in (('concrete_wall', z0, max(z0, DADO)), ('cladding', max(z0, DADO), EAVE)):
            if zb <= za:
                continue
            # Inner face looks towards -x, outer face towards +x.
            shell.patch(material, (x_wall, y1, za), (0, -1, 0), (0, 0, 1), y1 - y0, zb - za, (3.0, 3.0),
                        uv0=(-y1 / 3.0, 0))
            shell.patch('cladding', (x_wall + WALL, y0, za), (0, 1, 0), (0, 0, 1), y1 - y0, zb - za,
                        (3.0, 3.0), uv0=(y0 / 3.0, 0))
    for y, _ in openings:
        # Jambs facing into the opening, and the head facing down.
        shell.patch('galvanized', (x_wall + WALL, y - DOOR_W / 2, 0), (-1, 0, 0), (0, 0, 1), WALL, DOOR_H,
                    subdivide=False)
        shell.patch('galvanized', (x_wall, y + DOOR_W / 2, 0), (1, 0, 0), (0, 0, 1), WALL, DOOR_H,
                    subdivide=False)
        shell.patch('galvanized', (x_wall, y - DOOR_W / 2, DOOR_H), (0, 1, 0), (1, 0, 0), DOOR_W, WALL,
                    subdivide=False)


def shell_mesh():
    """Walls and roof slab: the closed box that keeps the sun out except through openings."""
    shell = mesh(['concrete_wall', 'cladding', 'roof', 'galvanized'])
    # Long walls: inner faces at y = -HALF_Y (facing +y) and y = +HALF_Y (facing -y).
    for side in (-1, 1):
        y_in, y_out = side * HALF_Y, side * (HALF_Y + WALL)
        # Inner faces look into the hall, outer faces away from it.
        x_start = HALF_X if side < 0 else -HALF_X
        direction = (-1, 0, 0) if side < 0 else (1, 0, 0)
        for material, za, zb in (('concrete_wall', 0.0, DADO), ('cladding', DADO, EAVE)):
            shell.patch(material, (x_start, y_in, za), direction, (0, 0, 1), 2 * HALF_X, zb - za, (3.0, 3.0))
            shell.patch('cladding', (-x_start, y_out, za), tuple(-v for v in direction), (0, 0, 1),
                        2 * HALF_X, zb - za, (3.0, 3.0))
    # West wall (x = -HALF_X, facing +x) with its gable.
    x = -HALF_X
    for material, za, zb in (('concrete_wall', 0.0, DADO), ('cladding', DADO, EAVE)):
        shell.patch(material, (x, -HALF_Y, za), (0, 1, 0), (0, 0, 1), 2 * HALF_Y, zb - za, (3.0, 3.0))
        shell.patch('cladding', (x - WALL, HALF_Y + WALL, za), (0, -1, 0), (0, 0, 1), 2 * HALF_Y + 2 * WALL,
                    zb - za, (3.0, 3.0))
    wall_with_openings(shell, HALF_X, DOORS)
    for x_in, x_out, sign in ((-HALF_X, -HALF_X - WALL, 1), (HALF_X, HALF_X + WALL, -1)):
        uv = lambda p: (p[1] / 3.0, -p[2] / 3.0)
        top = (x_in, 0.0, EAVE + RISE)
        a, b = (x_in, -HALF_Y, EAVE), (x_in, HALF_Y, EAVE)
        shell.triangle('cladding', *((a, b, top) if sign > 0 else (b, a, top)), uv)
        a, b, top = (x_out, -HALF_Y - WALL, EAVE), (x_out, HALF_Y + WALL, EAVE), (x_out, 0.0, EAVE + RISE)
        shell.triangle('cladding', *((b, a, top) if sign > 0 else (a, b, top)), uv)
    # Roof slab, both slopes, with skylight openings. Cells are cut at the
    # frame bays and skylight bands so that each opening is a whole cell.
    x_edges = sorted({-HALF_X - WALL, HALF_X + WALL} | set(FRAMES[1:-1]) |
                     {f + d for f in FRAMES[:-1] for d in SKYLIGHT_X})
    d_edges = sorted({0.0, HALF_Y + WALL} | {d for band in SKYLIGHT_BANDS for d in band})
    for side in (-1, 1):
        for d0, d1 in zip(d_edges, d_edges[1:]):
            for x0, x1 in zip(x_edges, x_edges[1:]):
                if is_skylight((x0 + x1) / 2, (d0 + d1) / 2):
                    continue
                ya, yb = side * d0, side * d1
                across = (0, yb - ya, roof_z(yb) - roof_z(ya))
                length = math.sqrt(sum(c * c for c in across))
                for offset, top in ((0.0, False), (ROOF, True)):
                    # u x across points up on the top and down on the underside.
                    if (side > 0) == top:
                        origin, u = (x0, ya, roof_z(ya) + offset), (1, 0, 0)
                    else:
                        origin, u = (x1, ya, roof_z(ya) + offset), (-1, 0, 0)
                    shell.patch('roof', origin, u, normalize(across), x1 - x0, length, (2.0, 2.0))
    return shell


def structure_mesh():
    """Portal frames, purlins, eave beams, wind bracing and the wall columns."""
    steel = mesh(['steel', 'galvanized'], max_cell=None)
    slope = math.atan2(RISE, HALF_Y)
    for x in FRAMES:
        for side in (-1, 1):
            y_col = side * (HALF_Y - 0.2)
            # Column: an I-section of two flanges and a web.
            for dx in (-0.14, 0.14):
                steel.box('steel', (x + dx, y_col, EAVE / 2), (0.02, 0.4, EAVE))
            steel.box('steel', (x, y_col, EAVE / 2), (0.28, 0.012, EAVE))
            # Rafter from the column head to the ridge, 0.6 m deep at the eave.
            start = (x, y_col, EAVE - 0.45)
            end = (x, 0.0, roof_z(0) - 0.45)
            up = (0, -side * math.sin(slope), math.cos(slope))
            steel.bar('steel', start, end, 0.012, 0.55, up)
            for offset in (-0.27, 0.27):
                steel.bar('steel', add(start, up, offset), add(end, up, offset), 0.2, 0.02, up)
            # Haunch plate at the knee.
            steel.box('steel', (x, side * (HALF_Y - 0.9), EAVE - 0.75), (0.012, 1.4, 0.9))
    for side in (-1, 1):
        # Purlins every 1.5 m along the slope, eave beams, and sag rods.
        for k in range(13):
            d = 0.35 + k * 1.45
            if d > HALF_Y:
                break
            y = side * d
            steel.box('galvanized', (0, y, roof_z(y) - 0.11), (2 * HALF_X, 0.06, 0.2))
        steel.box('steel', (0, side * (HALF_Y - 0.15), EAVE - 0.15), (2 * HALF_X, 0.2, 0.3))
    # Wind bracing rods in the end bays of the roof.
    for x0 in (FRAMES[0], FRAMES[-2]):
        for side in (-1, 1):
            for a, b in ((0.0, HALF_Y - 0.3), (HALF_Y - 0.3, 0.0)):
                start = (x0, side * a, roof_z(side * a) - 0.25)
                end = (x0 + 7.5, side * b, roof_z(side * b) - 0.25)
                steel.bar('galvanized', start, end, 0.025, 0.025)
    # Gable columns.
    for x in (-HALF_X + 0.2, HALF_X - 0.2):
        for y in (-12.0, -6.0, 0.0, 6.0, 12.0):
            if x > 0 and any(abs(y - d) < DOOR_W / 2 + 0.3 for d, _ in DOORS):
                continue
            steel.box('steel', (x, y, roof_z(y) / 2), (0.3, 0.25, roof_z(y)))
    return steel


def skylight_mesh():
    sky = mesh(['skylight'], max_cell=None)
    for side in (-1, 1):
        for a, b in SKYLIGHT_BANDS:
            for f in FRAMES[:-1]:
                x0, x1 = f + SKYLIGHT_X[0], f + SKYLIGHT_X[1]
                ya, yb = side * a, side * b
                za, zb = roof_z(ya) + ROOF / 2, roof_z(yb) + ROOF / 2
                across = (0, yb - ya, zb - za)
                length = math.sqrt(sum(c * c for c in across))
                origin, u = ((x1, ya, za), (-1, 0, 0)) if side > 0 else ((x0, ya, za), (1, 0, 0))
                sky.patch('skylight', origin, u, normalize(across), x1 - x0, length, (1.0, 1.0))
    return sky


def door_hardware(out):
    """Galvanized coil housings above the open doors, guide rails and yellow bollards."""
    for y, opened in DOORS:
        if opened:
            out.box('galvanized', (HALF_X - 0.3, y, DOOR_H + 0.35), (0.6, DOOR_W + 0.3, 0.7))
        for side in (-1, 1):
            out.box('galvanized', (HALF_X - 0.08, y + side * (DOOR_W / 2 + 0.06), DOOR_H / 2),
                    (0.12, 0.1, DOOR_H))
            out.box('yellow_steel', (HALF_X - 0.6, y + side * (DOOR_W / 2 + 0.25), 0.6), (0.16, 0.16, 1.2))
            out.box('rubber', (HALF_X - 0.12, y + side * (DOOR_W / 2 + 0.2), DOOR_H / 2 + 0.2),
                    (0.2, 0.25, DOOR_H + 0.4))


def personnel_doors(out):
    """Steel doors with push bars in the west wall, an exit sign over each."""
    for x, y, facing in ((-HALF_X, 1.3, 1), (-HALF_X, -9.5, 1)):
        face = x + facing * 0.03
        out.box('galvanized', (face, y, 1.08), (0.06, 1.16, 2.16))
        out.box('door_steel', (face + facing * 0.012, y, 1.05), (0.04, 1.0, 2.08))
        out.box('galvanized', (face + facing * 0.06, y, 1.0), (0.03, 0.8, 0.05))
        out.box('rubber', (x + facing * 0.012, y, 2.45), (0.02, 0.44, 0.2))
        out.sign('exit_sign', (x + facing * 0.025, y, 2.45), facing, 0.4, 0.15, (0.0, 1.0))


def aisle_signs(out, aisles):
    """A double-sided sign hanging over both ends of each aisle."""
    for number, y in enumerate(aisles):
        cell = (number / 5, (number + 1) / 5)
        for x in (RACK_X0 - 0.8, RACK_X1 + 0.8):
            z = 7.0
            out.box('galvanized', (x, y, z), (0.03, 1.66, 0.46))
            out.sign('aisle_sign', (x + 0.016, y, z), 1, 1.6, 0.4, cell)
            out.sign('aisle_sign', (x - 0.016, y, z), -1, 1.6, 0.4, cell)
            for side in (-0.7, 0.7):
                top = roof_z(y + side) - 0.25
                out.box('galvanized', (x, y + side, (z + 0.23 + top) / 2), (0.006, 0.006, top - z - 0.23))


def sprinklers(out):
    """Fire sprinkler mains along the hall near the ridge, branch lines down both slopes."""
    for side in (-1, 1):
        y_main = side * 2.6
        z_main = roof_z(y_main) - 0.95
        out.cylinder('sprinkler', (-HALF_X + 0.5, y_main, z_main), (HALF_X - 0.5, y_main, z_main), 0.075)
        for x in [f + 1.5 + 1.5 * k for f in FRAMES[:-1] for k in range(4)]:
            start = (x, y_main, z_main)
            end = (x, side * (HALF_Y - 0.8), roof_z(HALF_Y - 0.8) - 0.7)
            out.cylinder('sprinkler', start, end, 0.025, 6)
            for k in range(1, 6):
                d = 2.6 + k * 2.9
                if d > HALF_Y - 0.8:
                    break
                t = (d - 2.6) / (HALF_Y - 0.8 - 2.6)
                head = tuple(a + t * (b - a) for a, b in zip(start, end))
                out.box('galvanized', (head[0], head[1], head[2] - 0.05), (0.03, 0.03, 0.08))


# ---------------------------------------------------------------- racking
def racking_mesh():
    rack = mesh(['upright', 'brace', 'beam', 'galvanized', 'yellow_steel', 'label', 'hazard', 'rubber',
                 'door_steel', 'exit_sign', 'aisle_sign', 'sprinkler'], max_cell=None)
    lines, _ = rack_lines()
    for y0, facing in lines:
        y1 = y0 + FRAME_D
        front = y1 if facing > 0 else y0
        for k in range(BAYS + 1):
            x = RACK_X0 + POST / 2 + k * PITCH
            for y in (y0 + POST_D / 2, y1 - POST_D / 2):
                rack.box('upright', (x, y, UPRIGHT_H / 2), (POST, POST_D, UPRIGHT_H), tile=0.4, fit_u=True)
                rack.box('galvanized', (x, y, 0.004), (0.13, 0.12, 0.008))
            # Frame bracing: horizontals at the bottom and top, diagonals between.
            ya, yb = y0 + POST_D, y1 - POST_D
            heights = [0.15 + i * (UPRIGHT_H - 0.3) / 7 for i in range(8)]
            for h in (heights[0], heights[-1]):
                rack.bar('brace', (x, ya, h), (x, yb, h), 0.035, 0.035)
            for i, (h0, h1) in enumerate(zip(heights, heights[1:])):
                a, b = (ya, yb) if i % 2 == 0 else (yb, ya)
                rack.bar('brace', (x, a, h0), (x, b, h1), 0.035, 0.035)
            # Upright protector on the aisle-side post.
            guard_y = front + (0.04 if facing > 0 else -0.04)
            rack.box('yellow_steel', (x, guard_y, 0.2), (0.16, 0.05, 0.4))
        # Beams with connectors and location labels, on both faces of the frame.
        for level in LEVELS:
            for y in (y0 + BEAM_W / 2, y1 - BEAM_W / 2):
                rack.box('beam', ((RACK_X0 + RACK_X1) / 2, y, level - BEAM_H / 2),
                         (RACK_X1 - RACK_X0 - POST, BEAM_W, BEAM_H))
            label_y = front + (0.002 if facing > 0 else -0.002)
            for k in range(BAYS):
                for slot in SLOTS:
                    rack.box('label', (slot + k * PITCH, label_y, level - BEAM_H / 2), (0.16, 0.002, 0.06))
        # End-of-row barriers at both ends of each line.
        for x in (RACK_X0 - 0.35, RACK_X1 + 0.35):
            rack.box('hazard', (x, (y0 + y1) / 2, 0.3), (0.1, FRAME_D + 0.1, 0.25), tile=0.5)
            for y in (y0 + 0.05, y1 - 0.05):
                rack.box('yellow_steel', (x, y, 0.2), (0.08, 0.08, 0.4))
    return rack


def fixtures_mesh(aisles):
    """Linear LED high-bays over the aisles and over the dock area, hung from the purlins."""
    fx = mesh(['fixture', 'diffuser', 'galvanized'], max_cell=None)
    spots = [(RACK_X0 + POST / 2 + PITCH * (i + 0.5), y, 0) for y in aisles for i in range(BAYS)]
    spots += [(x, y, 90) for x in (11.0, 17.0, 23.0, 28.0) for y in (-13.5, -7.5, -1.5, 4.5, 10.5, 15.5)]
    for x, y, yaw in spots:
        fx.box('fixture', (x, y, FIXTURE_Z + 0.04), (1.25, 0.34, 0.08), yaw)
        fx.box('diffuser', (x, y, FIXTURE_Z - 0.002), (1.15, 0.26, 0.004), yaw, faces='z')
        for end in (-0.5, 0.5):
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            top = roof_z(y) - 0.22
            fx.box('galvanized', (x + c * end, y + s * end, (FIXTURE_Z + top) / 2), (0.006, 0.006, top - FIXTURE_Z))
    return fx, spots


def yard_mesh():
    """Asphalt yard outside the dock doors, and the facing wall of a neighbouring building."""
    yard = mesh(['asphalt', 'cladding', 'concrete_wall', 'galvanized'])
    x0 = HALF_X + WALL
    yard.patch('asphalt', (x0, -40.0, 0), (1, 0, 0), (0, 1, 0), 40.0, 80.0, (4.0, 4.0))
    for x in range(int(x0) + 40 - 0, int(GROUND_EXTENT), 40):
        yard.patch('asphalt', (x, -GROUND_EXTENT, 0), (1, 0, 0), (0, 1, 0), 40.0, 2 * GROUND_EXTENT,
                   (4.0, 4.0), subdivide=False)
    for y0, y1 in ((-GROUND_EXTENT, -40.0), (40.0, GROUND_EXTENT)):
        yard.patch('asphalt', (x0, y0, 0), (1, 0, 0), (0, 1, 0), 40.0, y1 - y0, (4.0, 4.0), subdivide=False)
    # The neighbour's wall 34 m across the yard, facing -x.
    xn = HALF_X + 34.0
    yard.patch('concrete_wall', (xn, 45.0, 0), (0, -1, 0), (0, 0, 1), 90.0, 2.5, (3.0, 3.0))
    yard.patch('cladding', (xn, 45.0, 2.5), (0, -1, 0), (0, 0, 1), 90.0, 9.5, (3.0, 3.0))
    yard.box('galvanized', (xn - 0.15, 0, 12.1), (0.3, 90.0, 0.2))
    return yard


# ---------------------------------------------------------------- loads and props
class Batches:
    """Instances per mesh: (x, y, z, yaw degrees, sx, sy, sz, colour weight).

    The weight blends a batch's colour from white to TINTS[kind], which
    multiplies the texture: cartons and pallets of one load share a shade.
    """

    def __init__(self):
        self.batches = {}

    def add(self, mesh_path, x, y, z, yaw=0.0, scale=(1.0, 1.0, 1.0), weight=0.0):
        self.batches.setdefault(mesh_path, []).append((x, y, z, yaw, *scale, weight))


TINTS = {'cartons': (0.68, 0.6, 0.5), 'pallets': (0.62, 0.6, 0.58)}


def carton_load(rng, out, cartons, x, y, z, max_height, kind=None):
    """A pallet with a stack of cartons; returns the top height of the load."""
    pallet = 'pallets/pallet_slatted.gltf' if rng.random() < 0.7 else 'pallets/pallet_plywood.gltf'
    out.add(pallet, x, y, z, rng.choice((0, 180)) + rng.uniform(-1.2, 1.2), weight=rng.random() ** 2)
    z += PALLET[2]
    shade = rng.random() ** 1.5
    name = kind or rng.choice(list(cartons))
    size = cartons[name]
    scale = rng.uniform(1.25, 1.6)
    lx, ly, lz = (v * scale for v in size)
    turned = rng.random() < 0.5
    if turned:
        lx, ly = ly, lx
    nx, ny = max(1, int(1.18 // lx)), max(1, int(1.02 // ly))
    layers = max(1, int(rng.uniform(0.55, 1.0) * max_height // lz))
    gap_x, gap_y = (1.18 - nx * lx) / max(nx, 1), (1.02 - ny * ly) / max(ny, 1)
    top = z
    for layer in range(layers):
        for i in range(nx):
            for j in range(ny):
                # Picked loads lose cartons from the top layer.
                if layer == layers - 1 and layers > 1 and rng.random() < 0.25:
                    continue
                bx = x - 0.59 + gap_x / 2 + (i + 0.5) * (lx + gap_x) + rng.uniform(-0.008, 0.008)
                by = y - 0.51 + gap_y / 2 + (j + 0.5) * (ly + gap_y) + rng.uniform(-0.008, 0.008)
                yaw = (90 if turned else 0) + rng.choice((0, 180)) + rng.uniform(-1.5, 1.5)
                out.add(f'cartons/{name}.gltf', bx, by, z + layer * lz, yaw, (scale, scale, scale),
                        min(1.0, max(0.0, shade + rng.uniform(-0.08, 0.08))))
                top = max(top, z + (layer + 1) * lz)
    return top


def drum_load(rng, out, x, y, z):
    pallet = 'pallets/pallet_slatted.gltf'
    out.add(pallet, x, y, z, 0, weight=rng.random() ** 2)
    drum = rng.choice(['props/Barrel_01/model.gltf', 'props/barrel_03/model.gltf'])
    for dx in (-0.3, 0.3):
        for dy in (-0.25, 0.25):
            if rng.random() < 0.9:
                out.add(drum, x + dx, y + dy, z + PALLET[2], rng.uniform(0, 360), (0.95, 0.95, 0.95))
    return z + PALLET[2] + 0.9


def fill_racks(rng, out, cartons):
    """Pallet loads in every rack position, a few left empty. Returns floor loads (x, y, top)."""
    lines, _ = rack_lines()
    floor_loads = []
    for y0, facing in lines:
        y = y0 + FRAME_D / 2
        for k in range(BAYS):
            for slot in SLOTS:
                x = slot + k * PITCH
                for level_index, base in enumerate([0.0] + LEVELS):
                    if rng.random() < 0.12:
                        continue
                    next_beam = (LEVELS[level_index] - BEAM_H) if level_index < len(LEVELS) else 8.9
                    room = next_beam - base - PALLET[2] - 0.1
                    if level_index <= 1 and rng.random() < 0.06:
                        top = drum_load(rng, out, x, y, base)
                    else:
                        top = carton_load(rng, out, cartons, x, y, base, min(room, 1.45))
                    if level_index == 0:
                        floor_loads.append((x, y, top))
    return floor_loads


def dock_area(rng, out, cartons):
    """Staging lanes, empty pallet stacks, the hazardous-goods bay and loose props."""
    floor_loads = []
    for door_y, opened in DOORS:
        count = rng.randint(4, 8) if opened else rng.randint(2, 5)
        for i in range(count):
            for side in (-1, 1):
                if rng.random() < 0.85:
                    x, y = 15.0 + i * 1.5, door_y + side * 0.65
                    top = carton_load(rng, out, cartons, x, y, 0.0, rng.uniform(0.9, 1.5))
                    floor_loads.append((x, y, top))
    # Empty pallet stacks along the north wall.
    for i, x in enumerate((12.0, 13.4, 14.8)):
        height = rng.randint(6, 12)
        for level in range(height):
            out.add('pallets/pallet_slatted.gltf', x, HALF_Y - 1.2, level * PALLET[2],
                    90 + rng.uniform(-2, 2), weight=rng.random() ** 2)
        floor_loads.append((x, HALF_Y - 1.2, height * PALLET[2]))
    # Drums on pallets in the hazardous-goods bay.
    for x in (22.0, 23.5, 25.0):
        top = drum_load(rng, out, x, -HALF_Y + 1.9, 0.0)
        floor_loads.append((x, -HALF_Y + 1.9, top))
    return floor_loads


def props_layout(rng):
    """Single props as (name, mesh, x, y, z, yaw, mode): mode is static (.rasset), visual or dynamic."""
    props = [
        ('Forklift', 'forklift/model.gltf', 11.6, -4.6, 0.0, 175.0, 'vehicle'),
        ('Forklift_1', 'forklift/model.gltf', -13.5, 7.4, 0.0, 2.0, 'vehicle'),
        ('PalletJack', 'pallet_jack/model.gltf', 18.5, 6.4, 0.0, 200.0, 'vehicle'),
        ('PackingDesk', 'props/metal_office_desk/model.rasset', 21.0, HALF_Y - 1.0, 0.0, 0.0, 'static'),
        ('ToolCart', 'props/tool_cart/model.rasset', 25.6, HALF_Y - 1.1, 0.0, 8.0, 'static'),
        ('HandTruck', 'props/hand_truck/model.rasset', 27.2, -9.2, 0.0, 250.0, 'static'),
    ]
    for i, x in enumerate((16.5, 17.7, 18.9)):
        props.append((f'Shelf_{i}', 'props/steel_frame_shelves_01/model.rasset', x, HALF_Y - 0.4, 0.0, 0.0,
                      'static'))
    for i, (x, y) in enumerate([(19.6, -HALF_Y + 0.35), (19.9, -HALF_Y + 0.35), (20.2, -HALF_Y + 0.35)]):
        props.append((f'Propane_{i}', 'props/propane_tank/model.rasset', x, y, 0.0, rng.uniform(0, 360),
                      'static'))
    for i, x in enumerate(FRAMES[1:-1]):
        for side in (-1, 1):
            if (i + (side > 0)) % 2:
                props.append((f'Extinguisher_{i}_{side}', 'props/korean_fire_extinguisher_01/model.rasset',
                              x + 0.5, side * (HALF_Y - 0.45), 0.0, 0.0 if side > 0 else 180.0, 'static'))
    for i, (x, y, yaw) in enumerate([(HALF_X - 0.3, -HALF_Y + 0.6, 135), (-HALF_X + 0.3, HALF_Y - 0.6, -45),
                                     (HALF_X - 0.3, HALF_Y - 0.6, -135)]):
        props.append((f'Camera_{i}', 'props/security_camera_01/model.gltf', x, y, 6.0, yaw, 'visual'))
    # Loose objects in the aisle near the robot, free to be pushed around.
    props += [
        ('Carton_0', 'props/cardboard_box_01/model.gltf', -1.8, -1.55, 0.0, 12.0, 'dynamic'),
        ('Carton_1', 'props/cardboard_box_01/model.gltf', -2.25, -1.35, 0.0, 70.0, 'dynamic'),
        ('Carton_2', 'props/cardboard_box_01/model.gltf', -2.0, -1.45, 0.345, 40.0, 'dynamic'),
        ('Crate_0', 'props/plastic_crate_03/model.gltf', 4.6, 1.55, 0.0, 85.0, 'dynamic'),
        ('Crate_1', 'props/plastic_crate_03/model.gltf', 5.1, 1.6, 0.0, 97.0, 'dynamic'),
        ('WetFloorSign', 'props/WetFloorSign_01/model.gltf', -6.5, 1.7, 0.0, 120.0, 'dynamic'),
        ('Carton_3', 'props/cardboard_box_01/model.gltf', 20.5, HALF_Y - 1.05, 0.79, 80.0, 'dynamic'),
        ('Crate_2', 'props/plastic_crate_03/model.gltf', 21.6, HALF_Y - 0.95, 0.79, 3.0, 'dynamic'),
    ]
    # Traffic cones of the city example around a spill in the dock area.
    for i, (x, y) in enumerate([(12.9, -9.4), (13.8, -10.3), (14.6, -9.5), (13.7, -8.6)]):
        props.append((f'Cone_{i}', '../city/traffic_cone/model.gltf', x, y, 0.0, rng.uniform(0, 360), 'dynamic'))
    return props


# ---------------------------------------------------------------- records
def instanced_record(path, mesh_path, instances, keys=''):
    values = ';'.join(','.join(fmt(v) for v in (x, y, z, *quat_yaw(yaw), sx, sy, sz))
                      for x, y, z, yaw, sx, sy, sz, _ in instances)
    tint = next((TINTS[kind] for kind in TINTS if mesh_path.startswith(kind + '/')), (1, 1, 1))
    weights = ','.join(fmt(w, 3) for *_, w in instances) if tint != (1, 1, 1) else ''
    return (f'instanced_visual {path} mesh meshPath={mesh_path} size=1,1,1 colorA=1,1,1,1 '
            f'colorB={",".join(fmt(c) for c in tint)},1 instances={values} '
            + (f'colorWeights={weights} ' if weights else '') +
            'castShadows=true automaticMeshLod=true' + (' ' + keys if keys else ''))


def visual_record(path, mesh_path, keys=''):
    return object_record(path, 'mesh', (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 1, True, mesh_path, 'visual_only',
                         f'renderMeshPath={mesh_path} visualUseMeshColor=false' + (' ' + keys if keys else ''))


def box_collider(path, centre, size, yaw=0.0):
    return object_record(path, 'box', centre, quat_yaw(yaw), size, 1, False, '-', 'static')


def generate(warehouse_dir):
    warehouse_dir = Path(warehouse_dir)
    rng = random.Random(SEED)
    cartons = json.loads((warehouse_dir / 'cartons/sizes.json').read_text())
    lines, aisle = rack_lines()
    aisles = aisle_centres()

    triangles = {}
    triangles['floor'] = floor_mesh(rng).write(warehouse_dir / 'building/floor.gltf')
    triangles['shell'] = shell_mesh().write(warehouse_dir / 'building/shell.gltf')
    triangles['structure'] = structure_mesh().write(warehouse_dir / 'building/structure.gltf')
    triangles['skylights'] = skylight_mesh().write(warehouse_dir / 'building/skylights.gltf')
    rack = racking_mesh()
    door_hardware(rack)
    personnel_doors(rack)
    aisle_signs(rack, aisles)
    sprinklers(rack)
    triangles['racking'] = rack.write(warehouse_dir / 'building/racking.gltf')
    fixtures, spots = fixtures_mesh(aisles)
    triangles['fixtures'] = fixtures.write(warehouse_dir / 'building/fixtures.gltf')
    triangles['yard'] = yard_mesh().write(warehouse_dir / 'building/yard.gltf')

    loads = Batches()
    floor_loads = fill_racks(rng, loads, cartons)
    floor_loads += dock_area(rng, loads, cartons)
    props = props_layout(rng)

    elevation, towards = math.radians(SUN_ELEVATION_DEG), math.radians(SUN_TOWARDS_DEG)
    sun_dir = (math.cos(elevation) * math.cos(towards), math.cos(elevation) * math.sin(towards),
               -math.sin(elevation))
    cam = CAMERA
    lines_out = [
        'raisim_engine_scene 2',
        '# rayrai warehouse: Poly Haven props and textures (CC0), Sketchfab forklift, pallet jack, pallets',
        '# and cartons (CC BY 4.0, see ATTRIBUTION.md), generated racking and building.',
        '# Generated by examples/tools/generate_warehouse_rscene.py; the quadruped is added by rayrai_warehouse.cpp.',
        'time_step 0.002',
        'gravity 0 0 -9.81',
        # raisim::World's own solver defaults (see the .rscene documentation).
        'solver 150 1e-08 1.5 accurate erp2=0.002 defaultRestitutionThreshold=0.01 '
        'defaultStaticFrictionVelocityThreshold=1 fixedContactSolverIterationOrder=true',
        'asset_root .',
        f'environment 0.62 0.7 0.82 0.32 0.315 0.31 0.0035 true false 10 {SKY_HDR} '
        'backgroundMode=hdr fogColor=0.74,0.74,0.72 exposure=1 gamma=2.2 bloomIntensity=0.08 '
        'bloomThreshold=1.5 bloomRadius=4 shadowMapSize=4096 shadowedLightBudget=1 colorMode=aces_approx '
        'pbrEnvironmentLightingTint=0.92,0.9,0.86 pbrEnvironmentIntensity=0.55 fxaa=true ssao=true',
        'rayrai_render preset=high custom=true colorMode=aces_approx viewerMsaaSamples=4 '
        'shadowResolution=4096 shadowBias=0.0005 shadowStrength=1 shadowPcfRadius=3 '
        'directionalShadowCascadeCount=2 directionalShadowCascadeLambda=0.7 '
        'directionalShadowCascadeMaxDistance=70 highFidelityPbr=true pbrToneMapping=true '
        'pbrExposure=1 pbrEnvironmentIntensity=0.55 addViewerFillLights=false reflectiveGround=false '
        'contactShadows=true screenSpaceReflections=true screenSpaceReflectionStrength=0.3',
        f'light /World/Lights/Sun {" ".join(fmt(v) for v in sun_dir)} 0.9 type=directional shadows=true '
        'color=1,0.95,0.86 ambientColor=0,0,0 shadowResolution=4096 shadowBias=0.0005 shadowStrength=1 '
        'shadowPcfRadius=3 shadowOrthoHalfSize=60 shadowNear=0.1 shadowFar=200',
    ]
    # One long area light over each aisle, and three over the dock area.
    length = RACK_X1 - RACK_X0 - 2.0
    for i, y in enumerate(aisles):
        lines_out.append(
            f'light /World/Lights/Aisle_{i} 0 0 -1 3.6 type=area position={fmt((RACK_X0 + RACK_X1) / 2)},{fmt(y)},'
            f'{fmt(FIXTURE_Z - 0.01)} areaSize={fmt(length)},1.2,0 areaRight=1,0,0 areaUp=0,1,0 '
            'color=1,0.97,0.92 attenuationConstant=1 attenuationLinear=0.02 attenuationQuadratic=0.008 '
            'radius=1 shadows=false')
    for i, y in enumerate((-10.5, 0.0, 10.5)):
        lines_out.append(
            f'light /World/Lights/Dock_{i} 0 0 -1 3.6 type=area position=19.5,{fmt(y)},{fmt(FIXTURE_Z - 0.01)} '
            'areaSize=18,7,0 areaRight=1,0,0 areaUp=0,1,0 color=1,0.97,0.92 attenuationConstant=1 '
            'attenuationLinear=0.02 attenuationQuadratic=0.008 radius=1 shadows=false')
    lines_out += [
        f'camera /World/Cameras/Aisle {" ".join(fmt(v) for v in (*cam["position"], *quat_yaw_pitch(cam["yaw_deg"], cam["pitch_deg"])))} '
        f'{fmt(cam["vfov"])} 0.05 400 1280 800 rgb true projection=perspective',
        object_record('/World/Physics/Ground', 'ground', (0, 0, 0), (1, 0, 0, 0), (1, 1, 1), 1, False, '-', 'static'),
        visual_record('/World/Building/Floor', 'building/floor.gltf', 'visualShadowCasting=off'),
        visual_record('/World/Building/Shell', 'building/shell.gltf', 'visualShadowCasting=double_sided'),
        visual_record('/World/Building/Structure', 'building/structure.gltf'),
        visual_record('/World/Building/Skylights', 'building/skylights.gltf', 'visualShadowCasting=off'),
        visual_record('/World/Building/Racking', 'building/racking.gltf'),
        visual_record('/World/Building/Fixtures', 'building/fixtures.gltf', 'visualShadowCasting=off'),
        visual_record('/World/Building/Yard', 'building/yard.gltf'),
    ]
    # Closed dock doors: the roller shutter scaled to the opening.
    for i, (y, opened) in enumerate(DOORS):
        if not opened:
            lines_out.append(object_record(
                f'/World/Building/Door_{i}', 'mesh', (HALF_X - 0.17, y, 0), quat_yaw(-90),
                (DOOR_W / 1.08, 1.0, (DOOR_H + 0.45) / 2.4), 1, True, 'props/rollershutter_door/model.gltf',
                'visual_only', 'renderMeshPath=props/rollershutter_door/model.gltf visualUseMeshColor=false'))
    # Colliders: walls (split around the open doors), columns, rack frames, beams and floor loads.
    t = WALL
    lines_out += [
        box_collider('/World/Physics/Wall_S', (0, -HALF_Y - t / 2, EAVE / 2), (2 * HALF_X + 2 * t, t, EAVE)),
        box_collider('/World/Physics/Wall_N', (0, HALF_Y + t / 2, EAVE / 2), (2 * HALF_X + 2 * t, t, EAVE)),
        box_collider('/World/Physics/Wall_W', (-HALF_X - t / 2, 0, EAVE / 2), (t, 2 * HALF_Y, EAVE)),
    ]
    edges = [-HALF_Y] + [y + s * DOOR_W / 2 for y, opened in DOORS if opened for s in (-1, 1)] + [HALF_Y]
    for i, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        lines_out.append(box_collider(f'/World/Physics/Wall_E_{i}', (HALF_X + t / 2, (a + b) / 2, EAVE / 2),
                                      (t, b - a, EAVE)))
    for i, (y, opened) in enumerate(DOORS):
        if opened:
            lines_out.append(box_collider(f'/World/Physics/DoorHead_{i}', (HALF_X + t / 2, y, (DOOR_H + EAVE) / 2),
                                          (t, DOOR_W, EAVE - DOOR_H)))
        else:
            lines_out.append(box_collider(f'/World/Physics/Door_{i}', (HALF_X + t / 2, y, EAVE / 2),
                                          (t, DOOR_W, EAVE)))
    for i, x in enumerate(FRAMES):
        for side in (-1, 1):
            lines_out.append(box_collider(f'/World/Physics/Column_{i}_{"N" if side > 0 else "S"}',
                                          (x, side * (HALF_Y - 0.2), EAVE / 2), (0.3, 0.4, EAVE)))
    for n, (y0, facing) in enumerate(lines):
        yc = y0 + FRAME_D / 2
        for k in range(BAYS + 1):
            x = RACK_X0 + POST / 2 + k * PITCH
            lines_out.append(box_collider(f'/World/Physics/Rack_{n}/Frame_{k}', (x, yc, UPRIGHT_H / 2),
                                          (POST, FRAME_D, UPRIGHT_H)))
        for level in LEVELS:
            for y in (y0 + BEAM_W / 2, y0 + FRAME_D - BEAM_W / 2):
                lines_out.append(box_collider(f'/World/Physics/Rack_{n}/Beam_{fmt(level)}_{"F" if y > yc else "B"}',
                                              ((RACK_X0 + RACK_X1) / 2, y, level - BEAM_H / 2),
                                              (RACK_X1 - RACK_X0, BEAM_W, BEAM_H)))
        for x in (RACK_X0 - 0.35, RACK_X1 + 0.35):
            lines_out.append(box_collider(f'/World/Physics/Rack_{n}/EndBarrier_{"W" if x < 0 else "E"}',
                                          (x, yc, 0.21), (0.12, FRAME_D + 0.1, 0.42)))
    for i, (x, y, top) in enumerate(floor_loads):
        lines_out.append(box_collider(f'/World/Physics/Load_{i}', (x, y, top / 2), (1.2, 1.0, top)))

    for mesh_path, instances in sorted(loads.batches.items()):
        name = Path(mesh_path).stem if 'cartons' in mesh_path or 'pallets' in mesh_path else Path(mesh_path).parent.name
        lines_out.append(instanced_record(f'/World/Loads/{name}', mesh_path, instances,
                                          'projectedLod=true projectedLodMinRadiusPixels=1.5'))
    dynamic = 0
    for name, mesh_path, x, y, z, yaw, mode in props:
        if mode == 'static':
            lines_out.append(object_record(f'/World/Props/{name}', 'mesh', (x, y, z), quat_yaw(yaw), (1, 1, 1), 1,
                                           True, mesh_path, 'static',
                                           f'renderMeshPath={mesh_path} visualUseMeshColor=false'))
        elif mode == 'vehicle':
            # The vehicles' simple colliders: chassis boxes in their own frame.
            lines_out.append(object_record(f'/World/Props/{name}', 'mesh', (x, y, z), quat_yaw(yaw), (1, 1, 1), 1,
                                           True, mesh_path, 'visual_only',
                                           f'renderMeshPath={mesh_path} visualUseMeshColor=false'))
            for i, (centre, size) in enumerate(VEHICLE_COLLIDERS[name.split('_')[0]]):
                c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
                world = (x + c * centre[0] - s * centre[1], y + s * centre[0] + c * centre[1], z + centre[2])
                lines_out.append(box_collider(f'/World/Physics/{name}_{i}', world, size, yaw))
        elif mode == 'visual':
            lines_out.append(object_record(f'/World/Props/{name}', 'mesh', (x, y, z), quat_yaw(yaw), (1, 1, 1), 1,
                                           True, mesh_path, 'visual_only',
                                           f'renderMeshPath={mesh_path} visualUseMeshColor=false'))
        else:
            dynamic += 1
            mass = {'Carton': 4.0, 'Crate': 1.2, 'WetFloorSign': 1.0, 'Cone': 1.5}[name.split('_')[0]]
            lines_out.append(object_record(f'/World/Props/{name}', 'mesh', (x, y, z + 0.002), quat_yaw(yaw),
                                           (1, 1, 1), mass, True, mesh_path, 'dynamic',
                                           f'collisionMode=convex_hull renderMeshPath={mesh_path}'))
    (warehouse_dir / 'rayrai_warehouse.rscene').write_text('\n'.join(lines_out) + '\n')
    instances = sum(len(v) for v in loads.batches.values())
    print(f'{len(lines)} rack lines x {BAYS} bays, aisles {aisle:.2f} m; {instances} instances in '
          f'{len(loads.batches)} batches; {len(floor_loads)} floor loads; {len(spots)} fixtures; '
          f'{dynamic} dynamic props; triangles {triangles}')


# Box colliders of the vehicles in their model frame: (centre, size).
VEHICLE_COLLIDERS = {
    'Forklift': [((-0.45, 0.0, 0.75), (2.3, 1.15, 1.5)), ((0.95, 0.0, 1.07), (0.3, 1.0, 2.14)),
                 ((1.4, 0.0, 0.03), (0.95, 0.9, 0.06))],
    'PalletJack': [((0.15, 0.0, 0.05), (1.3, 0.55, 0.1)), ((-0.6, 0.0, 0.5), (0.3, 0.3, 1.0))],
}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: generate_warehouse_rscene.py WAREHOUSE_DIR')
    generate(sys.argv[1])

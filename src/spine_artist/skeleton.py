"""Spine 4.2 skeleton JSON from the driver's controls, the layer meshes and the packed atlas.

Canvas pixels map to world units by the project frame: x = (px - ox) * s, y = (oy - py) * s (world y up).
Unweighted mesh vertices are bone-local; bones carry no rotation or scale, so local = world - bone world.
Animation names follow the target's contract (Blue Archive portraits): Idle_01, 00..NN, 99 and Eye_Close_01.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .atlas import Packed
from .mesh import Mesh

EASE = .37   # bezier handle length as a fraction of the key interval: a smooth ease in and out


def to_world(frame: dict, xy) -> np.ndarray:
    ox, oy = frame['origin']
    s = frame.get('scale', 1.)
    p = np.asarray(xy, dtype=np.float64)
    return np.stack([(p[..., 0] - ox) * s, (oy - p[..., 1]) * s], axis=-1)


def _r(v: float, nd: int = 4) -> float:
    v = round(float(v), nd)
    return 0. if v == 0 else v


def _curve(t0, t1, values0, values1):
    """Spine 4 bezier: per value channel cx1, cy1, cx2, cy2 in absolute time and value."""
    dt = t1 - t0
    out = []
    for a, b in zip(values0, values1):
        out += [_r(t0 + EASE * dt), _r(a), _r(t1 - EASE * dt), _r(b)]
    return out


def bone_world(project: dict) -> dict[str, np.ndarray]:
    return {b['name']: to_world(project['frame'], b['at']) for b in project['bones']}


def _slots(project: dict) -> list[dict]:
    """Slots in draw order: one per distinct slot name, at the position of its first layer."""
    seen, slots = {}, []
    for l in project['layers']:
        name = l['slot']
        if name not in seen:
            seen[name] = {'name': name, 'bone': l['bone'], 'blend': l.get('blend', 'normal'), 'layers': []}
            slots.append(seen[name])
        seen[name]['layers'].append(l)
    return slots


def attachment_name(layer: dict) -> str:
    return layer['expression'] if layer['role'] == 'features' else layer['id']


def build_skeleton(project: dict, meshes: dict[str, Mesh], packed: Packed) -> dict:
    frame = project['frame']
    s = frame.get('scale', 1.)
    bw = bone_world(project)
    expr = project.get('expressions', {})
    default_expr = expr.get('default')
    bones = []
    for b in project['bones']:
        entry = {'name': b['name']}
        if b.get('parent'):
            entry['parent'] = b['parent']
            local = bw[b['name']] - bw[b['parent']]
        else:
            local = bw[b['name']]
        if abs(local[0]) > 1e-9:
            entry['x'] = _r(local[0])
        if abs(local[1]) > 1e-9:
            entry['y'] = _r(local[1])
        bones.append(entry)
    slots, skin = [], {}
    all_world = []
    for slot in _slots(project):
        layers = slot['layers']
        setup = None
        if layers[0]['role'] == 'features':
            setup = default_expr
        elif layers[0]['role'] != 'eyeclose':
            setup = attachment_name(layers[0])
        entry = {'name': slot['name'], 'bone': slot['bone']}
        if setup:
            entry['attachment'] = setup
        if slot['blend'] != 'normal':
            entry['blend'] = slot['blend']
        slots.append(entry)
        atts = skin.setdefault(slot['name'], {})
        for l in layers:
            mesh = meshes[l['id']]
            pl = packed.placements[l['id']]
            x0, y0, x1, y1 = pl.box
            uv = (mesh.vertices - (x0, y0)) / (x1 - x0, y1 - y0)
            world = to_world(frame, mesh.vertices)
            if setup == attachment_name(l):
                all_world.append(world)
            local = world - bw[slot['bone']]
            att = {'type': 'mesh', 'uvs': [_r(v, 6) for v in uv.ravel()],
                   'triangles': [int(i) for i in mesh.triangles.ravel()],
                   'vertices': [_r(v) for v in local.ravel()],
                   'hull': int(len(mesh.vertices)),
                   'width': _r((x1 - x0) * s, 2), 'height': _r((y1 - y0) * s, 2)}
            if l['id'] != attachment_name(l):
                att['path'] = l['id']
            atts[attachment_name(l)] = att
    pts = np.concatenate(all_world) if all_world else np.zeros((1, 2))
    bx0, by0 = pts.min(axis=0)
    bx1, by1 = pts.max(axis=0)
    data = {
        'skeleton': {'hash': '', 'spine': project.get('spine', '4.2.33'), 'x': _r(bx0, 2), 'y': _r(by0, 2),
                     'width': _r(bx1 - bx0, 2), 'height': _r(by1 - by0, 2), 'images': './images/', 'audio': ''},
        'bones': bones,
        'slots': slots,
        'skins': [{'name': 'default', 'attachments': skin}],
        'animations': build_animations(project, meshes),
    }
    return stamp(data)


def stamp(data: dict) -> dict:
    """Set the skeleton hash from its content."""
    data['skeleton']['hash'] = ''
    digest = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).digest()
    data['skeleton']['hash'] = digest.hex()[:11]
    return data


def _loop(duration: float, key: str, rest: tuple, peak: tuple, names: tuple) -> list[dict]:
    """A there-and-back loop: rest at 0 and at the end, peak at mid-loop, eased both ways."""
    h = duration / 2
    def frame(t, values, curve_to=None):
        f = {} if t == 0 else {'time': _r(t)}
        for n, v in zip(names, values):
            if v != (1 if key == 'scale' else 0):
                f[n] = _r(v)
        if curve_to is not None:
            f['curve'] = _curve(t, curve_to[0], values, curve_to[1])
        return f
    return [frame(0, rest, (h, peak)), frame(h, peak, (duration, rest)), frame(duration, rest)]


def bone_timelines(duration: float, spec: dict) -> dict:
    """Idle controls for one bone: translate_x/y (world units), rotate (degrees), scale_x/y (factors), each at mid-loop."""
    out = {}
    tx, ty = float(spec.get('translate_x', 0)), float(spec.get('translate_y', 0))
    if tx or ty:
        out['translate'] = _loop(duration, 'translate', (0, 0), (tx, ty), ('x', 'y'))
    rot = float(spec.get('rotate', 0))
    if rot:
        out['rotate'] = _loop(duration, 'rotate', (0,), (rot,), ('value',))
    sx, sy = float(spec.get('scale_x', 1)), float(spec.get('scale_y', 1))
    if sx != 1 or sy != 1:
        out['scale'] = _loop(duration, 'scale', (1, 1), (sx, sy), ('x', 'y'))
    return out


def _deform_offsets(mesh: Mesh, frame: dict, max_y: float) -> list[float]:
    """A vertical squash of a patch: offset grows linearly from the patch top (0) to its bottom (max_y)."""
    world = to_world(frame, mesh.vertices)
    top, bottom = world[:, 1].max(), world[:, 1].min()
    span = max(top - bottom, 1e-6)
    out = []
    for x, y in world:
        out += [0., _r(max_y * (top - y) / span)]
    return out


def build_animations(project: dict, meshes: dict[str, Mesh]) -> dict:
    frame = project['frame']
    anim = project.get('animations', {})
    expr = project.get('expressions', {})
    layers = project['layers']
    features = {l['expression']: l for l in layers if l['role'] == 'features'}
    eyeclose = next((l for l in layers if l['role'] == 'eyeclose'), None)
    face_slot = next((l['slot'] for l in layers if l['role'] == 'features'), None)
    eye_slot = eyeclose['slot'] if eyeclose else None
    out = {}
    idle = anim.get('idle', {})
    d = float(idle.get('duration', 3.3333))
    bones = {}
    for name, spec in idle.get('bones', {}).items():
        timelines = bone_timelines(d, spec)
        if timelines:
            bones[name] = timelines
    out['Idle_01'] = {'bones': bones} if bones else {}
    if not features:
        return out
    ex = anim.get('expression', {})
    loop = float(ex.get('duration', 3.3333))
    breath = float(ex.get('breath_y', 0))
    def expression_anim(face_att, eye_att, deform_layer):
        a = {'slots': {}}
        a['slots'][face_slot] = {'attachment': [{'name': face_att}, {'time': _r(loop), 'name': face_att}]}
        if eye_slot:
            a['slots'][eye_slot] = {'attachment': [{'name': eye_att}, {'time': _r(loop), 'name': eye_att}]}
        if breath and deform_layer is not None:
            slot = deform_layer['slot']
            att = face_att if deform_layer['role'] == 'features' else eye_att
            offs = _deform_offsets(meshes[deform_layer['id']], frame, breath)
            h = loop / 2
            a['deform'] = {'default': {slot: {att: [
                {'curve': [_r(EASE * h), 0, _r(h - EASE * h), 1]},
                {'time': _r(h), 'vertices': offs, 'curve': [_r(h + EASE * h), 0, _r(loop - EASE * h), 1]},
                {'time': _r(loop)}]}}}
        return a
    default = expr.get('default')
    out['00'] = expression_anim(default, None, features.get(default))
    for e in expr.get('list', []):
        out[e] = expression_anim(e, None, features[e])
    if eyeclose:
        out['99'] = expression_anim(None, attachment_name(eyeclose), eyeclose)
        blink = anim.get('blink', {})
        bd = float(blink.get('duration', .1333))
        c0, c1 = (float(t) for t in blink.get('closed', [.0333, .1]))
        out['Eye_Close_01'] = {'slots': {
            face_slot: {'attachment': [{'name': default}, {'time': _r(c0), 'name': None},
                                       {'time': _r(c1), 'name': default}, {'time': _r(bd), 'name': default}]},
            eye_slot: {'attachment': [{'name': None}, {'time': _r(c0), 'name': attachment_name(eyeclose)},
                                      {'time': _r(c1), 'name': None}, {'time': _r(bd), 'name': None}]},
        }}
        squeeze = float(blink.get('squeeze_y', 0))
        if squeeze and default in features:
            offs = _deform_offsets(meshes[features[default]['id']], frame, squeeze)
            out['Eye_Close_01']['deform'] = {'default': {face_slot: {default: [
                {}, {'time': _r(c0 / 2), 'vertices': offs}, {'time': _r(c1)}]}}}
    return out

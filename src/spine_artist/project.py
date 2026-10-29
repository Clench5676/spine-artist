"""The driver's controls: a project folder of full-canvas RGBA layers plus project.json.

project.json holds every decision as plain data, so any driver (an AI that reads images and writes JSON,
or a person) can change it and rebuild. `propose` fills missing decisions with defaults taken from the
target's portrait conventions; it never overwrites a decision that is already present.

Layer roles: halo (a part that floats on its own bone), body (one per depth stratum, back to front), features
(one per expression: eyes, brows and mouth over the face), eyeclose, bangs, prop, effect.
The list order of `layers` is the draw order, bottom first; layers that share a slot (all features)
take the position of the first one.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

FORMAT = 'spine-artist/1'
ROLES = ('halo', 'body', 'features', 'eyeclose', 'bangs', 'prop', 'effect')
BLENDS = ('normal', 'additive', 'multiply', 'screen')
EXPRESSION = re.compile(r'^\d{2}$')

# Conventions of the built-in target, Blue Archive portraits (`*_spr`), read from official portraits (utaha_spr,
# CH0182_spr) with the runtime's public API on 2026-10-28. They are proposals; a driver may change any of them.
CONVENTIONS = {
    'spine': '4.2.33',
    'loop': 3.3333,                 # Idle_01 and every expression animation
    'idle_body_y': -2.5,            # world units at mid-loop
    'idle_halo_y': 7.5,
    'blink': {'duration': .1333, 'closed': [.0333, .1]},   # face hidden, eyeclose shown in [closed[0], closed[1])
    'expression_breath_y': -1.7,    # max feature-patch deform at mid-loop, scaled by height in the patch
    'blink_squeeze_y': -4.8,
}


class ProjectError(ValueError):
    pass


def load(folder: Path) -> dict:
    folder = Path(folder)
    path = folder / 'project.json'
    if not path.is_file():
        if (folder / 'full.png').is_file():
            return {'format': FORMAT, 'name': folder.name}     # images only: every decision proposed
        raise ProjectError(f'missing {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('format') != FORMAT:
        raise ProjectError(f'unsupported project format {data.get("format")!r}')
    return data


def save(folder: Path, data: dict) -> None:
    (Path(folder) / 'project.json').write_text(json.dumps(data, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


def resolve(folder: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(folder) / path


def read_layer(folder: Path, layer: dict) -> Image.Image:
    image = Image.open(Path(folder) / layer['file'])
    image.load()
    return image.convert('RGBA')


def _alpha_box(image: Image.Image):
    a = np.asarray(image.getchannel('A'))
    ys, xs = np.nonzero(a)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _centroid(image: Image.Image):
    a = np.asarray(image.getchannel('A'), dtype=np.float64)
    total = a.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(a.shape)
    return float((xs * a).sum() / total), float((ys * a).sum() / total)


def propose(folder: Path, data: dict) -> dict:
    """Fill missing decisions with defaults; returns a new dict and never changes existing values."""
    folder = Path(folder)
    out = copy.deepcopy(data)
    out.setdefault('format', FORMAT)
    out.setdefault('name', folder.name)
    layers = out.get('layers') or []
    if not layers:
        raise ProjectError('project has no layers')
    images = {l['id']: read_layer(folder, l) for l in layers}
    sizes = {im.size for im in images.values()}
    if len(sizes) != 1:
        raise ProjectError(f'layers differ in size: {sorted(sizes)}')
    width, height = sizes.pop()
    out.setdefault('canvas', {'width': width, 'height': height})
    boxes = {k: _alpha_box(im) for k, im in images.items()}
    drawn = [b for b in boxes.values() if b]
    if not drawn:
        raise ProjectError('every layer is empty')
    x0 = min(b[0] for b in drawn); y0 = min(b[1] for b in drawn)
    x1 = max(b[2] for b in drawn); y1 = max(b[3] for b in drawn)
    frame = out.setdefault('frame', {})
    target = out.get('target')
    if target and 'origin' not in frame:
        # put the new face where the target's official face is (dialogue framing, emoticon positions)
        from .reference import load_reference, propose_frame
        ref = load_reference(resolve(folder, target['reference']), target['core'])
        frame.update(propose_frame(out, {k: b for k, b in boxes.items() if b}, ref, target.get('fit', 'face')))
    frame.setdefault('origin', [round((x0 + x1) / 2, 2), float(y1)])     # feet centre, like a standing figure
    frame.setdefault('scale', 1.)
    body_ids = [l['id'] for l in layers if l.get('role') == 'body'] or [layers[0]['id']]
    halo_ids = [l['id'] for l in layers if l.get('role') == 'halo']
    if target:
        # a redraw keeps the official halo unless the images bring their own (halo.png)
        target.setdefault('halo', 'none' if halo_ids else 'official')
    bones = out.setdefault('bones', [])
    names = {b['name'] for b in bones}
    if 'root' not in names:
        bones.insert(0, {'name': 'root', 'at': list(frame['origin'])})
    if 'body' not in names:
        boxes_b = [boxes[i] for i in body_ids if boxes[i]]
        bx = (min(b[0] for b in boxes_b) + max(b[2] for b in boxes_b)) / 2
        by = max(b[3] for b in boxes_b)
        bones.append({'name': 'body', 'parent': 'root', 'at': [round(bx, 2), float(by)]})
    if halo_ids and 'halo' not in names:
        c = _centroid(images[halo_ids[0]])
        bones.append({'name': 'halo', 'parent': 'body', 'at': [round(c[0], 2), round(c[1], 2)]})
    expressions = sorted({l['expression'] for l in layers if l.get('role') == 'features' and 'expression' in l})
    for l in layers:
        role = l.get('role')
        if role not in ROLES:
            raise ProjectError(f'layer {l["id"]}: unknown role {role!r}')
        if role == 'features':
            l.setdefault('slot', 'face')
        elif role == 'eyeclose':
            l.setdefault('slot', 'eyeclose')
        else:
            l.setdefault('slot', l['id'])
        l.setdefault('bone', 'halo' if role == 'halo' and halo_ids else 'body')
        l.setdefault('blend', 'additive' if role == 'effect' else 'normal')
    expr = out.setdefault('expressions', {})
    expr.setdefault('list', expressions)
    expr.setdefault('default', expressions[0] if expressions else None)
    anim = out.setdefault('animations', {})
    idle = anim.setdefault('idle', {})
    idle.setdefault('duration', CONVENTIONS['loop'])
    idle_bones = idle.setdefault('bones', {})
    idle_bones.setdefault('body', {'translate_y': CONVENTIONS['idle_body_y']})
    if halo_ids:
        idle_bones.setdefault('halo', {'translate_y': CONVENTIONS['idle_halo_y']})
    blink = anim.setdefault('blink', {})
    blink.setdefault('duration', CONVENTIONS['blink']['duration'])
    blink.setdefault('closed', list(CONVENTIONS['blink']['closed']))
    if out.get('face_patches') != 'opaque':
        blink.setdefault('squeeze_y', CONVENTIONS['blink_squeeze_y'])
    ex = anim.setdefault('expression', {})
    ex.setdefault('duration', CONVENTIONS['loop'])
    opaque = out.get('face_patches') == 'opaque'     # whole-face patches move as one piece: no squash by default
    ex.setdefault('breath_y', 0. if opaque else CONVENTIONS['expression_breath_y'])
    if opaque:
        blink['squeeze_y'] = blink.get('squeeze_y', 0.)
    out.setdefault('mesh', {})
    out.setdefault('pack', {})
    out.setdefault('spine', CONVENTIONS['spine'])
    return out


def validate(folder: Path, data: dict) -> list[str]:
    """Hard problems that stop a build (the rails); an empty list means buildable."""
    errors = []
    layers = data.get('layers') or []
    ids = [l.get('id') for l in layers]
    if len(set(ids)) != len(ids):
        errors.append('layer ids must be unique')
    bones = {b['name']: b for b in data.get('bones', [])}
    if 'root' not in bones:
        errors.append('bone "root" is required')
    for b in bones.values():
        if b['name'] != 'root' and b.get('parent') not in bones:
            errors.append(f'bone {b["name"]}: unknown parent {b.get("parent")!r}')
        if len(b.get('at', [])) != 2:
            errors.append(f'bone {b["name"]}: "at" must be [x, y] in canvas pixels')
    slots = {}
    for l in layers:
        if l.get('role') not in ROLES:
            errors.append(f'layer {l.get("id")}: unknown role {l.get("role")!r}')
        if l.get('bone') not in bones:
            errors.append(f'layer {l.get("id")}: unknown bone {l.get("bone")!r}')
        if l.get('blend', 'normal') not in BLENDS:
            errors.append(f'layer {l.get("id")}: unknown blend {l.get("blend")!r}')
        if not (Path(folder) / l.get('file', '')).is_file():
            errors.append(f'layer {l.get("id")}: missing file {l.get("file")!r}')
        if l.get('role') == 'features' and not EXPRESSION.match(str(l.get('expression', ''))):
            errors.append(f'layer {l.get("id")}: features need a two-digit "expression"')
        slot = slots.setdefault(l.get('slot'), l)
        if slot is not l and (slot.get('bone'), slot.get('blend', 'normal')) != (l.get('bone'), l.get('blend', 'normal')):
            errors.append(f'slot {l.get("slot")}: layers disagree on bone or blend')
    expr = data.get('expressions', {})
    feats = {l.get('expression') for l in layers if l.get('role') == 'features'}
    for e in expr.get('list', []):
        if e not in feats:
            errors.append(f'expression {e}: no features layer')
    if feats and expr.get('default') not in feats:
        errors.append('expressions.default must name a features layer expression')
    if feats and not any(l.get('role') == 'eyeclose' for l in layers):
        errors.append('features need an eyeclose layer (expression 99 and the blink)')
    from .upscale import validate as validate_upscale
    errors += validate_upscale(data.get('upscale'))
    target = data.get('target')
    if target:
        halo = target.get('halo', 'none')
        if isinstance(halo, dict):
            offset = halo.get('offset', [0, 0])
            if not (isinstance(offset, list) and len(offset) == 2 and all(isinstance(v, (int, float)) for v in offset)):
                errors.append('target.halo.offset must be [x, y] in world units')
            if not isinstance(halo.get('scale', 1), (int, float)) or halo.get('scale', 1) <= 0:
                errors.append('target.halo.scale must be a positive number')
            if not isinstance(halo.get('rotate', 0), (int, float)):
                errors.append('target.halo.rotate must be a number of degrees')
            unknown = set(halo) - {'offset', 'scale', 'rotate'}
            if unknown:
                errors.append(f'target.halo: unknown keys {sorted(unknown)}')
        elif halo not in ('official', 'none'):
            errors.append('target.halo must be "official", "none" or {"offset", "scale", "rotate"}')
        if halo != 'none' and any(l.get('role') == 'halo' for l in layers):
            errors.append('a halo layer and the official halo: set target.halo to "none" or drop the halo layer')
    return errors

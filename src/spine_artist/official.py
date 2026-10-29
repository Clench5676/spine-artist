"""Parts carried over unchanged from the target's official portrait: its halo.

A redraw keeps the character's official halo instead of drawing one. The halo slots (Normal and Additive), their
attachments and region pixels, the bones they hang from and their keys in the animations the build also has
move into the new skeleton. The top carried bone keeps its official world transform, plus the driver's offset,
scale and rotation (`target.halo`). It hangs from the new body bone when the official body's bone is above it
(so it bobs with the body, as in the official idle), else from the root. Halo slots drawn behind every other
official slot go behind the new art; the others go in front of it.

Halo slots are those whose name or bone names a halo (census 2026-10-27: 567 of 634 portraits; 1474 of 1487 such
slots hang from a halo bone). Region pixels are resampled to the build's texel density, so the whole portrait
shares one density on the page.
"""
from __future__ import annotations

import copy
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import atlasfile as remap
from .skeleton import bone_world, stamp

HALO = re.compile(r'halo', re.I)
BONES_SCRIPT = Path(__file__).with_name('bones.mjs')
REGION_TYPES = ('region', 'mesh', 'linkedmesh')


class CarryError(ValueError):
    pass


def halo_setting(data: dict):
    """The resolved `target.halo` control: None (no official halo) or {'offset', 'scale', 'rotate'}."""
    value = (data.get('target') or {}).get('halo')
    if value in (None, 'none'):
        return None
    if value == 'official':
        value = {}
    return {'offset': [float(v) for v in value.get('offset', (0, 0))], 'scale': float(value.get('scale', 1)),
            'rotate': float(value.get('rotate', 0))}


def official_skeleton(folder: Path, core: str, converter: Path | None, scratch: Path) -> tuple[dict, Path]:
    """The official skeleton as Spine JSON, from <core>.json or converted from <core>.skel, and the file read."""
    js = folder / f'{core}.json'
    if js.is_file():
        return json.loads(js.read_text(encoding='utf-8')), js
    skel = folder / f'{core}.skel'
    if not skel.is_file():
        raise CarryError(f'{core}.skel (or .json) is missing in {folder}')
    if not converter or not Path(converter).is_file():
        raise CarryError(f'the official halo is read from {skel.name}: configure the converter (SPINE_ARTIST_CONVERTER)')
    scratch.mkdir(parents=True, exist_ok=True)
    out = scratch / f'{core}.json'
    r = subprocess.run([str(converter), str(skel), str(out)], capture_output=True, text=True)
    if r.returncode or not out.is_file():
        raise CarryError(f'cannot read {skel.name}: {(r.stderr or r.stdout).strip()[-300:]}')
    return json.loads(out.read_text(encoding='utf-8')), skel


def world_bones(atlas_file: Path, skeleton_file: Path) -> dict:
    """Setup-pose world transform per bone ({x, y, a, b, c, d, parent}), posed by the vendored runtime."""
    from . import runtime
    r = runtime.node(BONES_SCRIPT, atlas_file, skeleton_file)
    if r.returncode:
        raise CarryError(f'reading the official bones failed: {r.stderr.strip()[-1000:]}')
    return json.loads(r.stdout)


def region_image(atlas: remap.Atlas, region: remap.Region, pages: dict[str, Image.Image]) -> Image.Image:
    """A region's pixels upright and untrimmed (its original size), as the attachment maps them."""
    page = atlas.pages[region.page]
    image = remap._page_image(atlas, region.page, pages)
    cut = remap._rotate(image.crop(remap._box(region, page, image, atlas.file)), -region.degrees)
    sx, sy = image.width / (page.size[0] or image.width), image.height / (page.size[1] or image.height)
    full = Image.new('RGBA', (round(region.original[0] * sx), round(region.original[1] * sy)), (0, 0, 0, 0))
    full.paste(cut, (round(region.offset[0] * sx), round((region.original[1] - region.offset[1] - region.size[1]) * sy)))
    return full


def _local(world: dict, parent_xy, setting: dict) -> dict:
    """Local transform under an unrotated, unscaled parent at parent_xy, with the driver's adjustments."""
    a, b, c, d = world['a'], world['b'], world['c'], world['d']
    sx = math.hypot(a, c)
    if sx < 1e-9:
        raise CarryError('a halo bone has zero scale')
    sy = (a * d - b * c) / sx
    if abs(a * b + c * d) > 1e-3 * sx * max(abs(sy), 1e-9):
        raise CarryError('a halo bone is sheared in the official setup pose: not supported')
    out = {'x': world['x'] - parent_xy[0] + setting['offset'][0], 'y': world['y'] - parent_xy[1] + setting['offset'][1],
           'rotation': math.degrees(math.atan2(c, a)) + setting['rotate'],
           'scaleX': sx * setting['scale'], 'scaleY': sy * setting['scale']}
    return {k: round(v, 4) for k, v in out.items()}


@dataclass
class Carried:
    """The official halo, ready to pack (crops, masks) and to merge into a built skeleton."""
    crops: dict[str, tuple[Image.Image, tuple[int, int, int, int]]]
    masks: dict[str, np.ndarray]
    slots: list[str]                               # our slot names, in official draw order
    bones: list[dict]
    behind: list[dict]
    front: list[dict]
    attachments: dict[str, dict]
    animations: dict[str, dict]
    report: dict = field(default_factory=dict)

    def merge(self, skeleton: dict) -> dict:
        skeleton['bones'] += copy.deepcopy(self.bones)
        skeleton['slots'] = copy.deepcopy(self.behind) + skeleton['slots'] + copy.deepcopy(self.front)
        skin = next(s for s in skeleton['skins'] if s['name'] == 'default')
        skin['attachments'].update(copy.deepcopy(self.attachments))
        for name, parts in self.animations.items():
            if name not in skeleton['animations']:
                continue                             # only the animations the build has (the game's contract)
            anim = skeleton['animations'][name]
            for key in ('bones', 'slots'):
                if parts.get(key):
                    anim.setdefault(key, {}).update(copy.deepcopy(parts[key]))
            if parts.get('attachments'):
                anim.setdefault('attachments', {}).setdefault('default', {}).update(copy.deepcopy(parts['attachments']))
        return stamp(skeleton)


def carry_halo(data: dict, reference_folder: Path, converter: Path | None, scratch: Path,
               reserved: dict[str, set[str]]) -> Carried | None:
    """The target's official halo for the project `data` (frame and bones resolved), or None when it has none.

    reserved: the build's own 'bones', 'slots' and 'regions' names; carried names are renamed around them.
    """
    setting = halo_setting(data)
    if setting is None:
        return None
    from .reference import load_reference
    core = data['target']['core']
    skel, skel_file = official_skeleton(reference_folder, core, converter, scratch)
    atlas_file = reference_folder / f'{core}.atlas'
    atlas = remap.parse_atlas(atlas_file.read_text(encoding='utf-8'), atlas_file.name)
    ref = load_reference(reference_folder, core)
    pages = {remap.texture_key(name): image for name, image in ref.pages.items()}
    official_bones = {b['name']: b for b in skel.get('bones', [])}
    slots = skel.get('slots', [])
    halo = [s for s in slots if HALO.search(s['name']) or HALO.search(s.get('bone', ''))]
    if not halo:
        return None
    others = [i for i, s in enumerate(slots) if s not in halo]
    first_other = min(others) if others else len(slots)
    body_bone = Counter(s.get('bone') for s in slots if s not in halo).most_common(1)
    body_bone = body_bone[0][0] if body_bone else None

    # the bones the halo slots hang from, with their halo ancestors, parents first
    carried = set()
    for s in halo:
        b = s['bone']
        carried.add(b)
        parent = official_bones[b].get('parent')
        while parent and HALO.search(parent) and parent not in carried:
            carried.add(parent)
            parent = official_bones[parent].get('parent')
    order = [b['name'] for b in skel['bones'] if b['name'] in carried]
    taken = {kind: set(reserved.get(kind, ())) for kind in ('bones', 'slots', 'regions')}
    def rename(kind: str, name: str) -> str:
        new = name if name not in taken[kind] else f'official_{name}'
        while new in taken[kind]:
            new = f'official_{new}'
        taken[kind].add(new)
        return new
    bone_names = {b: rename('bones', b) for b in order}

    def ancestors(name):
        out, p = [], official_bones[name].get('parent')
        while p:
            out.append(p)
            p = official_bones[p].get('parent')
        return out
    ours = bone_world(data)
    body = next((l['bone'] for l in data['layers'] if l['role'] == 'body'), 'root')
    world = world_bones(atlas_file, skel_file)
    bones, tops = [], {}
    for name in order:
        # a skin-only bone would stay inactive: the build's one skin lists no bones
        entry = {k: v for k, v in official_bones[name].items() if k not in ('shearX', 'shearY', 'skin')}
        entry['name'] = bone_names[name]
        parent = official_bones[name].get('parent')
        if parent in carried:
            entry['parent'] = bone_names[parent]
        else:
            hang = body if body_bone and (name == body_bone or body_bone in ancestors(name)) else 'root'
            entry['parent'] = hang
            for k in ('x', 'y', 'rotation', 'scaleX', 'scaleY'):
                entry.pop(k, None)
            entry.update({k: v for k, v in _local(world[name], ours[hang], setting).items()
                          if v != (1 if k.startswith('scale') else 0)})
            tops[entry['name']] = {'official': name, 'parent': hang, 'world': [round(world[name]['x'], 3),
                                                                             round(world[name]['y'], 3)]}
        bones.append(entry)

    slot_names = {s['name']: rename('slots', s['name']) for s in halo}
    behind, front = [], []
    for i, s in enumerate(slots):
        if s not in halo:
            continue
        entry = {**s, 'name': slot_names[s['name']], 'bone': bone_names[s['bone']]}
        (behind if i < first_other else front).append(entry)

    skins = skel.get('skins', [])
    skin = next((s for s in skins if s.get('name') == 'default'), skins[0] if skins else {'attachments': {}})
    regions: dict[str, str] = {}                     # official region name -> our atlas region name
    attachments = {}
    for s in halo:
        for att_name, att in (skin.get('attachments', {}).get(s['name']) or {}).items():
            att = copy.deepcopy(att)
            kind = att.get('type', 'region')
            if kind in REGION_TYPES:
                if 'sequence' in att:
                    raise CarryError(f'halo attachment {att_name} uses a sequence: not supported')
                region = att.get('path', att_name)
                if region not in regions:
                    regions[region] = rename('regions', region)
                if regions[region] != att_name:
                    att['path'] = regions[region]
                else:
                    att.pop('path', None)
            attachments.setdefault(slot_names[s['name']], {})[att_name] = att

    # region pixels at the build's texel density (canvas pixels per world unit), before the pack's density
    density = (1. / data['frame'].get('scale', 1.)) / ref.poses['base'].get('textureDensity', 1.)
    crops, masks, region_report = {}, {}, {}
    for official_name, name in regions.items():
        found = [r for r in atlas.regions.values() if r.name == official_name and r.index == -1]
        if not found:
            raise CarryError(f'region {official_name} is not in {atlas_file.name}')
        image = region_image(atlas, found[0], pages)
        size = (max(1, round(image.width * density)), max(1, round(image.height * density)))
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        crops[name] = (image, (0, 0, *size))
        masks[name] = np.ones((size[1], size[0]), bool)   # attachments map the whole region
        region_report[name] = {'official': official_name, 'size': list(size)}

    animations = {}
    official_anims = skel.get('animations', {})
    for anim_name, anim in official_anims.items():
        parts = {}
        moved = {bone_names[b]: copy.deepcopy(t) for b, t in (anim.get('bones') or {}).items()
                 if b in carried and b != body_bone}
        if moved:
            parts['bones'] = moved
        keyed = {slot_names[s]: copy.deepcopy(t) for s, t in (anim.get('slots') or {}).items() if s in slot_names}
        if keyed:
            parts['slots'] = keyed
        deform = {}
        for skin_name, by_slot in (anim.get('attachments') or {}).items():
            if skin_name != skin.get('name', 'default'):
                continue
            for s, t in by_slot.items():
                if s in slot_names:
                    deform[slot_names[s]] = copy.deepcopy(t)
        if deform:
            parts['attachments'] = deform
        if parts:
            animations[anim_name] = parts
    report = {'from': core, 'skeleton': skel_file.name, 'setting': setting, 'bones': tops,
              'slots': {'behind': [s['name'] for s in behind], 'front': [s['name'] for s in front]},
              'regions': region_report, 'animations': sorted(animations)}
    return Carried(crops, masks, [slot_names[s['name']] for s in halo], bones, behind, front, attachments,
                   animations, report)

"""The target character's official portrait, used only as a reference: its world frame and renders to compare.

A reference folder holds the official `<core>.skel`, `<core>.atlas` and page images, read from the user's
own game files. It proposes where the new art sits in the world (so dialogue framing and emoticon positions
keep working) and renders the official portrait beside the result. Only the halo enters the build from it,
unchanged (official.py); the new art's meshes, joins and skeleton never come from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .composite import compose
from . import instruments as ins


@dataclass
class Reference:
    core: str
    folder: Path
    poses: dict                     # pose name -> geometry
    pages: dict[str, Image.Image]
    bounds: tuple[float, float, float, float]        # world left, bottom, right, top of the base pose
    face: tuple[float, float, float, float] | None   # world box of the expression-switching slots in the base pose
    face_slots: list[str]
    features: tuple[float, float, float, float] | None = None   # world box of the large changing areas (eyes)
    figure: tuple[float, float, float, float] | None = None     # world box of the base pose without the halo

    @property
    def face_share(self) -> float | None:
        """The changing areas' size against the figure height, as derive measures the new face."""
        if not self.features or not self.figure:
            return None
        l, b, r, t = self.features
        return float(((r - l) * (t - b)) ** .5 / max(1e-6, self.figure[3] - self.figure[1]))


def _box(attachments) -> tuple[float, float, float, float]:
    v = np.concatenate([np.asarray(a['vertices']).reshape(-1, 2) for a in attachments])
    return float(v[:, 0].min()), float(v[:, 1].min()), float(v[:, 0].max()), float(v[:, 1].max())


def load_reference(folder: Path, core: str) -> Reference:
    """The reference, posed once per folder and file state (callers must not change it)."""
    folder = Path(folder)
    state = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(folder.iterdir()) if p.is_file())
    return _load(str(folder.resolve()), core, state)


@lru_cache(maxsize=4)
def _load(folder: str, core: str, state) -> Reference:
    folder = Path(folder)
    skel = folder / f'{core}.skel'
    if not skel.is_file():
        skel = folder / f'{core}.json'
    atlas = folder / f'{core}.atlas'
    if not skel.is_file() or not atlas.is_file():
        raise FileNotFoundError(f'reference needs {core}.skel (or .json) and {core}.atlas in {folder}')
    posed = ins.pose_all(atlas, skel, idle_frames=0, blink_frames=0)
    poses = {p['name']: p['geometry'] for p in posed['poses']}
    pages = {}
    for p in poses['base']['pages']:
        candidates = [folder / p['name'], folder / f'{Path(p["name"]).stem}.png', folder / f'{core}-page-0.png']
        path = next((c for c in candidates if c.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'reference page {p["name"]} not found in {folder}')
        pages[p['name']] = Image.open(path).convert('RGBA')
    base = poses['base']
    # slots whose attachment differs between expressions are the face slots
    names = {}
    for g in poses.values():
        for slot, att in g['slotAttachments'].items():
            names.setdefault(slot, set()).add(att)
    face_slots = sorted(s for s, v in names.items() if len(v) > 1)
    face_atts = [a for a in base['attachments'] if a['slot'] in face_slots]
    ref = Reference(core, folder, poses, pages, _box(base['attachments']),
                    _box(face_atts) if face_atts else None, face_slots)
    ref.features = _features(ref)
    from .official import HALO
    body = [a for a in base['attachments'] if not HALO.search(a['slot'])]
    ref.figure = _box(body) if body else ref.bounds
    return ref


def _features(ref: Reference) -> tuple[float, float, float, float] | None:
    """The official expressions measured as derive measures the new ones: where they change in large areas."""
    from .derive import feature_box
    if not ref.face:
        return None
    l, b, r, t = ref.face
    m = .6 * max(r - l, t - b)
    frame = (l - m, b - m, r + m, t + m)
    density = ref.poses['base'].get('textureDensity', 1.)
    height = max(1, round((frame[3] - frame[1]) * density))
    def premul(name):
        a = np.asarray(render_world(ref.poses[name], ref.pages, frame, height), dtype=np.float32) / 255
        a[..., :3] *= a[..., 3:4]
        return a
    base = premul('base')
    union = np.zeros(base.shape[:2], bool)
    for name in ref.poses:
        if name != 'base' and name.isdigit():
            union |= np.abs(premul(name) - base).max(-1) > 24 / 255
    box = feature_box(union, (ref.bounds[3] - ref.bounds[1]) * density)
    if box is None:
        return None
    scale = height / (frame[3] - frame[1])
    return (frame[0] + box[0] / scale, frame[3] - box[3] / scale, frame[0] + box[2] / scale, frame[3] - box[1] / scale)


def propose_frame(project: dict, meshes_or_boxes: dict, ref: Reference, fit: str = 'face') -> dict:
    """Canvas-to-world frame that puts the new face where the official face is.

    fit 'face': scale from the face box widths; 'height': from the base silhouette heights.
    meshes_or_boxes maps layer id to a canvas box (x0, y0, x1, y1).
    """
    feats = [meshes_or_boxes[l['id']] for l in project['layers'] if l['role'] in ('features', 'eyeclose')]
    body = [meshes_or_boxes[l['id']] for l in project['layers'] if l['role'] not in ('features', 'eyeclose', 'effect')]
    measured = (project.get('derive') or {}).get('feature_box')
    if measured:
        # derive measures on the input images; the canvas may be their upscaled copies
        k = (project.get('derive') or {}).get('canvas_scale', 1)
        measured = [v * k for v in measured]
    if fit == 'face' and measured and ref.features:
        # like with like: the large changing areas of the new expressions against the official ones; the scale
        # is the geometric mean of the width and height ratios, so a turned head does not skew it one way
        fx0, fy0, fx1, fy1 = measured
        rx0, ry0, rx1, ry1 = ref.features
        s = (((rx1 - rx0) / max(1, fx1 - fx0)) * ((ry1 - ry0) / max(1, fy1 - fy0))) ** .5
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        wx, wy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
        fit = 'features'
    elif fit == 'face' and feats and ref.face:
        fx0 = min(b[0] for b in feats); fx1 = max(b[2] for b in feats)
        fy0 = min(b[1] for b in feats); fy1 = max(b[3] for b in feats)
        rx0, ry0, rx1, ry1 = ref.face
        s = (rx1 - rx0) / max(1, fx1 - fx0)
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        wx, wy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    else:
        y0 = min(b[1] for b in body); y1 = max(b[3] for b in body)
        x0 = min(b[0] for b in body); x1 = max(b[2] for b in body)
        l, b_, r, t = ref.bounds
        s = (t - b_) / max(1, y1 - y0)
        cx, cy = (x0 + x1) / 2, y1
        wx, wy = (l + r) / 2, b_
    return {'origin': [round(cx - wx / s, 3), round(cy + wy / s, 3)], 'scale': round(s, 6), 'fit': fit}


def render_world(geometry: dict, pages: dict, frame: tuple[float, float, float, float], height_px: int) -> Image.Image:
    left, bottom, right, top = frame
    scale = height_px / (top - bottom)
    density = scale / geometry.get('textureDensity', 1.)
    return compose(geometry, pages, density=density, frame=frame, inverse=False).image


def compare_frame(ref: Reference, ours: dict, margin: float = .04) -> tuple[float, float, float, float]:
    boxes = [ref.bounds, _box(ours['attachments'])]
    l = min(b[0] for b in boxes); b_ = min(b[1] for b in boxes)
    r = max(b[2] for b in boxes); t = max(b[3] for b in boxes)
    m = margin * (t - b_)
    return (l - m, b_ - m, r + m, t + m)

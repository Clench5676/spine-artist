"""The driver's instruments: renders of every pose, a face sheet, overlays and measured warnings.

Poses come from the vendored runtime (pose.mjs, public API); pixels are composed by composite.py.
Everything is rendered in canvas pixels so renders line up with the input layers. Instruments inform
the driver; they do not decide.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import runtime
from .composite import compose

POSE_SCRIPT = Path(__file__).with_name('pose.mjs')
BACKDROP = (128, 128, 128, 255)


class InstrumentError(RuntimeError):
    pass


def pose_all(atlas_file: Path, skeleton_file: Path, *, idle_frames: int = 8, blink_frames: int = 3) -> dict:
    result = runtime.node(POSE_SCRIPT, atlas_file, skeleton_file, idle_frames, blink_frames)
    if result.returncode:
        raise InstrumentError(f'posing failed: {result.stderr.strip()[-2000:]}')
    return json.loads(result.stdout)


def canvas_frame(project: dict) -> tuple[float, float, float, float]:
    """World (left, bottom, right, top) of the whole canvas."""
    ox, oy = project['frame']['origin']
    s = project['frame'].get('scale', 1.)
    w, h = project['canvas']['width'], project['canvas']['height']
    return (-ox * s, (oy - h) * s, (w - ox) * s, oy * s)


def render(geometry: dict, pages: dict[str, Image.Image], project: dict, *, skip_slots=(), scale: float = 1.) -> Image.Image:
    """Straight RGBA render on the canvas grid (times scale); skip_slots hides slots, e.g. the features."""
    g = dict(geometry)
    g['attachments'] = [a for a in geometry['attachments'] if a['slot'] not in skip_slots]
    w, h = round(project['canvas']['width'] * scale), round(project['canvas']['height'] * scale)
    if not g['attachments']:
        return Image.new('RGBA', (w, h), (0, 0, 0, 0))
    s = project['frame'].get('scale', 1.)
    density = (scale / s) / geometry.get('textureDensity', 1.)
    image = compose(g, pages, density=density, frame=canvas_frame(project), inverse=False).image
    if image.size != (w, h):
        image = image.crop((0, 0, w, h)) if image.width >= w and image.height >= h else image.resize((w, h))
    return image


def render_box(geometry: dict, pages: dict[str, Image.Image], project: dict, box, *, skip_slots=()) -> Image.Image:
    """Full-resolution render of one canvas box only (x0, y0, x1, y1), e.g. the face."""
    g = dict(geometry)
    g['attachments'] = [a for a in geometry['attachments'] if a['slot'] not in skip_slots]
    x0, y0, x1, y1 = box
    if not g['attachments']:
        return Image.new('RGBA', (x1 - x0, y1 - y0), (0, 0, 0, 0))
    ox, oy = project['frame']['origin']
    s = project['frame'].get('scale', 1.)
    frame = ((x0 - ox) * s, (oy - y1) * s, (x1 - ox) * s, (oy - y0) * s)
    density = (1. / s) / geometry.get('textureDensity', 1.)
    image = compose(g, pages, density=density, frame=frame, inverse=False).image
    return image.crop((0, 0, x1 - x0, y1 - y0)) if image.size != (x1 - x0, y1 - y0) else image


def on_backdrop(image: Image.Image) -> Image.Image:
    bg = Image.new('RGBA', image.size, BACKDROP)
    bg.alpha_composite(image)
    return bg.convert('RGB')


def layer_composite(layers: list[tuple[Image.Image, str]], size) -> Image.Image:
    """Direct composite of input layers (normal blend only), the reference for the round trip."""
    out = Image.new('RGBA', size, (0, 0, 0, 0))
    for image, blend in layers:
        if blend == 'normal':
            out.alpha_composite(image)
    return out


def difference(a: Image.Image, b: Image.Image) -> dict:
    """Premultiplied RGBA difference in 0..255 units."""
    x = np.asarray(a.convert('RGBA'), dtype=np.float32) / 255
    y = np.asarray(b.convert('RGBA'), dtype=np.float32) / 255
    x[..., :3] *= x[..., 3:4]
    y[..., :3] *= y[..., 3:4]
    d = np.abs(x - y).max(axis=-1) * 255
    drawn = (x[..., 3] > 0) | (y[..., 3] > 0)
    v = d[drawn] if drawn.any() else np.zeros(1)
    return {'mean': float(v.mean()), 'p99': float(np.percentile(v, 99)), 'max': float(v.max()),
            'over_8': int((v > 8).sum()), 'pixels': int(drawn.sum())}


def geometry_match(a: Image.Image, b: Image.Image) -> dict:
    """Shape agreement that blur cannot hide: silhouette IoU at alpha 0.5 and the translation between the two."""
    x = np.asarray(a.getchannel('A'), dtype=np.float32) / 255
    y = np.asarray(b.getchannel('A'), dtype=np.float32) / 255
    mx, my = x > .5, y > .5
    iou = float((mx & my).sum() / max(1, (mx | my).sum()))
    fa, fb = np.fft.rfft2(x - x.mean()), np.fft.rfft2(y - y.mean())
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    c = np.fft.irfft2(r, s=x.shape)
    dy, dx = np.unravel_index(int(np.argmax(c)), c.shape)
    dy = dy - x.shape[0] if dy > x.shape[0] // 2 else dy
    dx = dx - x.shape[1] if dx > x.shape[1] // 2 else dx
    return {'iou': iou, 'dx': int(dx), 'dy': int(dy)}


def revealed(base: Image.Image, pose: Image.Image) -> dict:
    a = np.asarray(base.getchannel('A')) > 127
    b = np.asarray(pose.getchannel('A')) > 127
    return {'revealed': int((a & ~b).sum()), 'covered': int((~a & b).sum())}


def _label(image: Image.Image, text: str) -> Image.Image:
    out = Image.new('RGB', (image.width, image.height + 18), (255, 255, 255))
    out.paste(image, (0, 18))
    ImageDraw.Draw(out).text((4, 3), text, fill=(0, 0, 0))
    return out


def sheet(items: list[tuple[str, Image.Image]], *, cell: int = 360, columns: int = 6) -> Image.Image:
    tiles = []
    for name, image in items:
        im = on_backdrop(image) if image.mode == 'RGBA' else image
        scale = cell / max(im.size)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
        tiles.append(_label(im, name))
    columns = max(1, min(columns, len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    tw = max(t.width for t in tiles)
    th = max(t.height for t in tiles)
    out = Image.new('RGB', (columns * tw, rows * th), (255, 255, 255))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % columns) * tw, (i // columns) * th))
    return out


def overlay(base: Image.Image, geometry: dict, project: dict) -> Image.Image:
    """Base render with every mesh's triangles and the bone pivots."""
    out = on_backdrop(base).convert('RGBA')
    ink = Image.new('RGBA', out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ink)
    ox, oy = project['frame']['origin']
    s = project['frame'].get('scale', 1.)
    palette = [(255, 64, 64), (64, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 255, 255)]
    for k, a in enumerate(geometry['attachments']):
        v = np.asarray(a['vertices']).reshape(-1, 2)
        px = np.stack([v[:, 0] / s + ox, oy - v[:, 1] / s], axis=1)
        col = palette[k % len(palette)] + (200,)
        t = a['triangles']
        for i in range(0, len(t), 3):
            tri = [tuple(px[j]) for j in t[i:i + 3]]
            d.line(tri + [tri[0]], fill=col, width=1)
        c = px.mean(axis=0)
        d.text((c[0], c[1]), a['slot'], fill=col)
    for b in project['bones']:
        x, y = b['at']
        d.ellipse((x - 8, y - 8, x + 8, y + 8), outline=(255, 0, 0, 255), width=3)
        d.text((x + 10, y - 6), b['name'], fill=(255, 0, 0, 255))
    out.alpha_composite(ink)
    return out.convert('RGB')


def face_box(project: dict, meshes: dict, margin: int = 40):
    ids = [l['id'] for l in project['layers'] if l['role'] in ('features', 'eyeclose')]
    if not ids:
        return None
    boxes = [meshes[i].bbox for i in ids]
    w, h = project['canvas']['width'], project['canvas']['height']
    return (max(0, min(b[0] for b in boxes) - margin), max(0, min(b[1] for b in boxes) - margin),
            min(w, max(b[2] for b in boxes) + margin), min(h, max(b[3] for b in boxes) + margin))


def expression_change(renders: list[Image.Image], patch: np.ndarray, *, grow: int = 2) -> dict:
    """Pixels that change between expression renders of one box, and how many of them lie outside the patch.

    patch: where the face patches draw in the box; texture filtering may spread them by up to `grow` px.
    """
    stack = np.stack([np.asarray(r.convert('RGBA'), dtype=np.int16) for r in renders])
    changing = (stack.max(0) - stack.min(0)).max(-1) > 8
    near = patch.copy()
    h, w = near.shape
    for dy in range(-grow, grow + 1):
        for dx in range(-grow, grow + 1):
            near[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] |= \
                patch[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return {'renders': len(renders), 'changing_px': int(changing.sum()), 'patch_px': int(patch.sum()),
            'outside_patch_px': int((changing & ~near).sum())}


def stroke_energy(image: Image.Image, box, mask=None) -> float:
    """Mean gradient magnitude of luminance inside box (and mask), on the grey backdrop: how much is drawn there."""
    g = np.asarray(on_backdrop(image).convert('L').crop(box), dtype=np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.
    gx = np.abs(np.diff(g, axis=1))[:-1]
    gy = np.abs(np.diff(g, axis=0))[:, :-1]
    e = np.hypot(gx, gy)
    if mask is not None:
        m = np.asarray(mask)[:-1, :-1]
        return float(e[m].mean()) if m.any() else 0.
    return float(e.mean())

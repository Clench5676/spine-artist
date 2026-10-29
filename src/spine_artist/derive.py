"""Images in, layers out: a full-body image and one image per expression become the portrait's layers.

Input (any image API can make these; transparent background):
  full.png                  the whole character, default face
  expressions/NN.png        the same picture with expression NN (edits of full.png, only the face changes)
  eyeclose.png              the same picture with the eyes closed (or expressions/99.png)
  blank.png      optional   the same picture with the face left blank (skin only)
  halo.png       optional   the halo alone; then full.png is drawn without it, and the halo moves on its own

Each expression is registered to full.png (scale and translation), compared with it, and the face is found
where they change. The body layer is full.png (or blank.png). Nothing is drawn here; pixels come from the
input images only.

Patches (`patch`):
  features  (default) one patch shape shared by every expression, as official portraits switch feature patches
            over one face: the eyes, brows, mouth and blush, where any edit changes them, plus a margin of skin.
            Each edit's face is registered to the body again; a change counts only where the edit adds or removes
            paint, not where it redraws a stroke a few pixels away (`jitter`) or reshades it (`soft` is the
            smallest change of the local mean colour that counts, which takes blush and leaves strokes). The patch
            edge grows over pixels where an edit and the body disagree (`seam`, up to `seam_reach`), so its
            feather lies where they agree. Switching expressions changes pixels only inside the patch; the face
            outline, hair and skin around it stay the body's.
  face      the whole face as one patch per expression (the convex hull of the changes, a margin and a feather),
            cut from each edit: it also swaps the hair, skin and outline inside the hull, which a redrawing image
            model changes on every edit.

Image models that redraw the whole picture on every edit change strokes everywhere, often more than the face;
there the driver, who sees the picture, gives the face box. Without one, a found face whose size does not fit
the character's proportions (the official face against the official figure) stops the derivation.

With upscaled copies of the images, everything is measured on the input images (the driver's pixels) and the
layers are cut from the copies: the patch outline and registration scaled up, the pixels the copies' own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

EXPR_FILE = re.compile(r'^(\d{2})\.png$')
EXPRESSION = re.compile(r'^\d{2}$')
# per patch kind, in input image pixels; threshold, strong, soft and seam are colour differences (0..1)
PATCH_DEFAULTS = {
    'features': {'close': 3, 'margin': 4, 'feather': 4, 'jitter': 2, 'strong': .3, 'soft': .04, 'soft_radius': 3,
                 'seam': 12 / 255, 'seam_reach': 6, 'face_reach': 4},
    'face': {'close': 12, 'margin': 10, 'feather': 14},
}


class DeriveError(ValueError):
    pass


def _rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGBA'), dtype=np.float32) / 255


def _premul(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[..., :3] *= out[..., 3:4]
    return out


def _grow(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask.copy()
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] |= \
                mask[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _shrink(mask: np.ndarray, r: int) -> np.ndarray:
    return ~_grow(~mask, r)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Unmasked components that do not touch the border become masked."""
    from .mesh import _label
    labels, n = _label(~mask)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])).tolist())
    holes = np.isin(labels, [i for i in range(1, n + 1) if i not in border])
    return mask | holes


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of 2-D points."""
    pts = np.unique(points, axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (q[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(q)
        return out
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def _resample(a: np.ndarray, scale: float, shift: tuple[float, float], size) -> np.ndarray:
    """a scaled about the origin and shifted (dx, dy), on a canvas of size (w, h); bilinear."""
    im = Image.fromarray(np.uint8(np.rint(np.clip(a, 0, 1) * 255)), 'RGBA')
    inv = (1 / scale, 0, -shift[0] / scale, 0, 1 / scale, -shift[1] / scale)
    out = im.transform(size, Image.Transform.AFFINE, inv, resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255


def _shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a moved by whole pixels (dx, dy); what comes in is transparent."""
    if not dx and not dy:
        return a
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _near(a: np.ndarray, b: np.ndarray, r: int) -> np.ndarray:
    """Per pixel of a: the smallest difference (largest channel) to any pixel of b within r px."""
    h, w = a.shape[:2]
    best = np.full((h, w), np.inf, np.float32)
    pb = np.pad(b, ((r, r), (r, r), (0, 0)), mode='edge')
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                best = np.minimum(best, np.abs(a - pb[r + dy:r + dy + h, r + dx:r + dx + w]).max(-1))
    return best


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    """About a Gaussian of r px per channel (three box passes of 2r+1 px); edges repeat."""
    k = 2 * r + 1
    out = a.astype(np.float64)
    for axis in (0, 1) * 3:
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode='edge')
        c = np.cumsum(p, axis)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis)), c], axis)
        out = (np.take(c, range(k, c.shape[axis]), axis) - np.take(c, range(0, c.shape[axis] - k), axis)) / k
    return out.astype(np.float32)


def _feather(region: np.ndarray, feather: int) -> tuple[np.ndarray, np.ndarray]:
    """Alpha 1 on the region, falling linearly over `feather` px outside it; and the support (alpha > 0)."""
    alpha = region.astype(np.float32)
    ring = region.copy()
    for k in range(1, feather + 1):
        nxt = _grow(ring, 1)
        alpha[nxt & ~ring] = 1 - k / (feather + 1)
        ring = nxt
    return alpha, ring


def local_shift(base: np.ndarray, img: np.ndarray, box, reach: int) -> tuple[int, int]:
    """Whole-pixel move (dx, dy), up to `reach` px, that puts img's content in `box` onto base's (premultiplied)."""
    x0, y0, x1, y1 = box
    g0, g1 = base[..., :3].mean(-1), img[..., :3].mean(-1)
    h, w = g0.shape
    x0, y0 = max(x0, reach), max(y0, reach)
    x1, y1 = min(x1, w - reach), min(y1, h - reach)
    core = g0[y0:y1, x0:x1]
    best = (float(np.abs(core - g1[y0:y1, x0:x1]).mean()), 0, 0)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            score = float(np.abs(core - g1[y0 - dy:y1 - dy, x0 - dx:x1 - dx]).mean())
            if score < best[0] - 1e-6:
                best = (score, dx, dy)
    return best[1], best[2]


def feature_patch(base: np.ndarray, edits: dict[str, np.ndarray], roi, p: dict) -> tuple[np.ndarray, dict]:
    """The features patch: one alpha for every expression (base and edits premultiplied, registered).

    Where any edit changes the face inside `roi` (strokes it adds or removes, blush), closed and filled, grown by
    `margin` and over disagreeing pixels by up to `seam_reach`, feathered by `feather`.
    """
    h, w = base.shape[:2]
    r, sr = int(p['jitter']), int(p['soft_radius'])
    g = int(p['close'] + p['margin'] + p['seam_reach'] + p['feather'] + 3 * sr + r + 2)
    x0, y0, x1, y1 = max(0, roi[0] - g), max(0, roi[1] - g), min(w, roi[2] + g), min(h, roi[3] + g)
    b = base[y0:y1, x0:x1]
    inside = np.zeros(b.shape[:2], bool)
    inside[max(0, roi[1] - y0):roi[3] - y0, max(0, roi[0] - x0):roi[2] - x0] = True
    from .mesh import _label
    strokes, soft = np.zeros_like(inside), np.zeros_like(inside)
    disagree = np.zeros(b.shape[:2], np.float32)
    strong = {}
    for key, e in edits.items():
        e = e[y0:y1, x0:x1]
        # a stroke redrawn up to `jitter` px away is the same stroke; a change must also be strong somewhere
        near = np.maximum(_near(e, b, r), _near(b, e, r))
        weak = (near > p['threshold']) & inside
        labels, n = _label(weak)
        area = np.bincount(labels.ravel(), minlength=n + 1)
        seeds = np.unique(labels[weak & (near > p['strong'])])
        seeds = seeds[(seeds > 0) & (area[seeds] >= (2 * p['speck'] + 1) ** 2)]    # specks of a redrawn edge
        strokes |= np.isin(labels, seeds)
        strong[key] = (near > p['strong']) & inside
        if p['soft']:
            # moved strokes cancel in the local mean, added paint (blush) does not; away from the silhouette, where
            # a moved edge has no opposite half to cancel it
            opaque = _shrink((e[..., 3] > .95) & (b[..., 3] > .95), r + 1)[..., None]     # models leave 253, 254
            blob = (np.abs(_blur((e - b) * opaque, sr)).max(-1) > p['soft']) & inside
            labels, n = _label(blob)
            area = np.bincount(labels.ravel(), minlength=n + 1)
            area[0] = 0
            soft |= area[labels] >= (4 * sr + 1) ** 2 // 4
        disagree = np.maximum(disagree, np.abs(e - b).max(-1))
    sites = _fill_holes(_shrink(_grow(strokes | soft, p['close']), p['close']))
    if not sites.any():
        raise DeriveError('the expressions do not change the features inside the face box')
    region = _grow(sites, p['margin'])
    seam = disagree > p['seam']
    grown = 0
    for _ in range(int(p['seam_reach'])):
        nxt = region | (_grow(region, 1) & seam)
        grown += int((nxt & ~region).sum())
        region = nxt
    alpha_w, support_w = _feather(region, int(p['feather']))
    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y1, x0:x1] = alpha_w
    band = (alpha_w > 0) & (alpha_w < 1)
    seams, missed = {}, {}
    for key, e in edits.items():
        e = e[y0:y1, x0:x1]
        a = alpha_w[..., None]
        comp = e * a + b * (1 - a)
        # where the feather mixes two different strokes, the mix differs from both (a doubled line)
        ghost = np.minimum(np.abs(comp - b).max(-1), np.abs(comp - e).max(-1))[band]
        seams[key] = round(float(np.percentile(ghost, 99)), 4) if ghost.size else 0.
        missed[key] = int((strong[key] & ~region).sum())
    info = {'sites_px': int(sites.sum()), 'soft_px': int((soft & ~_grow(strokes, p['close'])).sum()),
            'patch_px': int(support_w.sum()), 'seam_grown_px': grown, 'seam_ghost_p99': seams,
            'missed_strong_px': missed}
    return alpha, info


def _gray(a: np.ndarray, factor: int) -> np.ndarray:
    p = _premul(a)
    g = p[..., :3].mean(-1)
    h, w = g.shape
    h2, w2 = h // factor * factor, w // factor * factor
    return g[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation that moves b onto a, with its peak height."""
    fa, fb = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    r = fa * np.conj(fb)
    r /= np.maximum(np.abs(r), 1e-9)
    c = np.fft.irfft2(r, s=a.shape)
    y, x = np.unravel_index(int(np.argmax(c)), c.shape)
    if y > a.shape[0] // 2: y -= a.shape[0]
    if x > a.shape[1] // 2: x -= a.shape[1]
    return float(x), float(y), float(c.max())


def feature_box(union: np.ndarray, figure_height: float, *, open_cells: int = 2, reach: float = 2.5,
                min_share: float = .05) -> tuple[int, int, int, int] | None:
    """Where the expressions change in large areas (eyes, open mouths), as a pixel box, or None.

    Thin strokes and the edge jitter of a regenerated picture are opened away on a grid scaled to the figure,
    so the same call on the official expressions measures the same thing (the face fit compares the two).
    Areas near the largest one are kept.
    """
    from .mesh import _label
    h, w = union.shape
    k = max(1, round(figure_height / 1500))
    gh, gw = -(-h // k), -(-w // k)
    pad = np.zeros((gh * k, gw * k), bool)
    pad[:h, :w] = union
    grid = pad.reshape(gh, k, gw, k).any(axis=(1, 3))
    blobs = _grow(_shrink(grid, open_cells), open_cells)
    if not blobs.any():
        return None
    labels, n = _label(blobs)
    flat = labels.ravel()
    ys, xs = np.divmod(np.arange(flat.size), gw)
    area = np.bincount(flat, minlength=n + 1)
    cx = np.bincount(flat, weights=xs, minlength=n + 1) / np.maximum(area, 1)
    cy = np.bincount(flat, weights=ys, minlength=n + 1) / np.maximum(area, 1)
    x0 = np.full(n + 1, gw); y0 = np.full(n + 1, gh); x1 = np.zeros(n + 1, int); y1 = np.zeros(n + 1, int)
    np.minimum.at(x0, flat, xs); np.minimum.at(y0, flat, ys)
    np.maximum.at(x1, flat, xs); np.maximum.at(y1, flat, ys)
    big = int(np.argmax(area[1:])) + 1
    span = max(x1[big] - x0[big] + 1, y1[big] - y0[big] + 1)
    keep = [i for i in range(1, n + 1) if area[i] >= min_share * area[big]
            and np.hypot(cx[i] - cx[big], cy[i] - cy[big]) <= reach * span]
    return (int(min(x0[i] for i in keep) * k), int(min(y0[i] for i in keep) * k),
            int(min(w, (max(x1[i] for i in keep) + 1) * k)), int(min(h, (max(y1[i] for i in keep) + 1) * k)))


def register(full: np.ndarray, other: np.ndarray, *, scales=np.arange(.97, 1.0301, .005), factor: int = 4) -> dict:
    """Scale (about the top-left) and translation that put `other` onto `full`, coarse then at full size."""
    g_full = _gray(full, factor)
    best = None
    h, w = full.shape[:2]
    for s in scales:
        g = _gray(_resample(other, float(s), (0, 0), (w, h)), factor)
        dx, dy, peak = _phase_shift(g_full, g)
        if best is None or peak > best[3]:
            best = (float(s), dx * factor, dy * factor, peak)
    s, dx, dy, _ = best
    moved = _resample(other, s, (dx, dy), (w, h))
    fx, fy, _ = _phase_shift(_gray(full, 1), _gray(moved, 1))
    return {'scale': round(s, 4), 'dx': dx + fx, 'dy': dy + fy}


def sources(folder: Path) -> dict[str, Path]:
    """The input images by role name (full, blank, halo, eyeclose, NN)."""
    folder = Path(folder)
    out = {}
    for name in ('full', 'blank', 'halo'):
        if (folder / f'{name}.png').is_file():
            out[name] = folder / f'{name}.png'
    for f in sorted((folder / 'expressions').glob('*.png')) if (folder / 'expressions').is_dir() else []:
        m = EXPR_FILE.match(f.name)
        if m and m.group(1) not in ('00', '99'):
            out[m.group(1)] = f
    eyeclose = folder / 'eyeclose.png'
    if not eyeclose.is_file():
        eyeclose = folder / 'expressions' / '99.png'
    if eyeclose.is_file():
        out['99'] = eyeclose
    return out


def locate_sheet(full: np.ndarray, path: Path, step: int = 50) -> None:
    """full.png on grey with a pixel grid (lines every `step` px, labels every 2 steps), to read a face box from."""
    from PIL import ImageDraw
    h, w = full.shape[:2]
    rgb = full[..., :3] * full[..., 3:4] + .55 * (1 - full[..., 3:4])
    im = Image.fromarray(np.uint8(np.rint(np.clip(rgb, 0, 1) * 255)), 'RGB')
    draw = ImageDraw.Draw(im, 'RGBA')
    for x in range(0, w, step):
        draw.line([(x, 0), (x, h)], fill=(255, 0, 64, 150 if x % (2 * step) == 0 else 70))
        if x % (2 * step) == 0:
            draw.text((x + 2, 2), str(x), fill=(255, 0, 64, 255))
    for y in range(0, h, step):
        draw.line([(0, y), (w, y)], fill=(255, 0, 64, 150 if y % (2 * step) == 0 else 70))
        if y % (2 * step) == 0 and y:
            draw.text((2, y + 2), str(y), fill=(255, 0, 64, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def _share(box, figure_height: float) -> float:
    """Size of a box against the figure height (geometric mean of width and height, as the face fit)."""
    return float(((box[2] - box[0]) * (box[3] - box[1])) ** .5 / max(1., figure_height))


def derive(folder: Path, out: Path, params: dict | None = None, *, upscaled: dict[str, Path] | None = None,
           scale: int = 1, face_share: float | None = None) -> dict:
    """Write layers and a layer project into `out` from the images in `folder`; returns the layer project.

    upscaled: the sources' upscaled copies (`sources` names), `scale` times their size; the layers come from them.
    face_share: the official face's size against the official figure; a found face far from it stops here.
    """
    folder, out = Path(folder), Path(out)
    mode = (params or {}).get('patch', 'features')
    if mode not in PATCH_DEFAULTS:
        raise DeriveError(f'derive_params.patch must be one of {sorted(PATCH_DEFAULTS)}, not {mode!r}')
    p = {'patch': mode, 'threshold': 24 / 255, 'speck': 2, 'drift_p99': 18 / 255,
         'reach': 2.5, 'min_cells': 2, 'roi': .5, 'face_box': None, **PATCH_DEFAULTS[mode],
         **(params or {})}
    full_path = folder / 'full.png'
    if not full_path.is_file():
        raise DeriveError('full.png is required')
    full = _rgba(full_path)
    h, w = full.shape[:2]
    if full[..., 3].min() >= 1:
        raise DeriveError('full.png needs a transparent background')
    locate_sheet(full, out / 'locate.png')
    found = sources(folder)
    exprs = {k: v for k, v in found.items() if EXPRESSION.match(k) and k != '99'}
    if not exprs:
        raise DeriveError('at least one expressions/NN.png is required')
    if '99' not in found:
        raise DeriveError('eyeclose.png (or expressions/99.png) is required')
    aligned, report = {}, {'registration': {}, 'drift': {}}
    pf = _premul(full)
    union = np.zeros((h, w), bool)
    for key, path in {**exprs, '99': found['99']}.items():
        img = _rgba(path)
        if img.shape[:2] != (h, w):
            img = np.asarray(Image.fromarray(np.uint8(img * 255), 'RGBA').resize((w, h), Image.Resampling.LANCZOS),
                             dtype=np.float32) / 255
        d0 = np.abs(_premul(img) - pf).max(-1)
        if np.percentile(d0, 99) < p['drift_p99']:
            reg = {'scale': 1., 'dx': 0., 'dy': 0., 'identity': True}     # already aligned: keep pixels untouched
        else:
            reg = register(full, img)
            img = _resample(img, reg['scale'], (reg['dx'], reg['dy']), (w, h))
        aligned[key] = img
        report['registration'][key] = reg
        # no morphological opening here: mouths and lashes are thin strokes; specks are dropped by area below
        union |= np.abs(_premul(img) - pf).max(-1) > p['threshold']
    if not union.any():
        raise DeriveError('the expressions do not differ from full.png')
    # where to look: the driver's face_box, else the large changing areas (eyes, mouths) and a margin around
    # them; a regenerated picture also changes strokes everywhere, which must not pull the patch away
    ys_, _ = np.nonzero(full[..., 3] > 0)
    figure_height = float(ys_.max() - ys_.min() + 1) if ys_.size else float(h)
    report['figure_height'] = figure_height
    if p['face_box']:
        roi = [int(v) for v in p['face_box']]
        inside = np.zeros((h, w), bool)
        inside[max(0, roi[1]):roi[3], max(0, roi[0]):roi[2]] = True
        feats = feature_box(union & inside, figure_height)
    else:
        feats = feature_box(union, figure_height)
        if feats is None:
            raise DeriveError('no large changing area (eyes): set derive_params.face_box [x0, y0, x1, y1] '
                              f'in full.png pixels (grid: {out / "locate.png"})')
        share = _share(feats, figure_height)
        if face_share and not .5 <= share / face_share <= 2:
            raise DeriveError(
                f'the largest changes ({list(feats)}, {share:.3f} of the figure height) do not fit the face '
                f'({face_share:.3f} of the official figure): the edits redraw the picture beyond the face; set '
                f'derive_params.face_box [x0, y0, x1, y1] in full.png pixels (grid: {out / "locate.png"})')
        # twice the margin below: closed mouths are thin strokes, opened away, about one eye span under the eyes
        m_ = p['roi'] * max(feats[2] - feats[0], feats[3] - feats[1])
        roi = [int(max(0, feats[0] - m_)), int(max(0, feats[1] - m_)), int(min(w, feats[2] + m_)),
               int(min(h, feats[3] + 2 * m_))]
    report['feature_box'] = list(feats) if feats else None
    report['face_share'] = {'found': round(_share(feats, figure_height), 4) if feats else None,
                            'official': round(face_share, 4) if face_share else None}
    if p['face_box'] and feats and face_share and not .5 <= _share(feats, figure_height) / face_share <= 2:
        report.setdefault('warnings', []).append('the changes inside face_box do not fit the official face size')
    report['roi'] = roi
    inside = np.zeros((h, w), bool)
    inside[roi[1]:roi[3], roi[0]:roi[2]] = True
    report['changed_outside_roi'] = round(float((union & ~inside).sum()) / max(1, int(union.sum())), 4)
    union &= inside
    if not union.any():
        raise DeriveError('the expressions do not differ inside the face box')
    shifts = {}
    if p['patch'] == 'features':
        # the patch shows on the body: compare the edits with it (blank.png, else full.png)
        base = _premul(_rgba(folder / 'blank.png')) if (folder / 'blank.png').is_file() else pf
        if base.shape != pf.shape:
            raise DeriveError('blank.png must have the size of full.png')
        for key in aligned:
            dx, dy = local_shift(base, _premul(aligned[key]), roi, int(p['face_reach']))
            aligned[key] = _shift(aligned[key], dx, dy)
            shifts[key] = [dx, dy]
        alpha, info = feature_patch(base, {k: _premul(v) for k, v in aligned.items()}, roi, p)
        support = alpha > 0
        report['features'] = {**info, 'face_shift': shifts}
        report['outside_face_pixels'] = sum(info['missed_strong_px'].values())
    else:
        alpha, support = _face_hull(union, p, (h, w))
        report['outside_face_pixels'] = int((union & ~support).sum())
    for key, img in aligned.items():
        d = np.abs(_premul(img) - pf).max(-1)
        outside = d[~support & ((img[..., 3] > 0) | (full[..., 3] > 0))]
        report['drift'][key] = {'p99': float(np.percentile(outside, 99)) if outside.size else 0.,
                                'max': float(outside.max()) if outside.size else 0.}
    ys, xs = np.nonzero(support)
    report['face_box'] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    layers_dir = out / 'layers'
    layers_dir.mkdir(parents=True, exist_ok=True)
    def save(name, a):
        Image.fromarray(np.uint8(np.rint(np.clip(a, 0, 1) * 255)), 'RGBA').save(layers_dir / f'{name}.png')
        return f'layers/{name}.png'
    if upscaled and scale > 1:
        # cut the layers from the upscaled copies: same outline and registration, scaled; their own pixels
        size = (w * scale, h * scale)
        alpha = np.asarray(Image.fromarray(alpha, 'F').resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
        def pixels(key):
            img = _rgba(upscaled[key])
            if img.shape[:2] != (size[1], size[0]):
                raise DeriveError(f'upscaled {key} is {img.shape[1]}x{img.shape[0]}, expected {size[0]}x{size[1]}')
            reg = report['registration'].get(key)
            if reg and not reg.get('identity'):
                img = _resample(img, reg['scale'], (reg['dx'] * scale, reg['dy'] * scale), size)
            if key in shifts:
                img = _shift(img, shifts[key][0] * scale, shifts[key][1] * scale)
            return img
        aligned = {key: pixels(key) for key in aligned}
        full = pixels('full')
        report['canvas_scale'] = scale
    else:
        def pixels(key):
            return _rgba(found[key])
    layers = []
    if (folder / 'halo.png').is_file():
        layers.append({'id': 'halo', 'file': save('halo', pixels('halo')), 'role': 'halo'})
    body = pixels('blank') if (folder / 'blank.png').is_file() else full
    layers.append({'id': 'body', 'file': save('body', body), 'role': 'body'})
    for key, img in aligned.items():
        patch = img.copy()
        patch[..., 3] *= alpha
        if key == '99':
            layers.append({'id': 'eyeclose', 'file': save('eyeclose', patch), 'role': 'eyeclose'})
        else:
            layers.append({'id': f'face_{key}', 'file': save(f'face_{key}', patch), 'role': 'features',
                           'expression': key})
    # the eyeclose patch draws above the features in the slot list only by convention; order is by role
    layers.sort(key=lambda l: {'halo': 0, 'body': 1, 'features': 2, 'eyeclose': 3}[l['role']])
    report['params'] = p
    project = {'layers': layers, 'face_patches': 'opaque', 'derive': report}
    (out / 'derive.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    return project


def _face_hull(union: np.ndarray, p: dict, shape) -> tuple[np.ndarray, np.ndarray]:
    """The whole-face patch (`patch: face`): alpha and support from the changes inside the face box."""
    h, w = shape
    from .mesh import _label
    # find the face: every changed area near the largest one (eyes, brows and mouth are separate areas),
    # located on a coarse grid; the patch is their convex hull, so the skin between them comes along
    f = 8
    gh, gw = -(-h // f), -(-w // f)
    pad = np.zeros((gh * f, gw * f), bool)
    pad[:h, :w] = union
    coarse = pad.reshape(gh, f, gw, f).any(axis=(1, 3))
    labels, n = _label(_grow(coarse, 1))
    comps = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        comps.append((len(ys), xs, ys))
    comps.sort(key=lambda c: -c[0])
    _, bx, by = comps[0]
    cx, cy = bx.mean(), by.mean()
    reach = p['reach'] * max(bx.max() - bx.min() + 1, by.max() - by.min() + 1, 8)
    keep = [c for c in comps if c[0] >= p['min_cells'] or c is comps[0]]
    keep = [c for c in keep if
            np.hypot(c[1].mean() - cx, c[2].mean() - cy) <= reach]
    pts = np.concatenate([np.stack([c[1], c[2]], 1) for c in keep]).astype(np.float64)
    corners = np.concatenate([pts + d for d in ((0, 0), (1, 0), (0, 1), (1, 1))]) * f
    hull = _convex_hull(corners)
    from .mesh import rasterize
    m = p['margin'] + p['feather'] + 16
    x0, y0 = int(max(0, hull[:, 0].min() - m)), int(max(0, hull[:, 1].min() - m))
    x1, y1 = int(min(w, hull[:, 0].max() + m)), int(min(h, hull[:, 1].max() + m))
    region = rasterize([hull - (x0, y0)], (y1 - y0, x1 - x0))
    region = _grow(region, p['margin'])
    alpha_c, ring = _feather(region, p['feather'])
    alpha = np.zeros((h, w), np.float32)
    alpha[y0:y1, x0:x1] = alpha_c
    support = np.zeros((h, w), bool)
    support[y0:y1, x0:x1] = ring
    return alpha, support

"""Background removal for a full.png drawn on a background: a learned character matte (ISNet anime-seg).

The body layer is full.png, so it needs its own alpha. The matte comes from an ONNX character segmentation model
(`SPINE_ARTIST_MATTE`, for example SkyTNT's anime-seg `isnetis.onnx`), run with onnxruntime (the `matte` extra):
the whole picture, scaled to fit 1024 px and centred on black, decides what is character; along its uncertain edge
the mask comes from 1024 px tiles at the picture's own resolution, which keep finer edges. The mask is still soft
over a few pixels and keeps a rim of background colour: along the edge the alpha is capped by the pixel's distance
from the local background colour and the colour unmixed from it (the compositing equation, the background spread
in from the pixels the mask leaves out). Gaps narrower than about 3 px (between a stray strand and the head) stay
filled. Only full.png is matted; the expressions are edits of it, so derive gives them the same alpha. Results are
cached next to the build, keyed by the input bytes, the model and the settings.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ENV = 'SPINE_ARTIST_MATTE'
MODELS = {'isnet-anime': {'size': 1024, 'input': 'img'}}
VERSION = 3
OVERLAP = 128        # px shared by neighbouring tiles, blended linearly
EDGE_TAU = .1        # colour distance from the background at which an edge pixel may be opaque


class MatteError(RuntimeError):
    pass


def model_path(project: dict) -> Path | None:
    value = project.get('binary', {}).get('matte') or os.environ.get(MODEL_ENV)
    return Path(value).expanduser() if value else None


def needs_matte(rgba: np.ndarray) -> bool:
    """A picture without a transparent background: fewer than 0.1 % of its pixels below half alpha."""
    return float((rgba[..., 3] < .5).mean()) < .001


def validate(value) -> list[str]:
    if value is None or value == 'none':
        return []
    if not isinstance(value, dict) or set(value) - {'model'}:
        return ['matte must be "none" or {"model": ...}']
    if value.get('model', 'isnet-anime') not in MODELS:
        return [f'matte.model must be one of {sorted(MODELS)}']
    return []


def _session(model: Path):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise MatteError('matting needs onnxruntime (install the "matte" extra)') from e
    return ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])


def mask(session, rgb: np.ndarray, size: int, input_name: str) -> np.ndarray:
    """The model's 0..1 character mask for an RGB picture (h, w, 3, 0..1), at the picture's size."""
    h0, w0 = rgb.shape[:2]
    h, w = (size, max(1, round(size * w0 / h0))) if h0 > w0 else (max(1, round(size * h0 / w0)), size)
    ph, pw = (size - h) // 2, (size - w) // 2
    small = Image.fromarray(np.uint8(np.rint(np.clip(rgb, 0, 1) * 255)), 'RGB').resize((w, h), Image.Resampling.BILINEAR)
    x = np.zeros((size, size, 3), np.float32)
    x[ph:ph + h, pw:pw + w] = np.asarray(small, dtype=np.float32) / 255
    out = np.asarray(session.run(None, {input_name: x.transpose(2, 0, 1)[None]})[0], dtype=np.float32)
    m = out.reshape(out.shape[-2], out.shape[-1])[ph:ph + h, pw:pw + w]
    back = Image.fromarray(np.clip(m, 0, 1).astype(np.float32), 'F').resize((w0, h0), Image.Resampling.BILINEAR)
    return np.clip(np.asarray(back, dtype=np.float32), 0, 1)


def tiled_mask(session, rgb: np.ndarray, size: int, input_name: str) -> np.ndarray:
    """The mask from tiles of `size` px at the picture's own resolution, blended where they overlap."""
    h, w = rgb.shape[:2]
    def starts(n):
        return sorted({*range(0, max(1, n - size), size - OVERLAP), max(0, n - size)})
    total = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    for y in starts(h):
        for x in starts(w):
            crop = rgb[y:y + size, x:x + size]
            ch, cw = crop.shape[:2]
            yy, xx = np.mgrid[0:ch, 0:cw]
            ramp = np.clip(np.minimum.reduce([yy + 1, xx + 1, ch - yy, cw - xx]) / OVERLAP, 1e-3, 1).astype(np.float32)
            total[y:y + ch, x:x + cw] += mask(session, crop, size, input_name) * ramp
            weight[y:y + ch, x:x + cw] += ramp
    return total / np.maximum(weight, 1e-6)


def character_mask(session, rgb: np.ndarray, size: int, input_name: str) -> np.ndarray:
    """The whole picture's mask, with the tiles' finer one along its uncertain edge."""
    from .derive import _grow
    whole = mask(session, rgb, size, input_name)
    if max(rgb.shape[:2]) <= size:
        return whole
    edge = _grow((whole > .02) & (whole < .98), 2)
    return np.where(edge, tiled_mask(session, rgb, size, input_name), whole).astype(np.float32)


def refine(rgb: np.ndarray, alpha: np.ndarray, tau: float = EDGE_TAU) -> tuple[np.ndarray, np.ndarray]:
    """Colour and alpha along the mask's edge without the background's rim (rgb, alpha 0..1, straight)."""
    from .derive import _grow
    from .upscale import fill_colour
    background = fill_colour(np.concatenate([rgb, (alpha < .05)[..., None].astype(np.float32)], -1))
    edge = (alpha > 0) & _grow(alpha < .98, 1)
    distance = np.linalg.norm(rgb - background, axis=-1)
    out = np.where(edge, np.minimum(alpha, np.clip(distance / tau, 0, 1)), alpha).astype(np.float32)
    a = np.maximum(out, 1e-6)[..., None]
    unmixed = np.clip((rgb - (1 - out[..., None]) * background) / a, 0, 1)
    colour = np.where((edge & (out > .02))[..., None], unmixed, rgb).astype(np.float32)
    return colour, out


def matte_image(source: Path, cache: Path, setting: dict, model: Path, session=None) -> Path:
    """full.png with the model's alpha (RGBA, colour unchanged), cached by input, model and settings."""
    name = setting.get('model', 'isnet-anime')
    spec = MODELS[name]
    if not model.is_file():
        raise MatteError(f'matte model not found at {model} ({MODEL_ENV})')
    data = Path(source).read_bytes()
    h = hashlib.sha256(data)
    h.update(hashlib.sha256(model.read_bytes()).digest())
    h.update(f'{name}:{spec["size"]}:{VERSION}'.encode())
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f'{Path(source).stem}-{h.hexdigest()[:20]}.png'
    if target.is_file():
        return target
    with Image.open(source) as im:
        rgb = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
    colour, m = refine(rgb, character_mask(session or _session(model), rgb, spec['size'], spec['input']))
    out = np.concatenate([np.uint8(np.rint(colour * 255)), np.uint8(np.rint(m * 255))[..., None]], -1)
    partial = target.with_suffix('.part')
    Image.fromarray(out, 'RGBA').save(partial, format='PNG')
    shutil.move(partial, target)
    return target

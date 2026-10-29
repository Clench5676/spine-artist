"""Learned upscaling of the input images (waifu2x) before the build cuts and packs them.

An image tool's picture (about 1.5 MP for a full figure) holds fewer texels per world unit than an official
portrait, so the game would show it enlarged and soft. waifu2x (waifu2x-ncnn-vulkan, MIT) enlarges line art
without the blur of resampling; it adds pixels, not detail.

Colour and alpha are enlarged separately. Colour first gets the edge colour pushed into the transparent
background: the colour under clear pixels of a generated image is arbitrary, and the model would mix it into
the edge. Alpha goes through the same model as a grey picture, so edges stay as sharp as the strokes.
Results are cached next to the build, keyed by the input bytes, the tool and its arguments.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TOOL_ENV = 'SPINE_ARTIST_WAIFU2X'
SCALES = (1, 2, 4)
NOISE = (-1, 0, 1, 2, 3)
MODELS = ('cunet', 'upconv_7_anime_style_art_rgb')
DEFAULTS = {'noise': 0, 'model': 'cunet'}
SOFT = .8           # below this ratio of the official texel density the game shows the art softer (build.py)


class UpscaleError(RuntimeError):
    pass


def tool_path(project: dict) -> Path | None:
    value = project.get('binary', {}).get('waifu2x') or os.environ.get(TOOL_ENV)
    return Path(value).expanduser() if value else None


def propose_scale(estimate: float | None) -> int:
    """The smallest factor that lifts the estimated resolution ratio out of the soft range, at most 2.

    Larger factors are the driver's call (`upscale.scale` 4): memory and build time grow with the square.
    """
    if not estimate or estimate >= SOFT:
        return 1
    return 2


def resolve(project: dict, estimate: float | None) -> dict:
    """The upscale decision: the driver's `upscale` filled with defaults, or a proposal from the estimate."""
    given = project.get('upscale')
    out = {**DEFAULTS, **(given or {})}
    out['proposed'] = 'scale' not in out
    out.setdefault('scale', propose_scale(estimate))
    out['estimate'] = None if estimate is None else round(estimate, 3)
    return out


def validate(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ['upscale must be {"scale", "noise", "model"}']
    errors = []
    if value.get('scale', 1) not in SCALES:
        errors.append(f'upscale.scale must be one of {list(SCALES)}')
    if value.get('noise', 0) not in NOISE:
        errors.append(f'upscale.noise must be one of {list(NOISE)}')
    if value.get('model', 'cunet') not in MODELS:
        errors.append(f'upscale.model must be one of {list(MODELS)}')
    unknown = set(value) - {'scale', 'noise', 'model'}
    if unknown:
        errors.append(f'upscale: unknown keys {sorted(unknown)}')
    return errors


def _down2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    pad = np.zeros((h + h % 2, w + w % 2) + a.shape[2:], a.dtype)
    pad[:h, :w] = a
    return pad.reshape(pad.shape[0] // 2, 2, pad.shape[1] // 2, 2, *a.shape[2:]).mean(axis=(1, 3))


def fill_colour(rgba: np.ndarray) -> np.ndarray:
    """Straight RGB with every clear pixel given the colour of the nearest drawn ones.

    A pyramid of alpha-weighted averages: a clear pixel takes the average drawn colour of the smallest block
    around it that holds drawn pixels. Drawn pixels, partly transparent ones included, keep their own colour.
    """
    a = rgba[..., 3:4]
    levels = [(rgba[..., :3] * a, a)]
    while min(levels[-1][1].shape[:2]) > 1:
        c, w = levels[-1]
        levels.append((_down2(c), _down2(w)))
    colour = np.zeros_like(levels[-1][0])
    for c, w in reversed(levels):
        h, wd = w.shape[:2]
        up = np.repeat(np.repeat(colour, 2, 0), 2, 1)[:h, :wd] if colour.shape[:2] != (h, wd) else colour
        colour = np.where(w > 0, c / np.maximum(w, 1e-9), up)
    return np.where(a > 0, rgba[..., :3], colour)


def _key(data: bytes, tool_sha: str, scale: int, noise: int, model: str) -> str:
    h = hashlib.sha256(data)
    h.update(f'{tool_sha}:{scale}:{noise}:{model}'.encode())
    return h.hexdigest()[:20]


def upscale_images(sources: dict[str, Path], cache: Path, setting: dict, tool: Path) -> dict[str, Path]:
    """Upscaled copies of the source images (name -> path), each exactly `scale` times its size."""
    scale, noise, model = setting['scale'], setting['noise'], setting['model']
    if not tool.is_file():
        raise UpscaleError(f'waifu2x not found at {tool} ({TOOL_ENV})')
    models = tool.parent / f'models-{model}'
    if not models.is_dir():
        raise UpscaleError(f'waifu2x models not found at {models}')
    tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    done, todo = {}, {}
    for name, path in sources.items():
        data = Path(path).read_bytes()
        target = cache / f'{Path(name).stem}-{_key(data, tool_sha, scale, noise, model)}.png'
        done[name] = target
        if not target.is_file():
            todo[name] = (Path(path), target)
    if not todo:
        return done
    with tempfile.TemporaryDirectory(prefix='spine-artist-waifu2x-') as tmp:
        src, dst = Path(tmp) / 'in', Path(tmp) / 'out'
        src.mkdir()
        dst.mkdir()
        sizes = {}
        for i, (name, (path, _)) in enumerate(todo.items()):
            with Image.open(path) as im:
                rgba = np.asarray(im.convert('RGBA'), dtype=np.float32) / 255
            sizes[name] = rgba.shape[1], rgba.shape[0]
            rgb = np.uint8(np.rint(np.clip(fill_colour(rgba), 0, 1) * 255))
            alpha = np.uint8(np.rint(rgba[..., 3] * 255))
            Image.fromarray(rgb, 'RGB').save(src / f'{i:03d}-rgb.png')
            Image.fromarray(np.repeat(alpha[..., None], 3, -1), 'RGB').save(src / f'{i:03d}-a.png')
        # one call for every picture; an absolute model path, since the tool resolves relative ones itself
        r = subprocess.run([str(tool), '-i', str(src), '-o', str(dst), '-s', str(scale), '-n', str(noise),
                            '-m', str(models), '-f', 'png'], capture_output=True, text=True)
        if r.returncode:
            raise UpscaleError(f'waifu2x failed: {(r.stderr or r.stdout).strip()[-500:]}')
        for i, (name, (_, target)) in enumerate(todo.items()):
            w, h = sizes[name]
            parts = []
            for part in ('rgb', 'a'):
                f = dst / f'{i:03d}-{part}.png'
                if not f.is_file():
                    raise UpscaleError(f'waifu2x wrote no {f.name}')
                with Image.open(f) as im:
                    parts.append(np.asarray(im.convert('RGB')))
            rgb, alpha = parts
            if rgb.shape[:2] != (h * scale, w * scale) or alpha.shape[:2] != rgb.shape[:2]:
                raise UpscaleError(f'waifu2x returned {rgb.shape[1]}x{rgb.shape[0]} for {name}, '
                                   f'expected {w * scale}x{h * scale}')
            a = np.uint8(np.rint(alpha.astype(np.float32).mean(-1)))
            out = np.concatenate([rgb, a[..., None]], -1)
            partial = target.with_suffix('.part')
            Image.fromarray(out, 'RGBA').save(partial, format='PNG')
            shutil.move(partial, target)
    return done


def estimate_ratio(figure_px: float, official_figure_world: float, official_density: float) -> float:
    """Expected texels per world unit against the official portrait, before any build.

    The face fit gives the new figure about the official figure's world height, so one canvas pixel covers
    about official height / figure height world units.
    """
    if figure_px <= 0 or official_figure_world <= 0 or official_density <= 0:
        return math.nan
    return figure_px / (official_figure_world * official_density)

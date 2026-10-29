"""Ship a build as a portrait mod: the target's own bundles with the skeleton, atlas and pages replaced.

The target core's official bundles are the container only (names, CRC and every other object stay).
The build must provide the binary skeleton (`<core>.skel`), the atlas and exactly as many pages as the
official core has. Pages are stored as RGBA32, so nothing is re-encoded lossily.

Bundles are written with the `ba_modding_toolkit` package, which must be importable (for example from a
checkout on PYTHONPATH): its named-asset packer writes them with LZ4 and the CRC fix.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image


class ModError(RuntimeError):
    pass


def pack_mod(build_dir: Path, official_bundles: list[Path], out_dir: Path) -> dict:
    try:
        from ba_modding_toolkit import atlas_remap as remap, core
        from ba_modding_toolkit.bundle import Bundle
        from ba_modding_toolkit.models import SaveOptions
    except ImportError as e:
        raise ModError('the mod step needs the ba_modding_toolkit package on the path') from e
    build_dir, out_dir = Path(build_dir), Path(out_dir)
    manifest = json.loads((build_dir / 'manifest.json').read_text(encoding='utf-8'))
    name = manifest['controls']['name']
    skel = build_dir / f'{name}.skel'
    atlas = build_dir / f'{name}.atlas'
    if not skel.is_file():
        raise ModError(f'{skel.name} is missing: the build needs the binary skeleton (configure the converter)')
    pages = [build_dir / n for n in manifest['outputs'] if n.endswith('.png')]
    paths = [Path(p).resolve() for p in official_bundles]
    inputs = remap.read_inputs(paths)
    if f'{name}.skel' not in inputs.skeletons or f'{name}.atlas' not in inputs.atlases:
        raise ModError(f'the official bundles do not hold {name}.skel and {name}.atlas')
    official_atlas = remap.parse_atlas(inputs.atlases[f'{name}.atlas'], f'{name}.atlas')
    if len(official_atlas.pages) != len(pages):
        raise ModError(f'the build has {len(pages)} page(s); {name} has {len(official_atlas.pages)}')
    keys = {remap.texture_key(p.name) for p in pages}
    out_dir.mkdir(parents=True, exist_ok=True)
    options = SaveOptions(perform_crc=True, extra_bytes=b'\x08\x08\x08\x08', compression='lz4')
    with tempfile.TemporaryDirectory(prefix='spine-artist-') as temporary:
        scratch = Path(temporary)
        staged = []
        for path in paths:
            bundle = Bundle.load(path)
            if bundle is None:
                raise ModError(f'cannot load {path.name}')
            changed = False
            holds = any((obj.type.name == 'TextAsset' and obj.peek_name() in (f'{name}.skel', f'{name}.atlas'))
                        or (obj.type.name == 'Texture2D' and remap.texture_key(obj.peek_name()) in keys)
                        for obj in bundle.env.objects)
            if not holds:
                continue                     # bundles without the skeleton, atlas or pages stay out of the mod
            for obj in bundle.env.objects:
                if obj.type.name == 'Texture2D' and remap.texture_key(obj.peek_name()) in keys:
                    texture = obj.read()
                    page = next(p for p in pages if remap.texture_key(p.name) == remap.texture_key(texture.m_Name))
                    with Image.open(page) as image:
                        texture.set_image(image.convert('RGBA'), target_format=4, mipmap_count=1)
                    texture.save()
                    changed = True
            if changed:
                target = scratch / path.name
                ok, message = bundle.save(target, SaveOptions(compression='lz4'))
                if not ok:
                    raise ModError(f'cannot stage {path.name}: {message}')
                staged.append(target)
            else:
                staged.append(path)
        ok, message, pairs = core.process_asset_packing(staged, [skel, atlas, *pages], out_dir, options,
                                                       skip_unchanged=False)
        if not ok:
            raise ModError(f'packing failed: {message}')
    written = []
    for pair in pairs:
        out = Path(pair.output)
        if not Bundle.check_crc_match(out):
            raise ModError(f'{out.name} failed the CRC check')
        written.append(out.name)
    return {'core': name, 'bundles': sorted(written)}

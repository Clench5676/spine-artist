"""One build: controls and layers in, Spine skeleton, atlas, pages, instruments and manifest out.

The build never changes the driver's project.json. Missing decisions are filled in memory by
project.propose and recorded, resolved, in the manifest; `propose --write` saves them for editing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import project as proj
from .atlas import PackParams, atlas_text, pack
from .mesh import MeshParams, build_mesh, coverage
from .skeleton import attachment_name, build_skeleton
from . import instruments as ins
from . import upscale

ENGINE = 'spine-artist 0.1'
CONVERTER_ENV = 'SPINE_ARTIST_CONVERTER'


class BuildError(RuntimeError):
    def __init__(self, errors: list[str]):
        super().__init__('; '.join(errors))
        self.errors = errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mesh_params(project: dict, layer: dict) -> MeshParams:
    merged = {**project.get('mesh', {}), **layer.get('mesh', {})}
    return MeshParams(**{k: v for k, v in merged.items() if k in MeshParams.__dataclass_fields__})


def _pack_params(project: dict) -> PackParams:
    p = project.get('pack', {})
    return PackParams(**{k: v for k, v in p.items() if k in PackParams.__dataclass_fields__})


def converter_path(project: dict) -> Path | None:
    value = project.get('binary', {}).get('converter') or os.environ.get(CONVERTER_ENV)
    return Path(value) if value else None


def _upscale(raw: dict, found: dict[str, Path], ref, out: Path, warnings: list[str]):
    """The upscale decision and the upscaled copies of the input images (None at scale 1)."""
    estimate = None
    if ref is not None and ref.figure:
        with Image.open(found['full']) as im:
            rows = np.nonzero(np.asarray(im.convert('RGBA').getchannel('A')).any(1))[0]
        figure = float(rows.max() - rows.min() + 1) if rows.size else 0.
        estimate = upscale.estimate_ratio(figure, ref.figure[3] - ref.figure[1],
                                          ref.poses['base'].get('textureDensity', 1.))
    setting = upscale.resolve(raw, estimate)
    if setting['scale'] == 1:
        return setting, None
    tool = upscale.tool_path(raw)
    if tool is None or not tool.is_file():
        if not setting['proposed']:
            raise BuildError([f'upscale.scale {setting["scale"]}: waifu2x not found ({upscale.TOOL_ENV})'])
        warnings.append(f'not upscaled: waifu2x not found ({upscale.TOOL_ENV}); expected resolution '
                        f'{estimate:.2f} of the official portrait')
        setting.update(scale=1, skipped=True)
        return setting, None
    try:
        upscaled = upscale.upscale_images(found, out / 'upscaled', setting, tool)
    except upscale.UpscaleError as e:
        raise BuildError([str(e)])
    setting['tool'] = {'name': tool.name, 'sha256': _sha256(tool)}
    setting['outputs'] = {k: _sha256(v) for k, v in upscaled.items()}
    return setting, upscaled


def build(folder: Path, out: Path | None = None, *, render: bool = True, idle_frames: int = 8,
          blink_frames: int = 3) -> dict:
    folder = Path(folder)
    out = Path(out) if out else folder / 'build'
    started = time.time()
    raw = proj.load(folder)
    image_hashes = None
    warnings: list[str] = []
    upscaled_report = None
    if not raw.get('layers') and (folder / 'full.png').is_file():
        # images in: derive the layers (body, face patches, eye-close, halo) from full.png and the expressions
        from .derive import derive, sources
        found = sources(folder)
        ref = None
        if raw.get('target'):
            from .reference import load_reference
            ref = load_reference(proj.resolve(folder, raw['target']['reference']), raw['target']['core'])
        errors = upscale.validate(raw.get('upscale'))
        if errors:
            raise BuildError(errors)
        setting, upscaled = _upscale(raw, found, ref, out, warnings)
        upscaled_report = setting
        derived = derive(folder, out / 'derived', raw.get('derive_params'), upscaled=upscaled,
                         scale=setting['scale'], face_share=ref.face_share if ref else None)
        warnings += derived['derive'].get('warnings', [])
        for l in derived['layers']:
            l['file'] = str((out / 'derived' / l['file']).resolve())
        image_hashes = {p.relative_to(folder).as_posix(): _sha256(p) for p in sorted(folder.rglob('*.png'))
                        if out not in p.parents}
        raw = {**raw, **derived}
    data = proj.propose(folder, raw)
    data['_folder'] = str(folder)
    errors = proj.validate(folder, data)
    if errors:
        raise BuildError(errors)
    name = data['name']
    w, h = data['canvas']['width'], data['canvas']['height']
    images = {l['id']: proj.read_layer(folder, l) for l in data['layers']}
    for lid, im in images.items():
        if im.size != (w, h):
            raise BuildError([f'layer {lid}: size {im.size} differs from canvas {(w, h)}'])
    digests = {}
    for lid, im in images.items():
        digest = hashlib.sha256(im.tobytes()).hexdigest()
        if digest in digests:
            warnings.append(f'layers {digests[digest]} and {lid} are identical')
        digests.setdefault(digest, lid)
    meshes, mesh_report = {}, {}
    for l in data['layers']:
        alpha = np.asarray(images[l['id']].getchannel('A'))
        params = _mesh_params(data, l)
        try:
            mesh = build_mesh(alpha, params)
        except ValueError as e:
            raise BuildError([f'layer {l["id"]}: {e}'])
        cov = coverage(alpha, mesh, params.threshold)
        if cov['outside_pixels']:
            raise BuildError([f'layer {l["id"]}: {cov["outside_pixels"]} drawn pixels outside its mesh'])
        meshes[l['id']] = mesh
        mesh_report[l['id']] = {**mesh.stats, **cov, 'params': params.as_dict(), 'bbox': list(mesh.bbox)}
    pack_params = _pack_params(data)
    # layers too large for a page at the expected density are cut into pieces along rows (pieces.py)
    from . import pieces
    params_of = lambda layer: _mesh_params(data, layer)
    errors = pieces.validate(data.get('cuts'), data, meshes)
    if errors:
        raise BuildError(errors)
    cuts = pieces.snap(data['cuts'], data, params_of) if data.get('cuts') is not None \
        else pieces.propose(data, meshes, images, pack_params, params_of)
    data['cuts'] = {lid: rows for lid, rows in cuts.items() if rows}
    try:
        sources = pieces.split(data, data['cuts'], meshes, images, params_of)
    except ValueError as e:
        raise BuildError([f'cut: {e}'])
    for of in {s['piece']['of'] for s in data['layers'] if 'piece' in s}:
        mesh_report.pop(of, None)
    for pid in sources:
        layer = next(l for l in data['layers'] if l['id'] == pid)
        params = _mesh_params(data, layer)
        cov = coverage(np.asarray(images[pid].getchannel('A')), meshes[pid], params.threshold)
        if cov['outside_pixels']:
            raise BuildError([f'layer {pid}: {cov["outside_pixels"]} drawn pixels outside its mesh'])
        mesh_report[pid] = {**meshes[pid].stats, **cov, 'params': params.as_dict(), 'bbox': list(meshes[pid].bbox),
                            'band': layer['piece']['band']}
    from .mesh import rasterize
    crops, masks = {}, {}
    for lid, mesh in meshes.items():
        if lid in sources:
            image, box, masks[lid] = pieces.crop(sources[lid], mesh)
            crops[lid] = (image, box)
            continue
        # the hull may pass the canvas edge (padding around a figure that touches it); PIL pads such a crop with
        # transparent pixels, so the crop box always equals the hull box and UVs stay inside 0..1
        crops[lid] = (images[lid].crop(mesh.bbox), mesh.bbox)
        x0, y0, x1, y1 = mesh.bbox
        polys = [p - (x0, y0) for p in mesh.polygons]
        masks[lid] = rasterize(polys, (y1 - y0, x1 - x0))
    out.mkdir(parents=True, exist_ok=True)
    carried = None
    if data.get('target'):
        # the target's official halo, unchanged, unless the images bring their own
        from .official import CarryError, carry_halo
        reserved = {'bones': {b['name'] for b in data['bones']}, 'slots': {l['slot'] for l in data['layers']},
                    'regions': set(crops)}
        try:
            carried = carry_halo(data, proj.resolve(folder, data['target']['reference']), converter_path(data),
                                 out / 'official', reserved)
        except CarryError as e:
            raise BuildError([str(e)])
        if carried:
            crops.update(carried.crops)
            masks.update(carried.masks)
            data['_carried_slots'] = carried.slots
    packed = pack(crops, pack_params, masks)
    if packed.density < pack_params.density:
        warnings.append(f'texture density reduced to {packed.density:.3f} to fit the pages')
    resolution = None
    if data.get('target'):
        from .reference import load_reference
        ref = load_reference(proj.resolve(folder, data['target']['reference']), data['target']['core'])
        official = ref.poses['base'].get('textureDensity', 1.)
        texels = packed.density / data['frame'].get('scale', 1.)
        resolution = {'texels_per_world_unit': round(texels, 4), 'official': round(official, 4),
                      'ratio': round(texels / official, 3)}
        if texels / official < .8:
            warnings.append(f'texture resolution is {texels / official:.2f} of the official portrait: '
                            'the game shows it upscaled (softer)')
    page_names = [f'{name}.png' if i == 0 else f'{name}_{i + 1}.png' for i in range(len(packed.pages))]
    for page, page_name in zip(packed.pages, page_names):
        page.save(out / page_name)
    atlas_file = out / f'{name}.atlas'
    atlas_file.write_text(atlas_text(packed, page_names), encoding='utf-8')
    skeleton = build_skeleton(data, meshes, packed)
    if carried:
        carried.merge(skeleton)
    skeleton_file = out / f'{name}.json'
    skeleton_file.write_text(json.dumps(skeleton, separators=(',', ':')), encoding='utf-8')
    binary = None
    conv = converter_path(data)
    if conv and conv.is_file():
        skel_file = out / f'{name}.skel'
        r = subprocess.run([str(conv), str(skeleton_file), str(skel_file), '-v', data.get('spine', '4.2.33')],
                           capture_output=True, text=True)
        if r.returncode or not skel_file.is_file():
            raise BuildError([f'binary conversion failed: {(r.stderr or r.stdout).strip()[-500:]}'])
        binary = skel_file
        # rail: the runtime must read the binary exactly as the JSON
        a = ins.pose_all(atlas_file, skeleton_file, idle_frames=2, blink_frames=2)['poses']
        b = ins.pose_all(atlas_file, skel_file, idle_frames=2, blink_frames=2)['poses']
        worst = 0.
        for pa, pb in zip(a, b):
            if [x['slot'] for x in pa['geometry']['attachments']] != [x['slot'] for x in pb['geometry']['attachments']]:
                raise BuildError([f'binary skeleton differs from JSON in pose {pa["name"]}'])
            for xa, xb in zip(pa['geometry']['attachments'], pb['geometry']['attachments']):
                worst = max(worst, float(np.abs(np.subtract(xa['vertices'], xb['vertices'])).max()))
        if len(a) != len(b) or worst > .01:
            raise BuildError([f'binary skeleton differs from JSON (max {worst:.4f} world units)'])
    else:
        warnings.append(f'binary .skel not written: no converter configured ({CONVERTER_ENV})')
    report = {'engine': ENGINE, 'name': name, 'warnings': warnings, 'meshes': mesh_report, 'pack': packed.stats,
              'resolution': resolution, 'upscale': upscaled_report, 'official': carried.report if carried else None,
              'atlas': {lid: {'page': pl.page, 'x': pl.x, 'y': pl.y, 'width': pl.width, 'height': pl.height,
                              'box': list(pl.box)} for lid, pl in packed.placements.items()}}
    data['_density'] = packed.density
    if render:
        report['instruments'] = instrument(data, meshes, images, out, atlas_file, binary or skeleton_file,
                                           page_names, idle_frames=idle_frames, blink_frames=blink_frames,
                                           warnings=warnings)
    outputs = {p.name: _sha256(p) for p in [atlas_file, skeleton_file, *(out / n for n in page_names)]
               + ([binary] if binary else [])}
    manifest = {
        'format': 'spine-artist-manifest/1', 'engine': ENGINE,
        'built': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'seconds': round(time.time() - started, 2),
        'images': image_hashes,
        'inputs': {l['id']: {'file': l['file'], 'sha256': _sha256(folder / l['file'])} for l in data['layers']},
        'controls': {k: v for k, v in data.items() if not k.startswith('_')}, 'outputs': outputs,
        'meshes': {lid: {k: v for k, v in r.items() if k != 'params'} for lid, r in mesh_report.items()},
        'atlas': report['atlas'], 'pack': packed.stats, 'resolution': resolution, 'upscale': upscaled_report,
        'warnings': warnings,
    }
    if data.get('target'):
        manifest['reference'] = str(proj.resolve(folder, data['target']['reference']).resolve())
    if carried:
        reference = proj.resolve(folder, data['target']['reference'])
        manifest['official'] = {**carried.report, 'files': {
            p.name: _sha256(p) for p in sorted(reference.iterdir())
            if p.is_file() and p.name.startswith(data['target']['core'])}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    (out / 'report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def instrument(data, meshes, images, out, atlas_file, skeleton_file, page_names, *, idle_frames, blink_frames,
               warnings) -> dict:
    renders = out / 'renders'
    if renders.exists():
        shutil.rmtree(renders)
    renders.mkdir()
    posed = ins.pose_all(atlas_file, skeleton_file, idle_frames=idle_frames, blink_frames=blink_frames)
    required = ['Idle_01']
    if any(l['role'] == 'features' for l in data['layers']):
        required += ['00', '99', 'Eye_Close_01', *data['expressions']['list']]
    if posed.get('animations') is not None:
        missing = [a for a in required if a not in posed['animations']]
        if missing:
            raise BuildError([f'animations missing: {missing}'])
    pages = {n: Image.open(out / n).convert('RGBA') for n in page_names}
    # the base pose at full size (round trip, overlays); every pose at the preview scale for the sheets
    preview = float(data.get('instruments', {}).get('preview_scale', .5))
    base_geometry = posed['poses'][0]['geometry']
    base = ins.render(base_geometry, pages, data)
    base.save(renders / 'pose-base.png')
    frames = {}
    for p in posed['poses']:
        frames[p['name']] = ins.render(p['geometry'], pages, data, scale=preview)
        frames[p['name']].save(renders / f'preview-{p["name"]}.png')
    result = {'poses': [p['name'] for p in posed['poses']], 'preview_scale': preview}
    # round trip: the setup pose against the input layers composed directly
    setup = []
    for l in data['layers']:
        role = l['role']
        visible = role not in ('features', 'eyeclose') or (role == 'features' and l['expression'] == data['expressions']['default'])
        if visible:
            setup.append((images[l['id']], l.get('blend', 'normal')))
    direct = ins.layer_composite(setup, base.size)
    # the carried official halo has no input layer: the round trip leaves it out
    own = ins.render(base_geometry, pages, data, skip_slots=tuple(data['_carried_slots'])) \
        if data.get('_carried_slots') else base
    # pixels must match at density 1; below it the texture is smaller, so check the shapes instead of the pixels
    d = float(data.get('_density', 1.))
    rt = ins.difference(own, direct)
    result['round_trip'] = rt
    result['geometry'] = gm = ins.geometry_match(own, direct)
    if d >= 1 and (rt['p99'] > 6 or rt['max'] > 64):
        warnings.append(f'setup render differs from the layers (p99 {rt["p99"]:.1f}, max {rt["max"]:.1f})')
    if gm['iou'] < .995 or abs(gm['dx']) > 1 or abs(gm['dy']) > 1:
        warnings.append(f'setup render is misplaced against the layers (IoU {gm["iou"]:.4f}, shift {gm["dx"]}, {gm["dy"]})')
    result['motion'] = {name: ins.revealed(frames['base'], im) for name, im in frames.items() if name != 'base'}
    ins.sheet([(k, v) for k, v in frames.items()]).save(renders / 'sheet-poses.png')
    ins.overlay(base, base_geometry, data).save(renders / 'overlay-mesh-bones.png')
    face_slot = next((l['slot'] for l in data['layers'] if l['role'] == 'features'), None)
    box = ins.face_box(data, meshes)
    if face_slot and box and data.get('face_patches') == 'opaque':
        crops = {p['name']: ins.render_box(p['geometry'], pages, data, box) for p in posed['poses']
                 if not p['name'].startswith('idle')}
        ins.sheet(list(crops.items()), cell=300, columns=5).save(renders / 'sheet-face.png')
        # switching expressions must change pixels only where the face patches draw
        patch = np.zeros((box[3] - box[1], box[2] - box[0]), bool)
        for l in data['layers']:
            if l['role'] == 'features':
                patch |= np.asarray(images[l['id']].getchannel('A').crop(box)) > 0
        names = [n for n in ['00', *data['expressions']['list']] if n in crops]
        result['expression_change'] = change = ins.expression_change([crops[n] for n in names], patch)
        if change['outside_patch_px']:
            warnings.append(f'switching expressions changes {change["outside_patch_px"]} px outside the face patches')
    elif face_slot and box:
        bare = ins.render(base_geometry, pages, data, skip_slots=(face_slot,))
        bare.save(renders / 'face-base-without-features.png')
        items = [('face base', bare.crop(box))]
        items += [(p['name'], ins.render_box(p['geometry'], pages, data, box)) for p in posed['poses']
                  if not p['name'].startswith('idle')]
        ins.sheet(items, cell=300, columns=5).save(renders / 'sheet-face.png')
        # strokes left on the face base, measured only where features draw and nothing above the face covers
        shape = (data['canvas']['height'], data['canvas']['width'])
        order = [l['slot'] for l in data['layers']]
        above = {l['id'] for l in data['layers'] if order.index(l['slot']) > order.index(face_slot)}
        feat_ids = [l['id'] for l in data['layers'] if l['role'] in ('features', 'eyeclose')]
        drawn = np.zeros(shape, bool)
        for lid in feat_ids:
            drawn |= np.asarray(images[lid].getchannel('A')) > 0
        covered = np.zeros(shape, bool)
        for lid in above:
            covered |= np.asarray(images[lid].getchannel('A')) > 200
        mask = drawn & ~covered
        e_bare = ins.stroke_energy(bare, box, mask[box[1]:box[3], box[0]:box[2]])
        e_full = ins.stroke_energy(base, box, mask[box[1]:box[3], box[0]:box[2]])
        result['face_base_strokes'] = {'without_features': e_bare, 'with_features': e_full}
        if e_full > 0 and e_bare > .5 * e_full:
            warnings.append('face base carries strokes where the features draw: check face-base-without-features.png')
        fbox = [meshes[l['id']].bbox for l in data['layers'] if l['role'] == 'features']
        cbox = [meshes[l['id']].bbox for l in data['layers'] if l['role'] == 'eyeclose']
        if fbox and cbox:
            def union(bs):
                return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))
            fb, cb = union(fbox), union(cbox)
            ix = max(0, min(fb[2], cb[2]) - max(fb[0], cb[0])) * max(0, min(fb[3], cb[3]) - max(fb[1], cb[1]))
            ua = (fb[2] - fb[0]) * (fb[3] - fb[1]) + (cb[2] - cb[0]) * (cb[3] - cb[1]) - ix
            iou = ix / max(1, ua)
            result['eyeclose_vs_features_box_iou'] = iou
            if iou < .5:
                warnings.append(f'eyeclose and features cover different face areas (box IoU {iou:.2f})')
    if data.get('target'):
        result['compare'] = compare_with_reference(data, posed, pages, renders)
    return result


def compare_with_reference(data, posed, pages, renders) -> dict:
    """Official portrait beside the build, in one world frame: whole figure and the face per expression."""
    from .reference import compare_frame, load_reference, render_world
    target = data['target']
    ref = load_reference(proj.resolve(Path(data.get('_folder', '.')), target['reference']), target['core'])
    ours = {p['name']: p['geometry'] for p in posed['poses']}
    frame = compare_frame(ref, ours['base'])
    a = render_world(ref.poses['base'], ref.pages, frame, 1200)
    b = render_world(ours['base'], pages, frame, 1200)
    ins.sheet([('official', a), ('build', b)], cell=900, columns=2).save(renders / 'compare-figure.png')
    feats = [a_ for a_ in ours['base']['attachments'] if a_['slot'] in
             {l['slot'] for l in data['layers'] if l['role'] in ('features', 'eyeclose')}]
    from .reference import _box
    boxes = [bx for bx in [ref.face, _box(feats) if feats else None] if bx]
    result = {'frame': frame, 'official_face': ref.face, 'face_slots': ref.face_slots}
    if boxes:
        l = min(x[0] for x in boxes); bt = min(x[1] for x in boxes)
        r = max(x[2] for x in boxes); t = max(x[3] for x in boxes)
        m = .35 * max(r - l, t - bt)
        face = (l - m, bt - m, r + m, t + m)
        items = []
        for name in [n for n in ours if n in ref.poses and not n.startswith(('idle', 'blink'))]:
            items.append((f'official {name}', render_world(ref.poses[name], ref.pages, face, 360)))
            items.append((f'build {name}', render_world(ours[name], pages, face, 360)))
        ins.sheet(items, cell=300, columns=6).save(renders / 'compare-face.png')
        result['face_frame'] = face
    return result

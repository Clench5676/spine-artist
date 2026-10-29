"""The Spine artist engine on synthetic layer sets: meshes, packing, skeleton, round trip and warnings."""
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from spine_artist import project as proj
from spine_artist.atlas import PackParams, atlas_text, pack
from spine_artist.build import BuildError, build
from spine_artist.mesh import MeshParams, build_mesh, coverage, is_simple, triangulate, _area

W, H = 400, 600
needs_node = pytest.mark.skipif(shutil.which('node') is None, reason='node is required for posing')


def _layer():
    return Image.new('RGBA', (W, H), (0, 0, 0, 0))


def make_project(folder, *, eyes_on_face=False, eyeclose=True):
    (folder / 'layers').mkdir(parents=True)
    halo = _layer(); ImageDraw.Draw(halo).ellipse((150, 20, 250, 60), outline=(255, 120, 200, 255), width=6)
    body = _layer(); d = ImageDraw.Draw(body)
    d.rectangle((150, 230, 250, 560), fill=(40, 40, 60, 255))
    d.ellipse((145, 80, 255, 220), fill=(255, 232, 220, 255))
    if eyes_on_face:
        d.ellipse((170, 130, 192, 150), fill=(90, 60, 160, 255)); d.ellipse((208, 130, 230, 150), fill=(90, 60, 160, 255))
    f01 = _layer(); d = ImageDraw.Draw(f01)
    d.ellipse((170, 130, 192, 150), fill=(90, 60, 160, 255)); d.ellipse((208, 130, 230, 150), fill=(90, 60, 160, 255))
    d.line((190, 185, 210, 185), fill=(200, 90, 110, 255), width=3)
    f02 = _layer(); d = ImageDraw.Draw(f02)
    d.ellipse((170, 128, 192, 152), fill=(90, 60, 160, 255)); d.ellipse((208, 128, 230, 152), fill=(90, 60, 160, 255))
    d.arc((185, 175, 215, 195), 0, 180, fill=(200, 90, 110, 255), width=3)
    ec = _layer(); d = ImageDraw.Draw(ec)
    d.line((170, 142, 192, 142), fill=(90, 60, 160, 255), width=3); d.line((208, 142, 230, 142), fill=(90, 60, 160, 255), width=3)
    d.line((190, 185, 210, 185), fill=(200, 90, 110, 255), width=3)
    bangs = _layer(); d = ImageDraw.Draw(bangs)
    for x in (170, 200, 230):
        d.polygon([(x - 12, 90), (x + 12, 90), (x + 3, 140)], fill=(170, 140, 225, 255))
    layers = [('halo', halo, 'halo', None), ('body', body, 'body', None), ('face_01', f01, 'features', '01'),
              ('face_02', f02, 'features', '02')]
    if eyeclose:
        layers.append(('eyeclose', ec, 'eyeclose', None))
    layers.append(('bangs', bangs, 'bangs', None))
    entries = []
    for lid, im, role, expr in layers:
        im.save(folder / 'layers' / f'{lid}.png')
        e = {'id': lid, 'file': f'layers/{lid}.png', 'role': role}
        if expr:
            e['expression'] = expr
        entries.append(e)
    proj.save(folder, {'format': proj.FORMAT, 'name': 'test_spr', 'layers': entries})
    return folder


def test_mesh_contains_every_drawn_pixel():
    im = Image.new('L', (300, 200), 0)
    d = ImageDraw.Draw(im)
    d.ellipse((10, 10, 120, 150), fill=255)
    d.polygon([(150, 20), (290, 40), (200, 190)], fill=30)
    d.line((20, 180, 280, 185), fill=200, width=2)
    a = np.asarray(im)
    for tol in (0, 2, 8):
        mesh = build_mesh(a, MeshParams(tolerance=tol))
        assert coverage(a, mesh)['outside_pixels'] == 0
        assert all(is_simple(p) for p in mesh.polygons)
        assert all(_area(p) > 0 for p in mesh.polygons)
    assert build_mesh(a, MeshParams(tolerance=8)).stats['vertices'] < build_mesh(a, MeshParams(tolerance=1)).stats['vertices']


def test_mesh_rejects_an_empty_layer():
    with pytest.raises(ValueError):
        build_mesh(np.zeros((10, 10), np.uint8))


def test_triangulation_covers_the_polygon_area():
    poly = np.array([(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)], dtype=float)
    tris = triangulate(poly)
    assert len(tris) == 3
    total = sum(abs(_area(poly[list(t)])) for t in tris)
    assert total == pytest.approx(abs(_area(poly)))


def test_pack_keeps_regions_apart_and_shrinks_to_fit():
    crops = {f'r{i}': (Image.new('RGBA', (300, 200), (i * 20, 0, 0, 255)), (0, 0, 300, 200)) for i in range(6)}
    packed = pack(crops, PackParams(page_width=512, page_height=512, padding=2, mode='rect'))
    assert packed.density < 1
    rects = [(p.x, p.y, p.x + (p.height if p.degrees in (90, 270) else p.width),
              p.y + (p.width if p.degrees in (90, 270) else p.height)) for p in packed.placements.values()]
    for i, a in enumerate(rects):
        assert a[2] <= 512 and a[3] <= 512
        for b in rects[i + 1:]:
            assert a[2] + 2 <= b[0] or b[2] + 2 <= a[0] or a[3] + 2 <= b[1] or b[3] + 2 <= a[1]
    text = atlas_text(packed, ['p.png'])
    assert text.startswith('p.png\n\tsize: 512, 512') and text.count('bounds:') == 6


def test_bleed_colours_transparent_pixels_without_touching_alpha():
    im = Image.new('RGBA', (10, 10), (0, 0, 0, 0))
    im.putpixel((5, 5), (200, 100, 50, 255))
    packed = pack({'a': (im, (0, 0, 10, 10))}, PackParams(page_width=64, page_height=64, bleed=2))
    pl = packed.placements['a']
    page = np.asarray(packed.pages[0])
    assert tuple(page[pl.y + 5, pl.x + 6]) == (200, 100, 50, 0)
    assert page[..., 3].sum() == 255


def test_propose_fills_conventions_without_overwriting(tmp_path):
    folder = make_project(tmp_path / 'p')
    data = proj.load(folder)
    data['animations'] = {'idle': {'bones': {'body': {'translate_y': -9}}}}
    out = proj.propose(folder, data)
    assert out['animations']['idle']['bones']['body']['translate_y'] == -9
    assert out['animations']['idle']['bones']['halo']['translate_y'] == proj.CONVENTIONS['idle_halo_y']
    assert out['expressions'] == {'list': ['01', '02'], 'default': '01'}
    assert {b['name'] for b in out['bones']} == {'root', 'body', 'halo'}
    assert proj.validate(folder, out) == []


def test_validate_requires_eyeclose_with_features(tmp_path):
    folder = make_project(tmp_path / 'p', eyeclose=False)
    errors = proj.validate(folder, proj.propose(folder, proj.load(folder)))
    assert any('eyeclose' in e for e in errors)
    with pytest.raises(BuildError):
        build(folder, render=False)


@needs_node
def test_build_round_trip_and_animation_contract(tmp_path):
    folder = make_project(tmp_path / 'p')
    report = build(folder)
    out = folder / 'build'
    skeleton = json.loads((out / 'test_spr.json').read_text())
    assert {'Idle_01', '00', '01', '02', '99', 'Eye_Close_01'} <= set(skeleton['animations'])
    assert skeleton['skeleton']['spine'] == '4.2.33'
    rt = report['instruments']['round_trip']
    assert rt['max'] <= 2 and rt['over_8'] == 0
    poses = report['instruments']['poses']
    assert {'base', '01', '02', '99'} <= set(poses) and any(p.startswith('blink') for p in poses)
    # eye-close replaces the features: above the closed lids, the closed pose draws no eye colour
    def eye_pixels(name):
        a = np.asarray(Image.open(out / 'renders' / f'preview-{name}.png').convert('RGBA')).astype(int)
        band = a[64:69, 84:117]                     # half-scale preview, rows above the lid line
        return int((np.abs(band[..., :3] - (90, 60, 160)).sum(-1) < 30).sum())
    assert eye_pixels('01') > 20 and eye_pixels('99') == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert set(manifest['inputs']) == {'halo', 'body', 'face_01', 'face_02', 'eyeclose', 'bangs'}
    assert manifest['controls']['frame']['origin']
    assert not any('face base' in w for w in report['warnings'])


@needs_node
def test_build_warns_when_the_face_base_keeps_eyes(tmp_path):
    folder = make_project(tmp_path / 'p', eyes_on_face=True)
    report = build(folder)
    assert any('face base' in w for w in report['warnings'])


def test_polygon_packing_interlocks_without_overlap():
    crops, masks = {}, {}
    for i in range(2):
        im = Image.new('RGBA', (300, 300), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.polygon([(0, 0), (299, 0), (0, 299)], fill=(60 * i, 100, 200, 255))
        crops[f't{i}'] = (im, (0, 0, 300, 300))
        masks[f't{i}'] = np.asarray(im.getchannel('A')) > 0
    rect = pack(crops, PackParams(page_width=340, page_height=340, mode='rect'))
    poly = pack(crops, PackParams(page_width=340, page_height=340), masks)
    assert rect.density < .7 and poly.density == 1.
    assert any(p.degrees for p in poly.placements.values())
    page = np.asarray(poly.pages[0])
    assert (page[..., 3] > 0).sum() == sum(int(m.sum()) for m in masks.values())


@needs_node
def test_rotated_regions_render_exactly(tmp_path):
    folder = make_project(tmp_path / 'p')
    data = proj.load(folder)
    data['pack'] = {'page_width': 640, 'page_height': 512}     # forces rotations and interlocking
    proj.save(folder, data)
    report = build(folder)
    atlas = (folder / 'build' / 'test_spr.atlas').read_text()
    assert 'rotate:' in atlas
    rt = report['instruments']['round_trip']
    assert report['pack']['density'] == 1.0 and rt['max'] <= 2 and rt['over_8'] == 0


@needs_node
def test_idle_controls_rotate_and_scale_bones(tmp_path):
    folder = make_project(tmp_path / 'p')
    data = proj.load(folder)
    data['animations'] = {'idle': {'bones': {'halo': {'translate_y': 6, 'rotate': 4, 'scale_x': 1.05}}}}
    proj.save(folder, data)
    report = build(folder)
    idle = json.loads((folder / 'build' / 'test_spr.json').read_text())['animations']['Idle_01']['bones']['halo']
    assert set(idle) == {'translate', 'rotate', 'scale'}
    assert idle['rotate'][1]['value'] == 4 and idle['scale'][1]['x'] == 1.05
    assert report['instruments']['motion']['idle-2']['revealed'] > 0


def make_images(folder, *, drift=False):
    """A full image plus edits that change only the face: two eyes and a mouth, as separate areas."""
    (folder / 'expressions').mkdir(parents=True)
    def figure(eyes, mouth):
        im = _layer(); d = ImageDraw.Draw(im)
        d.rectangle((150, 230, 250, 560), fill=(40, 40, 60, 255))
        d.ellipse((145, 80, 255, 220), fill=(255, 232, 220, 255))
        d.pieslice((140, 60, 260, 170), 180, 360, fill=(180, 150, 230, 255))
        eyes(d); mouth(d)
        return im
    def open_eyes(d, r=10):
        d.ellipse((181 - r, 140 - r, 181 + r, 140 + r), fill=(90, 60, 160, 255))
        d.ellipse((219 - r, 140 - r, 219 + r, 140 + r), fill=(90, 60, 160, 255))
    flat = lambda d: d.line((190, 190, 210, 190), fill=(200, 90, 110, 255), width=3)
    smile = lambda d: d.arc((185, 178, 215, 198), 0, 180, fill=(200, 90, 110, 255), width=3)
    closed = lambda d: (d.line((170, 142, 192, 142), fill=(90, 60, 160, 255), width=3),
                        d.line((208, 142, 230, 142), fill=(90, 60, 160, 255), width=3))
    figure(open_eyes, flat).save(folder / 'full.png')
    figure(open_eyes, flat).save(folder / 'expressions' / '01.png')
    two = figure(lambda d: open_eyes(d, 12), smile)
    if drift:
        ImageDraw.Draw(two).rectangle((160, 500, 240, 540), fill=(200, 40, 40, 255))
    two.save(folder / 'expressions' / '02.png')
    figure(closed, flat).save(folder / 'eyeclose.png')
    return folder


@needs_node
def test_images_in_build_feature_patches(tmp_path):
    folder = make_images(tmp_path / 'img')
    report = build(folder)
    derived = json.loads((folder / 'build' / 'derived' / 'derive.json').read_text())
    x0, y0, x1, y1 = derived['face_box']
    assert x0 <= 168 and x1 >= 232 and y0 <= 128 and y1 >= 198       # both eyes and the mouth
    assert derived['outside_face_pixels'] == 0
    change = report['instruments']['expression_change']
    assert change['changing_px'] > 0 and change['outside_patch_px'] == 0
    assert report['instruments']['geometry']['iou'] > .995
    manifest = json.loads((folder / 'build' / 'manifest.json').read_text())
    assert set(manifest['images']) == {'full.png', 'eyeclose.png', 'expressions/01.png', 'expressions/02.png'}
    a = np.asarray(Image.open(folder / 'build' / 'renders' / 'preview-99.png').convert('RGBA')).astype(int)
    band = a[64:68, 84:117]
    assert int((np.abs(band[..., :3] - (90, 60, 160)).sum(-1) < 30).sum()) == 0


def test_images_in_report_changes_outside_the_face(tmp_path):
    from spine_artist.derive import derive
    folder = make_images(tmp_path / 'img', drift=True)
    report = derive(folder, tmp_path / 'out')['derive']
    assert report['changed_outside_roi'] > 0 and report['drift']['02']['max'] > .5


def make_redrawn_images(folder, *, head_shift=None):
    """Edits as a redrawing image model makes them: every edit draws the face outline and the side hair a pixel
    away from full.png, besides changing the eyes and mouth (02 also blushes a cheek, below the change threshold).
    head_shift: (key, dx, dy) moves that edit's whole head (hair, face and features), not the body."""
    (folder / 'expressions').mkdir(parents=True)
    skin = (255, 232, 220, 255)
    def figure(eyes, mouth, jitter=(0, 0), blush=False, head=(0, 0)):
        im = _layer(); d = ImageDraw.Draw(im)
        d.rectangle((150, 230, 250, 560), fill=(40, 40, 60, 255))
        hx, hy = head
        jx, jy = jitter[0] + hx, jitter[1] + hy
        d.ellipse((145 + hx, 80 + hy, 255 + hx, 220 + hy), fill=skin)
        d.ellipse((145 + jx, 80 + jy, 255 + jx, 220 + jy), outline=(110, 60, 70, 255), width=2)    # face outline
        d.pieslice((140 + jx, 50 + jy, 260 + jx, 170 + jy), 180, 360, fill=(180, 150, 230, 255))
        for x in (146, 151, 249, 254):                                                       # side hair
            d.line((x + jx, 100 + jy, x + jx + (3 if x < 200 else -3), 205 + jy), fill=(120, 90, 180, 255), width=2)
        if blush:
            d.ellipse((202 + hx, 156 + hy, 226 + hx, 172 + hy), fill=(255, 218, 212, 255))
        eyes(d, hx, hy); mouth(d, hx, hy)
        return im
    def open_eyes(d, hx, hy, r=10):
        d.ellipse((181 - r + hx, 140 - r + hy, 181 + r + hx, 140 + r + hy), fill=(90, 60, 160, 255))
        d.ellipse((219 - r + hx, 140 - r + hy, 219 + r + hx, 140 + r + hy), fill=(90, 60, 160, 255))
    flat = lambda d, hx, hy: d.line((190 + hx, 190 + hy, 210 + hx, 190 + hy), fill=(200, 90, 110, 255), width=3)
    open_mouth = lambda d, hx, hy: d.ellipse((190 + hx, 182 + hy, 210 + hx, 198 + hy), fill=(150, 40, 60, 255))
    closed = lambda d, hx, hy: (d.line((170 + hx, 142 + hy, 192 + hx, 142 + hy), fill=(90, 60, 160, 255), width=3),
                                d.line((208 + hx, 142 + hy, 230 + hx, 142 + hy), fill=(90, 60, 160, 255), width=3))
    shift = {head_shift[0]: head_shift[1:]} if head_shift else {}
    figure(open_eyes, flat).save(folder / 'full.png')
    figure(open_eyes, flat, (1, 0), head=shift.get('01', (0, 0))).save(folder / 'expressions' / '01.png')
    figure(open_eyes, open_mouth, (-1, 1), blush=True, head=shift.get('02', (0, 0))).save(folder / 'expressions' / '02.png')
    figure(closed, flat, (0, -1), head=shift.get('99', (0, 0))).save(folder / 'eyeclose.png')
    return folder


FACE_BOX = [165, 122, 235, 201]


def _composites(out, keys=('01', '02', 'eyeclose')):
    body = Image.open(out / 'layers' / 'body.png').convert('RGBA')
    comps = {}
    for key in keys:
        im = body.copy()
        im.alpha_composite(Image.open(out / 'layers' / f'{key if key == "eyeclose" else "face_" + key}.png'))
        comps[key] = np.asarray(im).astype(int)
    return comps


def test_feature_patches_keep_the_redrawn_face_around_the_features(tmp_path):
    """Switching expressions changes the eyes, mouth and blush only: the redrawn outline and hair stay full.png's."""
    from spine_artist.derive import derive
    folder = make_redrawn_images(tmp_path / 'img')
    full = np.asarray(Image.open(folder / 'full.png').convert('RGBA')).astype(int)
    keep = np.zeros(full.shape[:2], bool)
    keep[60:230, 135:265] = True
    keep[114:215, 155:246] = False            # the eyes, mouth and blush, with room for margin, seam and feather
    report = derive(folder, tmp_path / 'out', {'face_box': FACE_BOX})['derive']
    comps = _composites(tmp_path / 'out')
    for key, comp in comps.items():
        assert np.array_equal(comp[keep], full[keep]), key
    assert report['outside_face_pixels'] == 0 and report['features']['face_shift']['02'] == [0, 0]
    edit = {k: np.asarray(Image.open(folder / f).convert('RGBA')).astype(int)
            for k, f in (('02', 'expressions/02.png'), ('eyeclose', 'eyeclose.png'))}
    assert np.array_equal(comps['02'][190, 200], edit['02'][190, 200])                 # the open mouth
    assert np.abs(comps['02'][164, 214] - edit['02'][164, 214]).max() <= 1              # the blush, under threshold
    assert np.array_equal(comps['01'][164, 214], full[164, 214])                         # no blush on 01
    assert np.array_equal(comps['eyeclose'][142, 181], edit['eyeclose'][142, 181])      # the closed lid
    lids = comps['eyeclose'][131:139, 172:190]
    assert int((np.abs(lids[..., :3] - (90, 60, 160)).sum(-1) < 30).sum()) == 0          # no open eye beneath
    assert max(report['features']['seam_ghost_p99'].values()) < .05
    # the whole-face patch swaps the redrawn outline and hair as well
    derive(folder, tmp_path / 'hull', {'face_box': FACE_BOX, 'patch': 'face'})
    assert any(not np.array_equal(c[keep], full[keep]) for c in _composites(tmp_path / 'hull').values())


def test_feature_patches_register_each_edit_face_to_the_body(tmp_path):
    from spine_artist.derive import derive
    folder = make_redrawn_images(tmp_path / 'img', head_shift=('02', 2, 1))
    report = derive(folder, tmp_path / 'out', {'face_box': FACE_BOX})['derive']
    reg = report['registration']['02']
    shift = report['features']['face_shift']['02']
    assert (round(reg['dx']) + shift[0], round(reg['dy']) + shift[1]) == (-2, -1)
    assert report['features']['seam_ghost_p99']['02'] < .05
    comp = _composites(tmp_path / 'out', ('02',))['02']
    edit = np.asarray(Image.open(folder / 'expressions' / '02.png').convert('RGBA')).astype(int)
    assert np.array_equal(comp[190, 200], edit[191, 202])                               # its mouth, moved back


def test_derive_rejects_an_unknown_patch_kind(tmp_path):
    from spine_artist.derive import DeriveError, derive
    folder = make_images(tmp_path / 'img')
    with pytest.raises(DeriveError, match='patch'):
        derive(folder, tmp_path / 'out', {'patch': 'hull'})


def test_official_region_pixels_come_back_upright_and_untrimmed():
    from spine_artist import atlasfile as remap
    from spine_artist.official import region_image
    original = np.zeros((20, 30, 4), np.uint8)
    original[6:16, 5:25] = np.stack(np.meshgrid(np.arange(20) * 12, np.arange(10) * 25), -1).reshape(10, 20, 2)[..., [0, 1, 0, 1]]
    original[6:16, 5:25, 2] = 77
    original[6:16, 5:25, 3] = 255
    trimmed = Image.fromarray(original).crop((5, 6, 25, 16))
    page = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    page.paste(trimmed.transpose(Image.Transpose.ROTATE_90), (3, 7))
    atlas = remap.parse_atlas('page.png\n\tsize: 64, 64\n\tfilter: Linear, Linear\nr\n\tbounds: 3, 7, 20, 10\n'
                              '\toffsets: 5, 4, 30, 20\n\trotate: 90\n')
    region = next(iter(atlas.regions.values()))
    out = np.asarray(region_image(atlas, region, {'page': page}))
    assert out.shape == original.shape and np.array_equal(out, original)


def make_official(folder):
    """A built layer project with a halo, standing in for the target's official portrait."""
    make_project(folder / 'src')
    build(folder / 'src', render=False)
    return folder / 'src' / 'build'


@needs_node
def test_images_in_carry_the_official_halo(tmp_path):
    from spine_artist import instruments as ins
    reference = make_official(tmp_path / 'official')
    folder = make_images(tmp_path / 'img')
    proj.save(folder, {'format': proj.FORMAT, 'name': 'test_spr',
                       'target': {'core': 'test_spr', 'reference': str(reference), 'fit': 'face'}})
    report = build(folder)
    out = folder / 'build'
    skel = json.loads((out / 'test_spr.json').read_text())
    assert skel['slots'][0]['name'] == 'halo'                    # behind the new art, as in the official
    halo_bone = next(b for b in skel['bones'] if b['name'] == 'halo')
    assert halo_bone['parent'] == 'body'                          # bobs with the body
    assert 'halo' in skel['animations']['Idle_01']['bones']
    assert report['instruments']['geometry']['iou'] > .995        # the round trip leaves the halo out
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['official']['slots']['behind'] == ['halo'] and 'test_spr.json' in manifest['official']['files']

    def halo_vertices(atlas, skeleton):
        base = ins.pose_all(atlas, skeleton, idle_frames=0, blink_frames=0)['poses'][0]['geometry']
        return np.asarray(next(a['vertices'] for a in base['attachments'] if a['slot'] == 'halo'))
    official = halo_vertices(reference / 'test_spr.atlas', reference / 'test_spr.json')
    ours = halo_vertices(out / 'test_spr.atlas', out / 'test_spr.json')
    assert np.abs(ours - official).max() < .01                   # the same place in the world

    data = proj.load(folder)
    data['target']['halo'] = {'offset': [5, -3]}
    proj.save(folder, data)
    build(folder, render=False)
    moved = halo_vertices(out / 'test_spr.atlas', out / 'test_spr.json').reshape(-1, 2)
    assert np.abs(moved - official.reshape(-1, 2) - (5, -3)).max() < .01


def test_validate_rejects_both_halos(tmp_path):
    folder = make_project(tmp_path / 'p')
    data = proj.load(folder)
    data['target'] = {'core': 'x', 'reference': '.', 'halo': 'official'}
    data['layers'][0].update(slot='halo', bone='body')
    data['bones'] = [{'name': 'root', 'at': [0, 0]}, {'name': 'body', 'parent': 'root', 'at': [0, 0]}]
    errors = proj.validate(folder, data)
    assert any('a halo layer and the official halo' in e for e in errors)


def test_images_in_find_the_face_in_a_regenerated_picture(tmp_path):
    """An image model redraws every stroke of an edit: thin changes all over the figure must not move the patch."""
    from spine_artist.derive import derive
    folder = make_images(tmp_path / 'img')
    rng = np.random.default_rng(7)
    for f in [*(folder / 'expressions').glob('*.png'), folder / 'eyeclose.png']:
        im = Image.open(f).convert('RGBA'); d = ImageDraw.Draw(im)
        for _ in range(40):
            x, y = int(rng.integers(150, 250)), int(rng.integers(240, 550))
            d.line((x, y, x + int(rng.integers(-30, 30)), y + int(rng.integers(5, 30))), fill=(230, 230, 250, 255), width=1)
        d.rectangle((150, 229, 250, 230), fill=(40, 40, 60, 255))     # the body's edge drawn a pixel higher
        im.save(f)
    report = derive(folder, tmp_path / 'out')['derive']
    x0, y0, x1, y1 = report['face_box']
    assert x0 >= 120 and x1 <= 280 and y1 <= 240 and y0 <= 128     # the face, not the body
    assert report['changed_outside_roi'] > 0


def test_images_in_stop_when_the_changes_do_not_fit_the_face(tmp_path):
    """Without a face box, changes far larger than the character's face stop the derivation with a grid to read."""
    from spine_artist.derive import DeriveError, derive
    folder = make_images(tmp_path / 'img')
    with pytest.raises(DeriveError, match='face_box'):
        derive(folder, tmp_path / 'out', face_share=.02)          # the eyes are about .08 of the figure height
    assert (tmp_path / 'out' / 'locate.png').is_file()
    report = derive(folder, tmp_path / 'out2', face_share=.08)['derive']
    assert report['face_share']['found'] == pytest.approx(.08, abs=.02)
    report = derive(folder, tmp_path / 'out3', {'face_box': [140, 100, 260, 230]}, face_share=.02)['derive']
    assert report['warnings'] and report['face_box'][3] <= 240


def test_fill_colour_keeps_drawn_pixels_and_fills_the_background():
    from spine_artist.upscale import fill_colour
    a = np.zeros((40, 50, 4), np.float32)
    a[10:20, 10:20] = (1, 0, 0, 1)
    a[10:20, 30:40] = (0, 0, 1, .5)
    filled = fill_colour(a)
    assert np.array_equal(filled[10:20, 10:20], a[10:20, 10:20, :3])
    assert np.array_equal(filled[10:20, 30:40], a[10:20, 30:40, :3])     # straight colour of soft pixels kept
    assert np.array_equal(filled[15, 9], (1, 0, 0)) and np.array_equal(filled[15, 41], (0, 0, 1))
    assert filled[15, 0, 0] > .9 and filled[15, 49, 2] > .9               # the nearest drawn colour, far out


def test_upscale_is_proposed_below_the_soft_resolution():
    from spine_artist import upscale
    assert upscale.resolve({}, .69)['scale'] == 2 and upscale.resolve({}, .69)['proposed']
    assert upscale.resolve({}, .9)['scale'] == 1 and upscale.resolve({}, None)['scale'] == 1
    assert upscale.resolve({'upscale': {'scale': 4}}, .9)['scale'] == 4
    assert upscale.validate({'scale': 3}) and upscale.validate({'scale': 2, 'tile': 1})
    assert not upscale.validate({'scale': 2, 'noise': -1, 'model': 'cunet'})


FAKE_WAIFU2X = '''#!{python}
import pathlib, sys
from PIL import Image
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
calls = pathlib.Path(__file__).with_name('calls')
calls.write_text(str(int(calls.read_text()) + 1 if calls.exists() else 1))
s = int(args['-s'])
for f in sorted(pathlib.Path(args['-i']).glob('*.png')):
    im = Image.open(f)
    im.resize((im.width * s, im.height * s), Image.Resampling.NEAREST).save(pathlib.Path(args['-o']) / f.name)
'''


def fake_waifu2x(folder):
    import os
    import sys
    folder.mkdir(parents=True)
    (folder / 'models-cunet').mkdir()
    tool = folder / 'waifu2x-ncnn-vulkan'
    tool.write_text(FAKE_WAIFU2X.format(python=sys.executable))
    os.chmod(tool, 0o755)
    return tool


@needs_node
def test_images_in_upscale_before_cutting(tmp_path, monkeypatch):
    tool = fake_waifu2x(tmp_path / 'tool')
    monkeypatch.setenv('SPINE_ARTIST_WAIFU2X', str(tool))
    plain = make_images(tmp_path / 'plain')
    build(plain, render=False)
    folder = make_images(tmp_path / 'img')
    proj.save(folder, {'format': proj.FORMAT, 'name': folder.name, 'upscale': {'scale': 2}})
    report = build(folder)
    out = folder / 'build'
    derived = json.loads((out / 'derived' / 'derive.json').read_text())
    assert derived['canvas_scale'] == 2
    assert derived['face_box'] == json.loads((plain / 'build' / 'derived' / 'derive.json').read_text())['face_box']
    assert Image.open(out / 'derived' / 'layers' / 'face_02.png').size == (2 * W, 2 * H)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['controls']['canvas'] == {'width': 2 * W, 'height': 2 * H}
    assert manifest['upscale']['scale'] == 2 and set(manifest['upscale']['outputs']) == {'full', '01', '02', '99'}
    assert report['instruments']['geometry']['iou'] > .995
    assert (tool.parent / 'calls').read_text() == '1'
    build(folder, render=False)
    assert (tool.parent / 'calls').read_text() == '1'                 # cached by input, tool and arguments
    monkeypatch.delenv('SPINE_ARTIST_WAIFU2X')
    with pytest.raises(BuildError, match='waifu2x'):
        build(folder, render=False)


def test_mesh_band_ends_on_the_cut_line():
    from spine_artist.mesh import rasterize
    alpha = np.zeros((200, 160), np.uint8)
    alpha[20:180, 40:120] = 255
    alpha[60:140, 10:150] = 128
    top, bottom = build_mesh(alpha, MeshParams(), band=(None, 100)), build_mesh(alpha, MeshParams(), band=(100, None))
    assert top.vertices[:, 1].max() == 100 and bottom.vertices[:, 1].min() == 100
    a, b = rasterize(top.polygons, alpha.shape), rasterize(bottom.polygons, alpha.shape)
    assert not (a & b).any()                                  # no pixel drawn twice
    assert not ((alpha > 0) & ~(a | b)).any()                 # no drawn pixel left out
    with pytest.raises(ValueError, match='cell boundary'):
        build_mesh(alpha, MeshParams(), band=(None, 101))


def _textured_body(folder):
    """Stripes and a gradient across the body, so any shift or resampling difference at a cut shows."""
    body = Image.open(folder / 'layers' / 'body.png').convert('RGBA')
    a = np.asarray(body).copy()
    ys, xs = np.mgrid[:H, :W]
    drawn = a[..., 3] > 0
    a[..., 0][drawn] = (ys[drawn] * 3) % 256
    a[..., 1][drawn] = np.where(((xs + ys) // 5) % 2 == 0, 230, 40)[drawn]
    Image.fromarray(a, 'RGBA').save(folder / 'layers' / 'body.png')


@needs_node
@pytest.mark.parametrize('density', [1., .6])
def test_cut_layers_render_like_one_region(tmp_path, density):
    renders = {}
    for name, cuts in (('whole', {}), ('cut', {'body': [300]})):
        folder = make_project(tmp_path / name)
        _textured_body(folder)
        data = proj.load(folder)
        data.update(cuts=cuts, pack={'density': density, 'allow_shrink': False})
        proj.save(folder, data)
        build(folder)
        a = np.asarray(Image.open(folder / 'build' / 'renders' / 'pose-base.png').convert('RGBA')).astype(int)
        a[..., :3] = a[..., :3] * a[..., 3:] // 255           # colour under clear pixels is not drawn
        renders[name] = a
    skel = json.loads((tmp_path / 'cut' / 'build' / 'test_spr.json').read_text())
    assert [s['name'] for s in skel['slots']][:3] == ['halo', 'body_1', 'body_2']
    manifest = json.loads((tmp_path / 'cut' / 'build' / 'manifest.json').read_text())
    assert manifest['controls']['cuts'] == {'body': [300]}
    diff = np.abs(renders['cut'] - renders['whole'])
    if density == 1:
        assert diff.max() <= 1                                    # pixel for pixel one region
    else:
        band = diff[296:304]                                      # the rows at the cut
        assert band[..., 3].max() <= 2 and np.percentile(band[..., :3], 99) <= 12


@needs_node
def test_cuts_are_proposed_when_a_layer_exceeds_the_page(tmp_path):
    density = {}
    for name, extra in (('cut', {}), ('whole', {'cuts': {}})):
        folder = make_project(tmp_path / name)
        data = proj.load(folder)
        data.update(pack={'page_width': 400, 'page_height': 400}, **extra)
        proj.save(folder, data)
        report = build(folder, render=False)
        density[name] = report['pack']['density']
        manifest = json.loads((folder / 'build' / 'manifest.json').read_text())
        if name == 'cut':
            rows = manifest['controls']['cuts']['body']
            assert len(rows) == 1 and 200 < rows[0] < 400 and rows[0] % 2 == 0
    assert density['cut'] > density['whole'] and density['cut'] >= .99


class _Variant:
    def __init__(self, key, spr, names, aliases=()):
        self.key, self.spr, self.aliases = key, spr, list(aliases)
        self.names = {lang: {'full': n, 'short': n} for lang, n in names.items()}


class _Family:
    def __init__(self, *variants):
        self.variants = list(variants)


class _Catalog:
    def families(self):
        return [_Family(_Variant('utaha', 'utaha_spr', {'en': 'Utaha', 'zh-CN': '歌原', 'ja': 'ウタハ'}),
                        _Variant('utahaturret', 'utahaturret_spr', {'en': 'Utaha (Thundergun)', 'zh-CN': '歌原（雷酱）'})),
                _Family(_Variant('ch0187', 'CH0187_spr', {'en': 'Someone (New)'}))]


class _Game:
    """The game reader's two queries: installed portrait cores and a core's files (the game's own spelling)."""
    spelled = {'utaha_spr': 'utaha_spr', 'utahaturret_spr': 'utahaturret_spr', 'ch0187_spr': 'CH0187_spr'}

    def __init__(self, folder):
        self.folder = folder

    def spine_cores(self, kind):
        assert kind == 'spr'
        return {core: [self.folder / f'assets-{core}-{i}.bundle' for i in range(2)] for core in self.spelled}

    def spine_files(self, kind, core):
        name = self.spelled[core.lower()]
        return {f'{name}.skel': b'skel', f'{name}.atlas': f'{name}.png\n'.encode(), f'{name}.png': b'png'}


def _upstream(game_dir, found=True):
    return {'find_game_dirs': lambda region: [game_dir] if found else [], 'search_dirs': lambda d: [d],
            'game': lambda dirs: _Game(dirs[0]), 'catalog': lambda game, region: _Catalog()}


def test_target_needs_the_game_folder_or_leave_to_search(tmp_path):
    from spine_artist.target import TargetError, resolve
    game = tmp_path / 'game'; game.mkdir()
    with pytest.raises(TargetError, match='--game DIR.*--search'):
        resolve(tmp_path / 'p', 'Utaha', up=_upstream(game))
    with pytest.raises(TargetError, match='no global install'):
        resolve(tmp_path / 'p', 'Utaha', search=True, up=_upstream(game, found=False))
    with pytest.raises(TargetError, match='no game folder'):
        resolve(tmp_path / 'p', 'Utaha', game=tmp_path / 'missing', up=_upstream(game))


def test_target_finds_a_character_by_any_name_and_sets_the_project(tmp_path):
    from spine_artist.target import resolve, bundles_of
    game = tmp_path / 'game'; game.mkdir()
    folder = tmp_path / 'p'; folder.mkdir()
    proj.save(folder, {'format': proj.FORMAT, 'derive_params': {'face_box': [1, 2, 3, 4]}, 'target': {'halo': 'none'}})
    result = resolve(folder, '歌原', search=True, up=_upstream(game))
    assert result['core'] == 'utaha_spr' and result['bundles'] == 2
    ref = folder / 'reference' / 'utaha_spr'
    assert {p.name for p in ref.iterdir()} == {'utaha_spr.skel', 'utaha_spr.atlas', 'utaha_spr.png', 'target.json'}
    data = json.loads((folder / 'project.json').read_text())
    assert data['name'] == 'utaha_spr' and data['derive_params'] == {'face_box': [1, 2, 3, 4]}
    assert data['target'] == {'halo': 'none', 'core': 'utaha_spr', 'reference': 'reference/utaha_spr', 'fit': 'face'}
    assert bundles_of(ref) == [game / f'assets-utaha_spr-{i}.bundle' for i in range(2)]
    assert resolve(folder, 'utaha (thundergun)', game=game, up=_upstream(game))['core'] == 'utahaturret_spr'
    assert resolve(folder, 'ch0187', game=game, up=_upstream(game))['core'] == 'CH0187_spr'     # the game's spelling
    assert json.loads((folder / 'project.json').read_text())['name'] == 'CH0187_spr'


def test_target_lists_the_candidates_when_several_match(tmp_path):
    from spine_artist.target import Ambiguous, TargetError, resolve
    game = tmp_path / 'game'; game.mkdir()
    with pytest.raises(Ambiguous) as e:
        resolve(tmp_path / 'p', 'uta', game=game, up=_upstream(game))
    assert [c['core'] for c in e.value.candidates] == ['utaha_spr', 'utahaturret_spr']
    assert e.value.candidates[1]['names']['zh-CN'] == '歌原（雷酱）'
    with pytest.raises(TargetError, match='no portrait matches'):
        resolve(tmp_path / 'p', 'nobody', game=game, up=_upstream(game))
    assert not (tmp_path / 'p' / 'project.json').exists()


def test_mod_takes_the_bundles_target_recorded(tmp_path):
    from spine_artist.mod import target_bundles
    from spine_artist.target import TargetError, resolve
    game = tmp_path / 'game'; game.mkdir()
    folder = tmp_path / 'p'; folder.mkdir()
    resolve(folder, 'Utaha', game=game, up=_upstream(game))
    out = tmp_path / 'build'; out.mkdir()
    (out / 'manifest.json').write_text(json.dumps({'reference': str(folder / 'reference' / 'utaha_spr')}))
    assert target_bundles(out) == [game / f'assets-utaha_spr-{i}.bundle' for i in range(2)]
    (out / 'manifest.json').write_text(json.dumps({}))
    with pytest.raises(TargetError, match='--bundles'):
        target_bundles(out)


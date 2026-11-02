"""The target portrait's own expressions, to know which edits to ask for and what each number means.

  uv run python skills/spine-artist/scripts/target_expressions.py <project>

Reads the project's `target` (set by `spine-artist target`), prints the numbered animations as JSON and renders them
into <reference folder>/faces: NN.png (the face in that expression), sheet.png (all of them, labelled) and figure.png
(the whole portrait).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spine_artist import instruments as ins, project as proj
from spine_artist.__main__ import STOPS
from spine_artist.reference import load_reference, render_world

FACE_MARGIN = .35       # of the face box's larger side, as compare-face.png
FACE_PX = 512
FIGURE_PX = 1600


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('project', type=Path)
    a = ap.parse_args(argv)
    try:
        target = proj.load(a.project).get('target')
        if not target:
            raise proj.ProjectError('the project has no target: run `spine-artist target` first')
        folder = proj.resolve(a.project, target['reference'])
        ref = load_reference(folder, target['core'])
    except STOPS as e:
        print(json.dumps({'errors': [str(e)]}, indent=1, ensure_ascii=False))
        return 1
    out = folder / 'faces'
    out.mkdir(exist_ok=True)
    render_world(ref.poses['base'], ref.pages, ref.bounds, FIGURE_PX).save(out / 'figure.png')
    names = sorted(n for n in ref.poses if n.isdigit())
    if ref.face:
        l, b, r, t = ref.face
        m = FACE_MARGIN * max(r - l, t - b)
        frame = (l - m, b - m, r + m, t + m)
        faces = {n: render_world(ref.poses[n], ref.pages, frame, FACE_PX) for n in names}
        for n, image in faces.items():
            image.save(out / f'{n}.png')
        if faces:
            ins.sheet(list(faces.items()), cell=300, columns=5).save(out / 'sheet.png')
    print(json.dumps({'core': target['core'], 'expressions': [n for n in names if n != '99'],
                      'eye_close': '99' in names, 'faces': str(out)}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())

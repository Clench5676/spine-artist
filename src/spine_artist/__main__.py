"""Command line for drivers: propose the default controls, then build and read the instruments.

  python -m spine_artist propose <project> [--write]
  python -m spine_artist build <project> [--out DIR] [--no-render] [--idle-frames N]
  python -m spine_artist target <project> <character or core> [--game DIR | --search] [--region global|japan]
  python -m spine_artist mod <build dir> [--bundles <the core's game bundles...>] --out DIR
  python -m spine_artist license [--confirm]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import project as proj, runtime
from .build import BuildError, build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='spine_artist')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('propose', help='fill missing decisions with defaults')
    p.add_argument('project', type=Path)
    p.add_argument('--write', action='store_true', help='save the proposals into project.json')
    b = sub.add_parser('build', help='build the skeleton, atlas, instruments and manifest')
    b.add_argument('project', type=Path)
    b.add_argument('--out', type=Path)
    b.add_argument('--no-render', action='store_true')
    b.add_argument('--idle-frames', type=int, default=8)
    b.add_argument('--blink-frames', type=int, default=3)
    lic = sub.add_parser('license', help='show the Spine license notice; --confirm that you hold a Spine license')
    lic.add_argument('--confirm', action='store_true')
    t = sub.add_parser('target', help="make a game portrait the target: find it, extract its files, set project.json")
    t.add_argument('project', type=Path)
    t.add_argument('query', help='a character name in any language, a skin included, or a core (utaha_spr)')
    t.add_argument('--game', type=Path, help='the game folder')
    t.add_argument('--search', action='store_true', help='search the usual install places (the user allows it)')
    t.add_argument('--region', default='global', choices=('global', 'japan'))
    m = sub.add_parser('mod', help="pack a build into the target core's bundles")
    m.add_argument('build', type=Path)
    m.add_argument('--bundles', type=Path, nargs='+', help="default: the bundles `target` recorded")
    m.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'license':
        print(runtime.NOTICE)
        if a.confirm:
            print(f'Confirmed: {runtime.confirm()}')
        elif not runtime.confirmed():
            print('Not confirmed. Confirm that you hold a Spine license: python -m spine_artist license --confirm')
        return 0
    if a.cmd == 'target':
        from .target import Ambiguous, TargetError, resolve
        try:
            print(json.dumps(resolve(a.project, a.query, game=a.game, search=a.search, region=a.region),
                             indent=1, ensure_ascii=False))
        except Ambiguous as e:
            print(json.dumps({'errors': [str(e)], 'candidates': e.candidates}, indent=1, ensure_ascii=False))
            return 2
        except TargetError as e:
            print(json.dumps({'errors': [str(e)]}, indent=1, ensure_ascii=False))
            return 1
        return 0
    if a.cmd == 'mod':
        from .mod import ModError, pack_mod, target_bundles
        from .target import TargetError
        try:
            bundles = a.bundles or target_bundles(a.build)
            print(json.dumps(pack_mod(a.build, bundles, a.out), indent=1))
        except (ModError, TargetError) as e:
            print(json.dumps({'errors': [str(e)]}, indent=1))
            return 1
        return 0
    if a.cmd == 'propose':
        data = proj.propose(a.project, proj.load(a.project))
        errors = proj.validate(a.project, data)
        if a.write:
            proj.save(a.project, data)
        print(json.dumps({'controls': data, 'errors': errors}, indent=1, ensure_ascii=False))
        return 1 if errors else 0
    try:
        report = build(a.project, a.out, render=not a.no_render, idle_frames=a.idle_frames,
                       blink_frames=a.blink_frames)
    except BuildError as e:
        print(json.dumps({'errors': e.errors}, indent=1, ensure_ascii=False))
        return 1
    except runtime.RuntimeUnavailable as e:
        print(json.dumps({'errors': [str(e)]}, indent=1, ensure_ascii=False))
        return 1
    summary = {'warnings': report['warnings'], 'pack': report['pack']}
    if 'instruments' in report:
        summary['round_trip'] = report['instruments']['round_trip']
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())

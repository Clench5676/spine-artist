"""The built-in target, Blue Archive portraits: find the game, the character's portrait core and its files.

`spine-artist target <project> <character>` reads the game with the `ba_modding_toolkit` package, read only: the
install folder (given, or searched in the usual Steam and Wine places when the user allows it), the portrait cores
it holds and the character names in every language the game has. The query is a name in any of them, a skin
included ("Utaha", "歌原（雷酱）"), or a core (`utaha_spr`). One match writes the core's skeleton, atlas and pages
into the project's reference folder, with `target.json` (the game folder and the core's bundles, which `mod`
packs into), and sets `name` and `target` in project.json; several matches are listed for the driver to pick.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from . import project as proj

REGIONS = ('global', 'japan')
TARGET_FILE = 'target.json'


class TargetError(ValueError):
    pass


class Ambiguous(TargetError):
    def __init__(self, query: str, candidates: list[dict]):
        super().__init__(f'{len(candidates)} portraits match {query!r}: run again with one core')
        self.candidates = candidates


def _upstream():
    """The game reader's entry points (ba_modding_toolkit), imported on use."""
    try:
        from ba_modding_toolkit.catalog import Catalog
        from ba_modding_toolkit.game_paths import find_game_dirs
        from ba_modding_toolkit.gameassets.cache import AssetCache
        from ba_modding_toolkit.gameassets.service import GameAssets
        from ba_modding_toolkit.searching import get_search_dirs
    except ImportError as e:
        raise TargetError('the target step needs the ba_modding_toolkit package on the path') from e
    return {'find_game_dirs': find_game_dirs, 'search_dirs': get_search_dirs,
            'game': lambda dirs: GameAssets(dirs, AssetCache()),
            'catalog': lambda game, region: Catalog(game, None, server=region)}


def find_game(up: dict, game: Path | None, search: bool, region: str) -> Path:
    """The game folder: the one given, else the first install found when searching is allowed."""
    if game is not None:
        game = Path(game).expanduser()
        if not game.is_dir():
            raise TargetError(f'no game folder at {game}')
        return game
    if not search:
        raise TargetError('give the game folder (--game DIR), or allow a search of the usual install places '
                          '(--search)')
    found = up['find_game_dirs'](region)
    if not found:
        raise TargetError(f'no {region} install found in the usual places: give the game folder (--game DIR)')
    return Path(found[0])


def _norm(text: str) -> str:
    """Case, spaces, bracket and separator forms folded, so names match however they are typed."""
    text = text.casefold().replace('（', '(').replace('）', ')')
    return re.sub(r'[\s,，、·・_\-]+', '', text)


def portraits(up: dict, game_dir: Path, region: str):
    """The game reader and every installed portrait core with its names: [{core, key, names, bundles}]."""
    dirs = up['search_dirs'](game_dir)
    if not dirs:
        raise TargetError(f'{game_dir} holds no game resources')
    game = up['game'](dirs)
    cores = game.spine_cores('spr')
    named = {}
    for family in up['catalog'](game, region).families():
        for v in family.variants:
            if v.spr and v.spr.lower() in cores:
                names = {lang: (n.get('full') if isinstance(n, dict) else n) for lang, n in (v.names or {}).items()}
                named[v.spr.lower()] = {'key': v.key, 'names': {k: n for k, n in names.items() if n},
                                        'aliases': list(getattr(v, 'aliases', []) or [])}
    out = []
    for core in sorted(cores):
        entry = named.get(core, {'key': None, 'names': {}, 'aliases': []})
        out.append({'core': core, **entry, 'bundles': [str(p) for p in cores[core]]})
    return game, out


def match(query: str, entries: list[dict]) -> list[dict]:
    """Entries named exactly by the query (core, key or any name), else those whose names contain it."""
    q = _norm(query)
    if not q:
        return []
    def words(e):
        return [e['core'], e['core'].removesuffix('_spr'), e['key'] or '', *e['names'].values(), *e['aliases']]
    exact = [e for e in entries if q in {_norm(w) for w in words(e) if w}]
    if exact:
        return exact
    return [e for e in entries if any(q in _norm(w) for w in words(e) if w)]


def resolve(folder: Path, query: str, *, game: Path | None = None, search: bool = False,
            region: str = 'global', up: dict | None = None) -> dict:
    """Make the query's portrait the project's target; raises Ambiguous with the candidates when several match."""
    if region not in REGIONS:
        raise TargetError(f'region must be one of {REGIONS}')
    folder = Path(folder)
    up = up or _upstream()
    game_dir = find_game(up, game, search, region)
    reader, entries = portraits(up, game_dir, region)
    found = match(query, entries)
    if not found:
        raise TargetError(f'no portrait matches {query!r} in {game_dir}')
    if len(found) > 1:
        raise Ambiguous(query, [{k: e[k] for k in ('core', 'names')} for e in found])
    entry = found[0]
    files = reader.spine_files('spr', entry['core'])
    skel = next((n for n in files if n.lower() == f'{entry["core"]}.skel'), None)
    if skel is None or not any(n.lower().endswith('.atlas') for n in files):
        raise TargetError(f'{entry["core"]}: the skeleton or the atlas is missing in the game files')
    core = skel[:-len('.skel')]                     # the game's own spelling, which the bundles use
    reference = folder / 'reference' / core
    reference.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (reference / Path(name).name).write_bytes(data)
    record = {'core': core, 'names': entry['names'], 'game': str(game_dir), 'region': region,
              'bundles': entry['bundles'],
              'files': {Path(n).name: hashlib.sha256(d).hexdigest() for n, d in sorted(files.items())}}
    (reference / TARGET_FILE).write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    path = folder / 'project.json'
    data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'format': proj.FORMAT}
    target = {**(data.get('target') or {}), 'core': core, 'reference': f'reference/{core}'}
    target.setdefault('fit', 'face')
    data.update(name=core, target=target)
    proj.save(folder, data)
    return {'core': core, 'names': entry['names'], 'reference': str(reference), 'bundles': len(entry['bundles'])}


def bundles_of(reference: Path) -> list[Path]:
    """The core's bundles recorded by `target`, for `mod`."""
    path = Path(reference) / TARGET_FILE
    if not path.is_file():
        raise TargetError(f'{path} is missing: run `spine-artist target`, or pass --bundles')
    return [Path(p) for p in json.loads(path.read_text(encoding='utf-8'))['bundles']]

"""The vendored Spine runtime (spine-player 4.2.120, vendor/) and the user's Spine license confirmation.

Posing, the round-trip checks and reading official bones run the runtime's public API under Node. The runtime
is licensed under the Spine Runtimes License Agreement, not under this project's license, and its use requires
that each user hold a valid Spine license. It is loaded only after the user confirms holding one:
`python -m spine_artist license --confirm` (kept in the user's config folder), or SPINE_ARTIST_SPINE_LICENSE=confirmed
for the current environment.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ENV = 'SPINE_ARTIST_SPINE_LICENSE'
NOTICE = ('spine-artist poses skeletons with the Spine Runtimes of Esoteric Software LLC, licensed under the '
          'Spine Runtimes License Agreement (https://esotericsoftware.com/spine-runtimes-license). Use of the Spine '
          'Runtimes requires that each user hold a valid Spine license.')


class RuntimeUnavailable(RuntimeError):
    pass


class LicenseError(RuntimeUnavailable):
    pass


def confirmation_file() -> Path:
    if sys.platform == 'win32' and os.environ.get('APPDATA'):
        base = Path(os.environ['APPDATA'])
    else:
        base = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config')
    return base / 'spine-artist' / 'spine-license'


def confirmed() -> bool:
    return os.environ.get(ENV) == 'confirmed' or confirmation_file().is_file()


def confirm() -> Path:
    path = confirmation_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'confirmed {time.strftime("%Y-%m-%d")}: {NOTICE}\n', encoding='utf-8')
    return path


def node(script: Path, *args) -> subprocess.CompletedProcess:
    """Run one of the package's Node scripts, which load the vendored runtime."""
    if not confirmed():
        raise LicenseError(f'{NOTICE} Confirm that you hold one: python -m spine_artist license --confirm '
                           f'(or {ENV}=confirmed).')
    exe = shutil.which('node')
    if not exe:
        raise RuntimeUnavailable('node is required for posing')
    return subprocess.run([exe, str(script), *map(str, args)], capture_output=True, text=True)

# Spine Artist

Turns images of a character into a working Spine 4.2 portrait with expression and blink animations. A full-body
image, one image per expression and an eye-close image, from any image model or artist, become a new skeleton,
atlas and texture page. With a target, the portrait is placed like a game's own portrait and packed into its
bundles; the built-in target is Blue Archive (`*_spr`).

## What it does

- Finds the target portrait by a character's name in any language the game has, and extracts its files.
- Mattes a `full.png` drawn on a background with a character segmentation model.
- Derives the layers from the images: the body, one feature patch per expression (eyes, brows, mouth and blush,
  one shape for all) and the eye-close patch; the rest of the face stays the body's.
- Builds hull meshes, a polygon-packed atlas and a Spine 4.2 skeleton with the target's animation names
  (`Idle_01`, `00`…`NN`, `99`, `Eye_Close_01`), plus the binary `.skel`.
- Puts the face where the target portrait's face is, at its size, so dialogue framing and emoticons keep working,
  and carries the target's halo over unchanged.
- Enlarges small images with waifu2x and cuts layers taller than a page into seamless pieces.
- Writes renders, a report (round trip, geometry, texture resolution against the target portrait, pixels that
  change between expressions) and a manifest with every input hash and decision.
- Packs a build into the target core's bundles (`mod`).

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/), Node.js (posing with the Spine runtime).
- [SpineSkeletonDataConverter](https://github.com/wang606/SpineSkeletonDataConverter) for the binary skeleton
  (`SPINE_ARTIST_CONVERTER`).
- Optional: [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) (`SPINE_ARTIST_WAIFU2X`, with its
  `models-*` folders beside it).
- Optional, for a `full.png` on a background: the `matte` extra (`uv sync --extra matte`, onnxruntime) and a character
  segmentation model, [anime-seg](https://huggingface.co/skytnt/anime-seg) `isnetis.onnx` (`SPINE_ARTIST_MATTE`).
- For `target` and `mod`: the `ba_modding_toolkit` package importable, and the game folder.

## Usage

```bash
uv sync
uv run spine-artist license --confirm
uv run spine-artist target <project> "<character or core>" --game <game folder>     # with a target
uv run spine-artist build <project>
uv run spine-artist mod <project>/build --out <project>/mod
```

A project folder holds `full.png`, `expressions/01.png` … `NN.png`, `eyeclose.png` and an optional
`project.json` with the decisions. The controls, the renders to check and the images to ask for are described in
[skills/spine-artist/SKILL.md](skills/spine-artist/SKILL.md), the manual for AI agents and people alike.

This repository contains no game files. The target portrait used as reference is read from the game files.

## License

Spine Artist is licensed under the GNU General Public License, version 3 (the "GPLv3").

Spine Artist poses skeletons with the Spine Runtimes of Esoteric Software LLC (the "Spine Runtimes"), which are
licensed under the [Spine Runtimes License Agreement](https://esotericsoftware.com/spine-runtimes-license) and not
under the GPLv3. Use of the Spine Runtimes requires that each user hold a valid Spine license. Spine Artist
obtains the user's confirmation that such a license is held (`spine-artist license --confirm`) before it loads the
Spine Runtimes; without it, every step that poses a skeleton stops.

The holders of the copyright in the code of Spine Artist grant the following additional permission with
respect to that code.

> **Additional permission under GNU GPL version 3 section 7**
>
> If you modify this Program, or any covered work, by linking or combining it with the Spine Runtimes (or a modified version of that library), containing parts covered by the terms of the Spine Runtimes License Agreement, the licensors of this Program grant you additional permission to convey the resulting work. Corresponding Source for a non-source form of such a combination shall include the source code for the parts of the Spine Runtimes used as well as that of the covered work.

## Credits

Thanks to [Spine Runtimes](https://github.com/EsotericSoftware/spine-runtimes) (Spine Runtimes License),
[SpineSkeletonDataConverter](https://github.com/wang606/SpineSkeletonDataConverter),
[waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan), [NumPy](https://numpy.org/) and
[Pillow](https://python-pillow.github.io/).

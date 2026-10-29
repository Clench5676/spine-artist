---
name: spine-artist
description: Turn images of a character (a full-body image plus one image per expression and an eye-close image, from any image model or artist) into a working Spine 4.2 portrait with expression and blink animations, by driving the spine_artist converter and judging its renders; with a target, the portrait replaces a game's own portrait (built-in target Blue Archive, `*_spr`). Use it whenever a portrait is redrawn, restyled or newly drawn and should become a Spine portrait, when reviewing such a build, or when writing a brief for that work.
---

# Spine artist

Images in, Spine portrait out. The converter is a black box; you drive it: get the images, build, look at the
renders, change a decision or ask the image model to redraw, build again, pack.

## 1. Get the images

Put them in a project folder:

| File | What it holds |
| --- | --- |
| `full.png` | the whole character, transparent background, the default expression |
| `expressions/01.png` … `NN.png` | an **edit of full.png** that changes only the face (eyes, brows, mouth); nothing else may move |
| `eyeclose.png` | the same edit with both eyes closed |
| `halo.png` (optional) | a part that floats on its own bone (a halo) alone; without it, a build with a `target` carries the target's own |
| `blank.png` (optional) | full.png with the face left blank (skin only) |

A floating part never goes into `full.png`: it moves on its own bone. Leave room for it.

Edits, not new drawings: the converter compares each expression with `full.png`, finds the face where large areas
change (eyes, open mouths), and cuts one feature patch shape that every expression and the eye-close share, as
layered portraits do: where any edit adds or removes paint on the eyes, brows, mouth or blush, plus a margin of
skin. Strokes an edit only redraws a few pixels away (hair, the face outline) stay the body's, so switching
expressions changes the features only. Changes away from the face are ignored and reported (`derive.json`
`changed_outside_roi`, `drift`).

Image models that redraw the whole picture on every edit (most edit APIs) change hair and props more than the face:
differences cannot find it. For them, give `derive_params.face_box` [x0, y0, x1, y1] before the first build: eyes,
brows and mouth, in `full.png` pixels, read off `build/derived/locate.png` (full.png with a grid, written by every
build). Without a box the converter checks the found changes against the character's face size (with a target: its
face against its figure, factor 2) and stops with that request; `derive.json` `face_share` gives both numbers. A
seam or doubled hair at the patch edge is an image problem: ask for the edit again.

## 2. Build, look, adjust

```bash
export SPINE_ARTIST_CONVERTER=<SpineSkeletonDataConverter binary>
export SPINE_ARTIST_WAIFU2X=<waifu2x-ncnn-vulkan binary>  # optional
uv run spine-artist license --confirm                     # once: you hold a Spine license
uv run spine-artist build <project>
```

Look at `build/renders/`:
- `sheet-face.png`: every expression, eye-close (99) and the blink frames. Wrong face, a seam at the patch edge or a
  shifted face is an image problem: ask for the edit again.
- `sheet-poses.png`: idle frames; anything that jumps is a rig decision.
- `compare-figure.png`, `compare-face.png` (with a `target`): position and size against the target's portrait.
- `overlay-mesh-bones.png`: meshes and pivots.
- `report.json`: `geometry` (silhouette IoU and shift against the input, halo left out), `round_trip`,
  `pack.density`, `resolution` (texels per world unit against the target's portrait; below 0.8 it shows softer),
  `upscale`, `official` (what was carried), `instruments.expression_change` (pixels that change between expressions,
  and those outside the face patch: 0), warnings.

Decisions go in an optional `project.json` (`format: spine-artist/1`); everything missing is proposed from the
target's conventions and the resolved decisions are recorded in `build/manifest.json` (`controls`):
- `name`: the skeleton's name, with a target its core (`utaha_spr`); `target`: `{"core", "reference": <folder>,
  "fit": "face"}`. The reference folder holds the target portrait's `<core>.skel`, `<core>.atlas` and page image,
  extracted from the game files. The face lands where the target's face is, at its size (dialogue framing,
  emoticons): the large changing areas of the new expressions are matched to the target's (`frame.fit: features`).
- `target.halo`: `"official"` (default without `halo.png`: the target's halo slots, pixels, bone and float carried
  over unchanged), `"none"`, or `{"offset": [x, y], "scale", "rotate"}` (world units, degrees) to seat it over a new
  head. Check it in `compare-figure.png`.
- `animations.idle.bones.<bone>`: `translate_x/y`, `rotate`, `scale_x/y` at mid-loop (proposed: body y −2.5, halo
  +7.5); `animations.blink`: `duration` 0.133, `closed` [0.033, 0.1].
- `pack`: `page_width`/`page_height` (keep the target's page size and count), `mesh`: `tolerance`, `padding`.
- `derive_params`, always in the input images' pixels: `face_box`, `threshold`, `margin`, `feather`, `reach`, `roi`,
  and `patch`: `"features"` (default) or `"face"`, a whole-face patch per expression (the convex hull of the
  changes, cut from each edit, so it swaps the hair and outline inside it too). For `features`:
  - `jitter`: px a redrawn stroke may move and still count as the same; `speck`: smaller stroke changes are redrawn
    edges;
  - `strong` and `soft`: the smallest stroke change, and the smallest change of the local mean colour (blush), over
    `soft_radius`; `close`;
  - `seam`, `seam_reach`: the patch edge grows over pixels where an edit and the body disagree, so the feather lies
    where they agree; `face_reach`: px each edit's face may be moved onto the body.

  `derive.json` `features` gives the patch size, each edit's `face_shift`, `seam_ghost_p99` (how much the feather
  mixes two different strokes) and `missed_strong_px` (strong changes left outside the patch: 0).
- `upscale`: `{"scale": 1|2|4, "noise": -1..3, "model": "cunet"}`. The images are enlarged with waifu2x
  (`SPINE_ARTIST_WAIFU2X`, the `models-*` folders beside it) before cutting; 2 is proposed when the expected
  resolution is below 0.8 (`report.json` `upscale.estimate`). It adds pixels, not detail. After upscaling, `frame`,
  `bones`, `mesh` and `cuts` are in the enlarged canvas's pixels.
- `cuts`: `{"<layer id>": [row, ...]}`. A layer taller than a page at the expected density is cut along rows into
  pieces (`body_1`, `body_2`: own slots and regions, same bone); the meshes meet exactly on the cut and each region
  carries 16 px of real pixels across it, so no seam. Missing: proposed where the drawing is narrowest near equal
  shares; present: used as given (`{}`: no cuts). Move a cut off a detail that must not be split, if
  `pose-base.png` shows a line there.

## 3. Pack (with a target)

```bash
uv run spine-artist mod <project>/build --bundles <the core's game bundles> --out <project>/mod
```

Needs the `ba_modding_toolkit` package on the path. The mod holds only the target's bundles with the skeleton,
atlas and page replaced (RGBA32, LZ4, CRC fixed).

## Targets

The built-in target is Blue Archive portraits (`*_spr`): the animation names `Idle_01`, `00`..`NN`, `99` and
`Eye_Close_01`, their loop and blink timings, the idle proposals above, scale 0.01, the halo carried from the
reference and the bundle step. Without a `target`, a build is a plain Spine 4.2 skeleton, atlas and page with the
same animations.

## What the converter guarantees (rails)

Every drawn pixel inside a mesh; the binary `.skel` read by the runtime exactly as the JSON (else the build stops);
the target's animation names; straight alpha; page count equal to the target's for a mod. A new skeleton per
drawing; a manifest per build with every input hash.

## Limits

- One body layer, cut only along rows: no depth strata, so nothing can move in front of or behind the body.
- Feature patches come from separately redrawn edits: hair strands over the eyes or brows and the skin right around
  the features come with them and may differ slightly per expression. Bangs are not separated from the face.
- An image model's single picture (about 1.5 MP) is below the resolution of a full-figure game portrait (941x1672:
  0.69 of Blue Archive's); waifu2x restores the texel count, not the detail (at 2x with one cut: 1.07).
- Target halo slots drawn in front of the target's other slots go in front of the whole new art.
- No hair sway, physics or weighted meshes; face patches move with the body.
- A build takes about 2 min on a 1882x3344 canvas, mostly full-size renders.

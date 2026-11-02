---
name: spine-artist
description: Turn images of a character (a full-body image plus one image per expression and an eye-close image, from any image model or artist) into a working Spine 4.2 portrait with expression and blink animations, by driving the spine_artist converter and judging its renders; with a target, the portrait replaces a game's own portrait (built-in target Blue Archive, `*_spr`). Use it whenever a portrait is redrawn, restyled or newly drawn and should become a Spine portrait or a portrait mod, even if the user only says "make this picture move" or "put this drawing in the game"; when reviewing such a build; or when writing a brief or image prompts for that work. Not for rigging with weights or physics, or for editing an existing Spine project by hand.
---

# Spine artist

Images in, Spine portrait out. The converter is a black box; you drive it: pick the target, get the images, build,
look at the renders, change a decision or ask the image model to redraw, build again, pack. The instruments inform
you and do not decide: show the user the renders; whether it looks right in the game is theirs to say.

## Setup

Once per machine; see the README's Requirements for where to get each tool.

```bash
uv sync                                                    # add --extra matte for a full.png on a background
export SPINE_ARTIST_CONVERTER=<SpineSkeletonDataConverter binary>
export SPINE_ARTIST_WAIFU2X=<waifu2x-ncnn-vulkan binary>   # optional: enlarges small images
export SPINE_ARTIST_MATTE=<isnetis.onnx>                   # optional: full.png on a background
uv run spine-artist license                                # shows the notice and whether it is confirmed
```

- `license --confirm` states that the user holds a Spine license, so ask them; only they can say it. Every step that
  poses a skeleton stops without it.
- Node.js must be on the path. `target` and `mod` need the `ba_modding_toolkit` package on the path.
- The converter is needed with a target (the target's halo is read from its `.skel`, and `mod` packs the `.skel`).
  Without one, a build still writes the JSON skeleton and warns.

## 1. Pick the target (Blue Archive)

Without a target, skip this step. It comes first because the target says which expressions to ask for.

```bash
uv run spine-artist target <project> "<character, a skin included, in any language, or a core>" --game <game folder>
uv run python skills/spine-artist/scripts/target_expressions.py <project>
```

Ask the user for the game folder, or for leave to search the usual Steam and Wine places (`--search` instead of
`--game`; `--region japan` for the Japanese release). Names match in every language the game has ("Utaha", "歌原",
"Hoshino (Swimsuit)"). Several matches are listed with their names (exit 2): pick one with the user and run again
with its core. One match writes `reference/<core>/` (skeleton, atlas, pages and `target.json` with the core's
bundles) and sets `name` and `target` in `project.json`.

`target_expressions.py` prints the target's numbered animations and renders them into `reference/<core>/faces/`
(`NN.png`, `sheet.png`, `figure.png`). The game calls expressions by number, so make one edit per number, meaning
what that number means on the target.

## 2. Get the images

Put them in the project folder:

| File | What it holds |
| --- | --- |
| `full.png` | required: the whole character, the default expression; a transparent background is best |
| `expressions/01.png` … `NN.png` | at least one: an **edit of full.png** that changes only the face (eyes, brows, mouth); nothing else may move. `01` is the default face (the game's `00` plays it too): if `full.png` already shows it, copy `full.png` |
| `eyeclose.png` (or `expressions/99.png`) | required: the same edit with both eyes closed |
| `halo.png` (optional) | a part that floats on its own bone (a halo) alone, in place, on `full.png`'s canvas; without it, a build with a `target` carries the target's own |
| `blank.png` (optional) | `full.png` with the face left blank (skin only), on the same canvas; it becomes the body layer |

Other names in `expressions/` are ignored without a message, `00.png` too.

### The brief

For `full.png`, ask for:
- the whole figure with all hair and props inside the picture, a small margin at every edge, filling the frame: the
  face's size in pixels sets the texture resolution with a target;
- the largest portrait size the model has, and a real transparent background (no checkerboard, ground or shadow);
- no floating part: it moves on its own bone. A `halo.png` needs room on the canvas; a carried halo does not;
- with a target: its design from `faces/figure.png`, and the default face from `faces/01.png`.

For each edit, give the model `full.png` as the only image to edit (with a target, `faces/NN.png` for the meaning of
the expression only), and ask for:
- only the eyes, brows, mouth and, where fitting, blush to change, in this face's angle and style;
- the face outline, ears, hair, bangs, clothes, props, pose, framing and canvas size to stay exactly as they are;
- one complete picture of the same size, not a face crop or a sheet; the same transparent background;
- for the eye-close: both eyes fully closed with clean lid strokes and no open-eye detail, brows relaxed, mouth closed.

### What the converter does with them

Edits, not new drawings: the converter compares each expression with `full.png`, finds the face where large areas
change (eyes, open mouths), and cuts one feature patch shape that every expression and the eye-close share, as
layered portraits do: where any edit adds or removes paint on the eyes, brows, mouth or blush, plus a margin of
skin. Strokes an edit only redraws a few pixels away (hair, the face outline) stay the body's, so switching
expressions changes the features only. Other sizes are resized, and a shift or a scale up to 3 % is registered.

Image models that redraw the whole picture on every edit (most edit APIs) change hair and props more than the face,
so differences cannot find it. Changes away from the face are ignored and reported, so a large `changed_outside_roi`
or `drift` in `derive.json` is normal for them and no reason to ask again (0.96 on a good build).

For such models give `derive_params.face_box` [x0, y0, x1, y1]: eyes, brows and mouth, in `full.png` pixels. Read it
off `build/derived/locate.png`: `full.png` with a grid (lines every 50 px, labels every 100), written by every build,
one that stops at the face included; crop around the head to read it. With a target, the converter checks the found
changes against the target's face size (factor 2) and stops with that request and both numbers (after a build:
`face_share` in `derive.json`). Without a target nothing checks this: see that `feature_box` lies on the face.

A `full.png` on a background is matted with a character segmentation model (`SPINE_ARTIST_MATTE`:
[anime-seg](https://huggingface.co/skytnt/anime-seg) `isnetis.onnx`, a few seconds on a CPU); the expressions take its
alpha. Check the edges in `pose-base.png` (see Limits); a transparent background avoids matting.

## 3. Build

```bash
uv run spine-artist build <project>     # about 2 min on a 1882x3344 canvas: give it a timeout of several minutes
```

It prints JSON (`warnings`, `pack`, `round_trip`) and exits 0, or stops with `errors` (the reason and what to do) and
exits 1. `--no-render` skips the renders; `--out DIR` writes elsewhere. A rebuild overwrites `build/`, so move a build
to keep it (the matte and upscale results cached inside it are then computed again). `project.json` is never changed;
the resolved decisions are in `build/manifest.json` (`controls`).

## 4. Look and adjust

In `build/renders/`, read:
- `sheet-face.png`: the base, every expression, eye-close (99) and the blink frames.
- `sheet-poses.png`: every pose, with the idle frames; anything that jumps is a rig decision.
- `compare-figure.png`, `compare-face.png` (with a `target`): position and size against the target's portrait, the
  faces number by number.
- `pose-base.png` (full size), `overlay-mesh-bones.png` (meshes and pivots), `preview-*.png` (each pose).

Renders on the canvas clip a carried halo that passes its top; `compare-figure.png` shows it whole.

| What you see | Cause | What to do |
| --- | --- | --- |
| Patch far from the face, figure tiny or huge in `compare-figure.png`, or the build stops at the face | the changes were not found on the face | set `derive_params.face_box` |
| Wrong face, seam or doubled hair at the patch edge, shifted face | an image problem | ask for that edit again |
| A feature cut off at the patch edge | the face box is too small | widen `face_box` |
| `missed_strong_px` above 0 | strong changes dropped as specks | if a feature is missing, lower `speck` |
| Head changes between expressions | a whole-face patch | keep `patch: "features"` (default) |
| Halo off the new head | the head differs from the target's | `target.halo` with `offset`, `scale`, `rotate` |
| A line across the figure at a cut | a cut on a detail | move it in `cuts` |
| Rim or holes on the edge | matting | a transparent `full.png` (see Limits) |
| `resolution.ratio` below 0.8 | too few pixels for the target | waifu2x (`upscale`), a larger face in the image |
| `texture density reduced` | the page is full; normal after upscaling | look at `resolution.ratio` instead |

`build/report.json` holds:
- `warnings`, all of them;
- `instruments.geometry` (silhouette IoU and shift against the input, halo left out; the build warns below 0.995 or
  beyond 1 px) and `instruments.round_trip` (pixels; they only count at `pack.density` 1);
- `pack.density`, `resolution` (texels per world unit against the target's portrait; below 0.8 it shows softer),
  `upscale`, `matte` (model, figure share), `official` (what was carried);
- `instruments.expression_change` (pixels that change between expressions, and those outside the face patch: 0).

`build/derived/derive.json` holds `feature_box` (the changes the face was found from), `roi`, `face_box` (the patch's
box, not the control), `face_share`, `changed_outside_roi`, `drift`, `registration`, and `features`: `patch_px`, each
edit's `face_shift`, `seam_ghost_p99` (how much the feather mixes two different strokes; 0.04 to 0.07 on a good build)
and `missed_strong_px` (strong changes left outside the patch: 0).

Stop when the face sheet and the comparisons look right and the warnings are explained, then show the user.

## 5. Pack (with a target)

```bash
uv run spine-artist mod <project>/build --out <project>/mod
```

The bundles are those `target` recorded (or `--bundles <the core's game bundles>`). The mod holds only the target's
bundles with the skeleton, atlas and page replaced (RGBA32, LZ4, CRC fixed). The build must have the `.skel` and as
many pages as the target.

## Controls

Decisions go in an optional `project.json` (`format: spine-artist/1`, required in the file); everything missing is
proposed, from the target's conventions where it has them.

- `name`: the skeleton's name, with a target its core (`utaha_spr`); `target`: `{"core", "reference": <folder>,
  "fit": "face"}`. The reference folder holds the target portrait's `<core>.skel`, `<core>.atlas` and page image,
  extracted from the game files. The face lands where the target's face is, at its size (dialogue framing,
  emoticons): the large changing areas of the new expressions are matched to the target's (`frame.fit: features`).
  `fit: "height"` matches the whole figure's height instead.
- `frame`: `{"origin": [x, y], "scale"}`, canvas pixels to world units; given, it replaces the fit. Give both.
- `target.halo`: `"official"` (default without `halo.png`: the target's halo slots, pixels, bone and float carried
  over unchanged), `"none"` (default with `halo.png`, which allows nothing else), or `{"offset": [x, y], "scale",
  "rotate"}` (world units, degrees) to seat it over a new head. Check it in `compare-figure.png`.
- `expressions`: `{"list", "default"}` (all found; the first).
- `animations.idle`: `duration` 3.333; `.bones.<bone>`: `translate_x/y`, `rotate`, `scale_x/y` at mid-loop (proposed:
  body y −2.5, a `halo.png` +7.5; a carried halo keeps the target's float). `animations.blink`: `duration` 0.133,
  `closed` [0.033, 0.1]. `animations.expression`: `duration` 3.333.
- `bones`: `[{"name", "parent", "at": [x, y]}]`, the pivots in canvas pixels (proposed: `root`, `body`, `halo`).
- `pack`: `page_width`, `page_height`, `max_pages` (the target's; without one 2048, 2048, 1), `density` (1), `padding`.
  `mesh`: `tolerance` (3), `padding` (2), in pixels.
- `derive_params`, always in the input images' pixels, colour differences 0..1; defaults in `derive.json` `params`:
  - `face_box`; `roi`: without a box, how far around the found changes to look, as a share of their size;
  - `patch`: `"features"` (default) or `"face"`, a whole-face patch per expression (the convex hull of the changes,
    cut from each edit, so it swaps the hair and outline inside it too; `reach` is for this one);
  - `threshold`, `close`, `margin` (4), `feather` (4);
  - for `features`, `jitter` (2): px a redrawn stroke may move and still count as the same; `speck`: smaller stroke
    changes are redrawn edges;
  - `strong` (0.3) and `soft` (0.04): the smallest stroke change, and the smallest change of the local mean colour
    (blush), over `soft_radius`;
  - `seam`, `seam_reach`: the patch edge grows over pixels where an edit and the body disagree, so the feather lies
    where they agree; `face_reach`: px each edit's face may be moved onto the body.
- `matte`: `{"model": "isnet-anime"}`, proposed when `full.png` has no transparent background, or `"none"` (then
  such a `full.png` stops the build).
- `upscale`: `{"scale": 1|2|4, "noise": -1..3, "model": "cunet" | "upconv_7_anime_style_art_rgb"}`. The images are
  enlarged with waifu2x (`SPINE_ARTIST_WAIFU2X`, the `models-*` folders beside it) before cutting; with a target, 2 is
  proposed when the expected resolution is below 0.8 (`report.json` `upscale.estimate`), and skipped with a warning if
  the tool is missing. It adds pixels, not detail. After upscaling, `frame`, `bones`, `mesh` and `cuts` are in the
  enlarged canvas's pixels.
- `cuts`: `{"<layer id>": [row, ...]}`. A layer taller than a page at the expected density is cut along rows into
  pieces (`body_1`, `body_2`: own slots and regions, same bone); the meshes meet exactly on the cut and each region
  carries 16 px of real pixels across it, so no seam. Missing: proposed where the drawing is narrowest near equal
  shares; present: used as given (`{}`: no cuts). Move a cut off a detail that must not be split, if
  `pose-base.png` shows a line there.
- `binary`: `{"converter", "waifu2x", "matte"}`: paths, instead of the environment.

A `project.json` with `layers` skips the images step: full-canvas RGBA layers `{"id", "file", "role"}` in draw order,
bottom first, with `role` one of `halo`, `body`, `features` (plus `"expression": "NN"`), `eyeclose`, `bangs`, `prop`,
`effect`. `spine-artist propose <project> --write` then saves every proposal into the file for editing.

## Targets

The built-in target is Blue Archive portraits (`*_spr`): the animation names `Idle_01`, `00`..`NN`, `99` and
`Eye_Close_01`, their loop and blink timings, the idle proposals above, the reference found and extracted by `target`,
the halo carried from it and the bundle step. Without a `target`, a build is a plain Spine 4.2 skeleton, atlas and
page with the same animations.

## What the converter guarantees (rails)

Every drawn pixel inside a mesh; the binary `.skel` read by the runtime exactly as the JSON (else the build stops);
the target's animation names; straight alpha; page count equal to the target's for a mod. A new skeleton per
drawing; a manifest per build with every input hash.

## Limits

- From images: one body layer, cut only along rows: no depth strata, so nothing can move in front of or behind the
  body.
- Feature patches come from separately redrawn edits: hair strands over the eyes or brows and the skin right around
  the features come with them and may differ slightly per expression. Bangs are not separated from the face.
- An image model's single picture (about 1.5 MP) is below the resolution of a full-figure game portrait (941x1672:
  0.69 of Blue Archive's); waifu2x restores the texel count, not the detail (at 2x with one cut: 1.07).
- Target halo slots drawn in front of the target's other slots go in front of the whole new art.
- Matting: gaps under about 3 px (a stray strand and the head) stay filled, white parts on a white background lose
  about 1 px of edge, and a background close to the character's colours can stay in pockets.
- No hair sway, physics or weighted meshes; face patches move with the body.

"""Spine atlas text: pages and regions, in the legacy (xy/size/orig/offset) and 4.x (bounds/offsets) syntax.

Offsets are measured from the bottom left of the untrimmed original;
`rotate: 90` (or `true`) regions are stored rotated counter-clockwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


class AtlasError(ValueError):
    pass


def texture_key(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower().removesuffix(".png")


@dataclass(frozen=True)
class Region:
    name: str
    index: int
    page: str
    xy: tuple[int, int]
    size: tuple[int, int]
    original: tuple[int, int]
    offset: tuple[int, int]
    degrees: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return self.name, self.index

    @property
    def label(self) -> str:
        return self.name if self.index == -1 else f"{self.name} [{self.index}]"


@dataclass
class Page:
    name: str
    size: tuple[int, int] = (0, 0)
    pma: bool = False
    format: str = "RGBA8888"


@dataclass
class Atlas:
    pages: dict[str, Page] = field(default_factory=dict)
    regions: dict[tuple[str, str, int], Region] = field(default_factory=dict)
    file: str = field(default="—", compare=False)


def parse_atlas(text: str | bytes, file: str = "—") -> Atlas:
    """Read legacy xy/size/orig/offset and modern bounds/offsets syntax."""
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeError as error:
            raise AtlasError(f"{file}: not UTF-8") from error
    atlas = Atlas(file=file)
    page = None
    name = None
    props = {}
    page_start = True

    def invalid(key: str, value: str) -> AtlasError:
        return AtlasError(f"{file}: page {page.name if page else '—'}, region {name or '—'}: invalid {key} {value!r}")

    def numbers(key: str, default: tuple) -> tuple:
        if key not in props:
            return default
        try:
            values = tuple(int(v.strip()) for v in props[key].split(","))
        except ValueError as error:
            raise invalid(key, props[key]) from error
        if len(values) != len(default):
            raise invalid(key, props[key])
        return values

    def flush() -> None:
        if name is None:
            return
        bounds = numbers("bounds", numbers("xy", (0, 0)) + numbers("size", (0, 0)))
        offsets = numbers("offsets", numbers("offset", (0, 0)) + numbers("orig", bounds[2:]))
        rotation = props.get("rotate", "false").lower()
        degrees = {"true": 90, "false": 0}.get(rotation)
        if degrees is None:
            try:
                degrees = int(rotation) % 360
            except ValueError as error:
                raise invalid("rotate", rotation) from error
        if degrees not in (0, 90, 180, 270):
            raise invalid("rotate", rotation)
        index, = numbers("index", (-1,))
        region = Region(name, index, page.name, bounds[:2], bounds[2:],
                        offsets[2:], offsets[:2], degrees)
        if min(*region.size, *region.original) <= 0 or min(*region.xy, *region.offset) < 0:
            raise invalid("bounds/offsets", f"{bounds}/{offsets}")
        if any(o + s > original for o, s, original in zip(region.offset, region.size, region.original)):
            raise invalid("offsets/orig", f"{offsets}/{region.size}")
        key = (region.page, *region.key)
        if key in atlas.regions:
            raise AtlasError(f"{file}: region {region.label} appears twice on page {region.page}")
        atlas.regions[key] = region

    for line_number, raw in enumerate(text.lstrip("﻿").splitlines(), 1):
        line = raw.strip()
        if not line:
            flush()
            name, props, page_start = None, {}, True
        elif page_start:
            if ":" in line:
                raise AtlasError(f"{file}: line {line_number}: expected a page name, found {line!r}")
            if line in atlas.pages:
                raise AtlasError(f"{file}: page {line} appears twice")
            page = Page(line)
            atlas.pages[line] = page
            page_start = False
        elif ":" not in line:
            flush()
            name, props = line, {}
        else:
            key, value = (v.strip() for v in line.split(":", 1))
            if name is not None:
                props[key] = value
            elif key == "size":
                try:
                    page.size = tuple(int(v.strip()) for v in value.split(","))
                except ValueError as error:
                    raise invalid(key, value) from error
                if len(page.size) != 2 or min(page.size) <= 0:
                    raise invalid(key, value)
            elif key == "pma":
                page.pma = value.lower() == "true"
            elif key == "format":
                page.format = value
    flush()
    if not atlas.pages:
        raise AtlasError(f"{file}: no pages")
    return atlas


def _box(region: Region, page: Page, image: Image.Image, file: str = "—") -> tuple[int, int, int, int]:
    """The region's pixel box on the page image, scaled when the image differs from the declared page size."""
    width, height = region.size[::-1] if region.degrees % 180 else region.size
    declared = page.size if all(page.size) else image.size
    sx, sy = image.width / declared[0], image.height / declared[1]
    x, y = region.xy
    box = round(x * sx), round(y * sy), round((x + width) * sx), round((y + height) * sy)
    if box[0] < 0 or box[1] < 0 or box[2] > image.width or box[3] > image.height:
        raise AtlasError(f"{file}: region {region.label} leaves page {page.name} ({image.width}x{image.height})")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise AtlasError(f"{file}: region {region.label} has an empty box {box}")
    return box


def _rotate(image: Image.Image, degrees: int) -> Image.Image:
    operation = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180,
                 270: Image.Transpose.ROTATE_270}.get(degrees % 360)
    return image.transpose(operation) if operation else image


def _page_image(atlas: Atlas, page: str, images: dict[str, Image.Image]) -> Image.Image:
    image = images.get(texture_key(page))
    if image is None:
        raise AtlasError(f"{atlas.file}: page {page} not found")
    return image if image.mode == "RGBA" else image.convert("RGBA")

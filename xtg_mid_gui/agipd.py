"""AGIPD1M tile layout (no Qt).

AGIPD1M = 16 tiles (modules) of 512 x 128 px, each made of 8 ASICs of 64 x 128 px,
in 4 quadrants of 4 tiles.  The all-tiles view copies the assembled image Den plots
in rocking_MID.ipynb (show_detector / show_detector_multi: DAMNIT 'agipd_mean',
plt.imshow): tiles drawn horizontally (512 px across), a 2 px gap between ASICs,
a 29 px gap between tiles in a quadrant, and the four quadrants offset from each other as
in that figure (offsets measured on it, in pixels).

Tile numbering (one line each in QUADS, change it there if needed): the Bragg peak
+ TG orders of p7016 tile 15 (Bragg at pixel 158, TG at 64 and 256 along the 512
axis) sit in the top tile of the bottom-right quadrant, as in the figure.  Tiles
8-15 have pixel 0 on the left; tiles 0-7 are turned by 180 degrees.  It is drawn
from Den's figure, not from a geometry file: good to find the peaks and pick tiles.

The analysis subset stacks the chosen tiles top to bottom (ascending tile
number), with a NaN gap between them, so the usual ROI tools work on it.
"""
from __future__ import annotations

import numpy as np

TILE_SHAPE = (512, 128)        # as stored: (rows, cols) of one module per frame
N_TILES = 16
ASIC = 64                      # px of one ASIC along the 512 axis
ASIC_GAP = 2                   # px between ASICs
MOD_H = TILE_SHAPE[1]                         # 128: tile height on the canvas
MOD_W = 8 * ASIC + 7 * ASIC_GAP               # 526: tile width on the canvas (with ASIC gaps)
MOD_PITCH = 128 + 29           # tile pitch in a quadrant (extra_geom panel_gap = 29 px)
STACK_GAP = 8                  # px between tiles in the analysis stack

# quadrant: (top row, left col) on the canvas, tiles from top to bottom
QUADS = {
    "top left":     ((56, 0),    (7, 6, 5, 4)),
    "top right":    ((0, 619),   (11, 10, 9, 8)),
    "bottom left":  ((707, 67),  (3, 2, 1, 0)),
    "bottom right": ((701, 675), (15, 14, 13, 12)),
}


def module_origins() -> dict:
    """Top-left (row, col) of every tile in the all-tiles canvas."""
    out = {}
    for (r0, c0), tiles in QUADS.values():
        for k, t in enumerate(tiles):
            out[t] = (r0 + k * MOD_PITCH, c0)
    return out


def module_size() -> tuple:
    return MOD_H, MOD_W


def canvas_shape() -> tuple:
    o = module_origins().values()
    return max(r for r, _ in o) + MOD_H, max(c for _, c in o) + MOD_W


def tile_to_canvas(t, im) -> np.ndarray:
    """(512, 128) tile -> (128, 526) as drawn: 512 axis across, NaN gaps between ASICs."""
    im = np.asarray(im, np.float32).T                      # (128, 512)
    if t < 8:
        im = im[::-1, ::-1]                                # tiles 0-7 turned by 180 deg
    out = np.full((MOD_H, MOD_W), np.nan, np.float32)
    for a in range(8):
        c = a * (ASIC + ASIC_GAP)
        out[:, c:c + ASIC] = im[:, a * ASIC:(a + 1) * ASIC]
    return out


def assemble(tiles: dict) -> np.ndarray:
    """dict tile -> (512, 128) image  ->  all-tiles canvas (NaN where no data)."""
    out = np.full(canvas_shape(), np.nan, np.float32)
    for m, (r0, c0) in module_origins().items():
        if m in tiles and tiles[m] is not None:
            out[r0:r0 + MOD_H, c0:c0 + MOD_W] = tile_to_canvas(m, tiles[m])
    return out


def module_at(row: float, col: float):
    """Tile number under canvas pixel (row, col), or None."""
    for m, (r0, c0) in module_origins().items():
        if r0 <= row < r0 + MOD_H and c0 <= col < c0 + MOD_W:
            return m
    return None


# ---------------------------------------------------------------- analysis stack
def stack_origins(tiles) -> dict:
    """Top-left (row, col) of each chosen tile in the stacked analysis image."""
    mh = TILE_SHAPE[1]
    return {t: (i * (mh + STACK_GAP), 0) for i, t in enumerate(dict.fromkeys(tiles))}   # top to bottom as chosen


def stack_shape(tiles) -> tuple:
    n = len(set(tiles))
    return n * TILE_SHAPE[1] + max(n - 1, 0) * STACK_GAP, TILE_SHAPE[0]


def stack(per_tile: dict, tiles) -> np.ndarray:
    """dict tile -> (n, 512, 128) per-train images  ->  (n, H, 512) stack, NaN gaps."""
    tiles = list(dict.fromkeys(tiles))
    n = len(next(iter(per_tile.values())))
    out = np.full((n,) + stack_shape(tiles), np.nan, np.float32)
    for t, (r0, c0) in stack_origins(tiles).items():
        im = np.swapaxes(np.asarray(per_tile[t]), 1, 2)          # (n, 128, 512)
        out[:, r0:r0 + im.shape[1], c0:c0 + im.shape[2]] = im
    return out


def parse_tiles(text: str) -> list:
    """'9 8, 15' -> [9, 8, 15] (valid tile numbers only, unique, in the order typed:
    that is the top-to-bottom order of the stacked analysis image)."""
    out = []
    for tok in str(text).replace(",", " ").split():
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return [t for t in dict.fromkeys(out) if 0 <= t < N_TILES]

"""Shared detection logic for the vertical-line splitter.

Used by both `split_on_vertical_line.ipynb` and `correction_tool.py`. Keeping
this in one place guarantees the GUI's "re-trace from corrected seed" produces
exactly the same curve the notebook would produce given the same seed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default config -- mirrored in the notebook's CONFIG cell. Notebook values
# override these by passing a cfg dict; GUI uses these as-is.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict = {
    'dark_threshold': 140,
    'gray_mode': 'gray',           # 'gray' | 'blue' | 'red' | 'green'
    'search_band': (0.30, 0.70),
    'flank_width': 30,
    'trace_max_jump': 4,
    'max_gap_rows': 200,
    'max_drift_from_seed': 15,
    'edge_trim_px': 8,
    'whiten_half_width': 3,
    'letter_guard_distance': 14,
    'pad_color_hex': '#E01AD1',
}


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def to_single_channel(img_rgb: np.ndarray, mode: str = 'gray') -> np.ndarray:
    if mode == 'blue':
        return img_rgb[:, :, 2].copy()
    if mode == 'red':
        return img_rgb[:, :, 0].copy()
    if mode == 'green':
        return img_rgb[:, :, 1].copy()
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Seed search (flank-aware "thin spike" score)
# ---------------------------------------------------------------------------
def find_seed_x(img_rgb: np.ndarray, cfg: Dict) -> int:
    chan = to_single_channel(img_rgb, cfg.get('gray_mode', 'gray'))
    H, W = chan.shape
    x_lo = int(cfg['search_band'][0] * W)
    x_hi = int(cfg['search_band'][1] * W)
    dark = chan <= cfg['dark_threshold']

    col_dark = dark[:, x_lo:x_hi].sum(axis=0).astype(np.float32)
    flank = cfg.get('flank_width', 30)
    n = len(col_dark)
    pad_arr = np.pad(col_dark, flank, mode='edge')
    cs = np.concatenate(([0.0], np.cumsum(pad_arr, dtype=np.float64)))
    left_mean  = (cs[flank:flank+n]         - cs[0:n])               / flank
    right_mean = (cs[2*flank+1:2*flank+1+n] - cs[flank+1:flank+1+n]) / flank
    flank_mean = (left_mean + right_mean) / 2.0
    score = col_dark - flank_mean
    return x_lo + int(np.argmax(score))


# ---------------------------------------------------------------------------
# Trace from a known seed_x (works for both auto and manual seeds)
# ---------------------------------------------------------------------------
def trace_from_seed(img_rgb: np.ndarray, seed_x: int, cfg: Dict
                    ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Returns (xs, (y0, y1)).
        xs[y] >= 0 if the line was found at row y, else -1.
        (y0, y1) inclusive bounds of the longest gap-bridged run.
    If no run can be established, returns the empty xs and (0, -1).
    """
    chan = to_single_channel(img_rgb, cfg.get('gray_mode', 'gray'))
    H, W = chan.shape
    dark = chan <= cfg['dark_threshold']

    half = cfg['trace_max_jump']
    max_drift = cfg['max_drift_from_seed']
    max_gap = cfg['max_gap_rows']
    xs = np.full(H, -1, dtype=np.int32)

    rows_with = np.where(
        dark[:, max(0, seed_x - half):seed_x + half + 1].any(axis=1)
    )[0]
    if len(rows_with) == 0:
        return xs, (0, -1)
    seed_y = int(rows_with[len(rows_with) // 2])

    def trace(start_y: int, direction: int) -> None:
        x = seed_x
        gap = 0
        y = start_y
        while 0 <= y < H:
            lo = max(0, x - half)
            hi = min(W, x + half + 1)
            row_slice = dark[y, lo:hi]
            if row_slice.any():
                idxs = np.where(row_slice)[0] + lo
                x_new = int(idxs[np.argmin(np.abs(idxs - x))])
                if abs(x_new - seed_x) > max_drift:
                    gap += 1
                    if gap > max_gap:
                        break
                else:
                    xs[y] = x_new
                    x = x_new
                    gap = 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            y += direction

    trace(seed_y, +1)
    trace(seed_y - 1, -1)
    xs[seed_y] = seed_x

    # Bridge small gaps -> longest run
    present = xs >= 0
    bridged = present.copy()
    last_true = -1
    for i in range(H):
        if present[i]:
            if last_true >= 0 and (i - last_true - 1) <= max_gap:
                bridged[last_true + 1:i] = True
            last_true = i

    best = (0, -1)
    cur_start: Optional[int] = None
    for i in range(H):
        if bridged[i] and cur_start is None:
            cur_start = i
        elif not bridged[i] and cur_start is not None:
            if i - cur_start > best[1] - best[0]:
                best = (cur_start, i - 1)
            cur_start = None
    if cur_start is not None and (H - cur_start) > best[1] - best[0]:
        best = (cur_start, H - 1)

    y0, y1 = best
    if y1 < y0:
        return xs, best

    # Interpolate over small gaps inside the run
    ys_known = [y for y in range(y0, y1 + 1) if xs[y] >= 0]
    if ys_known:
        xs_known = [xs[y] for y in ys_known]
        for y in range(y0, y1 + 1):
            if xs[y] < 0:
                xs[y] = int(np.interp(y, ys_known, xs_known))

    # Edge trim
    trim = cfg.get('edge_trim_px')
    if trim is not None and y1 > y0:
        seg = xs[y0:y1 + 1]
        med_x = int(np.median(seg[seg >= 0]))
        new_y0 = y0
        while new_y0 <= y1 and abs(int(xs[new_y0]) - med_x) > trim:
            xs[new_y0] = -1
            new_y0 += 1
        new_y1 = y1
        while new_y1 >= new_y0 and abs(int(xs[new_y1]) - med_x) > trim:
            xs[new_y1] = -1
            new_y1 -= 1
        best = (new_y0, new_y1)

    return xs, best


def find_vertical_line(img_rgb: np.ndarray, cfg: Optional[Dict] = None,
                       seed_x_override: Optional[int] = None
                       ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Top-level convenience. Auto-seeds unless seed_x_override is given."""
    if cfg is None:
        cfg = DEFAULT_CONFIG
    seed_x = int(seed_x_override) if seed_x_override is not None else find_seed_x(img_rgb, cfg)
    xs, (y0, y1) = trace_from_seed(img_rgb, seed_x, cfg)
    return xs, (y0, y1), seed_x


# ---------------------------------------------------------------------------
# JSON I/O for starts and overlays
# ---------------------------------------------------------------------------
# Starts JSON schema:
#   {
#     "config_used": {...},                       # snapshot of cfg used to autogen
#     "entries": {
#       "<filename>": {"seed_x": int, "autogen": bool}
#     }
#   }
# Overlays JSON schema:
#   {
#     "config_used": {...},
#     "entries": {
#       "<filename>": {
#         "y0": int, "y1": int,
#         "xs": [int, ...]   # length = image height; -1 means missing
#         "autogen": bool
#       }
#     }
#   }

def load_json_safely(path: Path) -> Dict:
    if path is None or not Path(path).exists():
        return {'config_used': None, 'entries': {}}
    with open(path) as f:
        data = json.load(f)
    data.setdefault('entries', {})
    data.setdefault('config_used', None)
    return data


def save_json(path: Path, data: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def xs_to_serializable(xs: np.ndarray) -> List[int]:
    return [int(v) for v in xs]


def xs_from_serializable(arr: List[int]) -> np.ndarray:
    return np.asarray(arr, dtype=np.int32)

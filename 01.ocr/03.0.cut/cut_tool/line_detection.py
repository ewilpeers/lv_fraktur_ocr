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
    # Vertical-preference parameters: when the trace would step away from
    # seed_x, also probe a small window centred on seed_x. If a dark pixel
    # exists there, prefer it -- that is, branch back toward 'more vertical'.
    # This recovers the trace when it gets latched onto a nearby letter edge
    # while a faded segment of the real divider is still present near seed_x.
    'vertical_pref_radius': 3,        # +/- px window around seed_x to probe
    'vertical_snap_threshold': 4,     # only switch when running pick is >= this far from seed_x
    # Multi-pass refinement: after the dual-window trace, scan for segments
    # that drift from the rolling median by more than `multipass_drift_threshold`
    # pixels. For each, predict an x value at every row by linearly interpolating
    # from the clean context before/after the segment. Then for each row in the
    # segment, snap the trace to a dark pixel within `multipass_line_tolerance`
    # of the predicted x; if there is no such dark pixel, mark it as a gap (the
    # interpolation step downstream will fill it). This favors straight (or
    # gently-curved between clean anchor points) traces over crooked ones that
    # follow letter strokes. Disable by setting `multipass_passes=0`.
    'multipass_passes': 4,
    'multipass_drift_threshold': 3,
    'multipass_min_seg_len': 8,
    'multipass_line_tolerance': 1,
    'multipass_context_window': 50,
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
    vert_radius = cfg.get('vertical_pref_radius', 3)
    snap_thresh = cfg.get('vertical_snap_threshold', 4)
    xs = np.full(H, -1, dtype=np.int32)

    rows_with = np.where(
        dark[:, max(0, seed_x - half):seed_x + half + 1].any(axis=1)
    )[0]
    if len(rows_with) == 0:
        return xs, (0, -1)
    seed_y = int(rows_with[len(rows_with) // 2])

    def trace(start_y: int, direction: int) -> None:
        # Dual-window strategy:
        # 1. 'running' window of width ±half around the running x (handles curve)
        # 2. 'vertical' window of width ±vert_radius around seed_x (handles
        #    recovery: when the trace has drifted onto a letter edge, but the
        #    real divider is still present near seed_x, snap back to it)
        x = seed_x
        gap = 0
        y = start_y
        while 0 <= y < H:
            # Running pick
            lo = max(0, x - half)
            hi = min(W, x + half + 1)
            run_slice = dark[y, lo:hi]
            run_pick: Optional[int] = None
            if run_slice.any():
                idxs = np.where(run_slice)[0] + lo
                idxs = idxs[np.abs(idxs - seed_x) <= max_drift]
                if len(idxs):
                    run_pick = int(idxs[np.argmin(np.abs(idxs - x))])

            # Vertical pick (probe a small window around seed_x)
            vlo = max(0, seed_x - vert_radius)
            vhi = min(W, seed_x + vert_radius + 1)
            vert_slice = dark[y, vlo:vhi]
            vert_pick: Optional[int] = None
            if vert_slice.any():
                vidxs = np.where(vert_slice)[0] + vlo
                vert_pick = int(vidxs[np.argmin(np.abs(vidxs - seed_x))])

            # Decide. Prefer vertical when the running pick has drifted away
            # from seed_x (or doesn't exist) AND a vertical pick is available.
            chosen: Optional[int] = None
            if vert_pick is not None and (
                run_pick is None or abs(run_pick - seed_x) > snap_thresh
            ):
                chosen = vert_pick
            elif run_pick is not None:
                chosen = run_pick

            if chosen is not None:
                xs[y] = chosen
                x = chosen
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


# ---------------------------------------------------------------------------
# Multi-pass refinement
# ---------------------------------------------------------------------------
def _find_drift_segments(xs: np.ndarray, y0: int, y1: int,
                         drift_threshold: int, min_len: int
                         ) -> List[Tuple[int, int]]:
    """Identify contiguous row ranges where xs deviates more than
    `drift_threshold` pixels from the global median over [y0, y1]. Returns
    list of (start, end) inclusive row indices."""
    seg = xs[y0:y1 + 1]
    valid = seg[seg >= 0]
    if len(valid) == 0:
        return []
    rolling_med = int(np.median(valid))
    bad = (seg >= 0) & (np.abs(seg - rolling_med) > drift_threshold)
    runs: List[Tuple[int, int]] = []
    cur: Optional[int] = None
    n = len(seg)
    for i in range(n):
        if bad[i] and cur is None:
            cur = i
        elif not bad[i] and cur is not None:
            if i - cur >= min_len:
                runs.append((y0 + cur, y0 + i - 1))
            cur = None
    if cur is not None and n - cur >= min_len:
        runs.append((y0 + cur, y0 + n - 1))
    return runs


def refine_multipass(img_rgb: np.ndarray, xs: np.ndarray,
                     y0: int, y1: int, cfg: Dict
                     ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Iteratively refine the trace to favor straight(-er) lines.

    For each pass:
      1. Find segments where xs deviates from the rolling median by more
         than `multipass_drift_threshold` (these are 'suspicious' segments
         where the trace likely strayed onto a letter stroke).
      2. For each suspicious segment, take medians of the clean context
         immediately before and after it. Linear-interpolate a 'predicted x'
         at every row in the segment.
      3. Snap each row's x to the nearest dark pixel within
         `multipass_line_tolerance` of the predicted x. If no dark pixel is
         within tolerance, mark the row as a gap; the final interpolation
         step will fill it from neighboring known rows.

    Returns refined (xs, (y0, y1), n_swaps).
    """
    if y1 < y0:
        return xs, (y0, y1), 0
    chan = to_single_channel(img_rgb, cfg.get('gray_mode', 'gray'))
    dark = chan <= cfg['dark_threshold']
    H, W = dark.shape
    drift_threshold = cfg.get('multipass_drift_threshold', 3)
    min_seg_len = cfg.get('multipass_min_seg_len', 8)
    line_tol = cfg.get('multipass_line_tolerance', 1)
    context_window = cfg.get('multipass_context_window', 50)
    max_passes = cfg.get('multipass_passes', 4)

    n_swaps_total = 0
    for _ in range(max_passes):
        suspicious = _find_drift_segments(xs, y0, y1, drift_threshold, min_seg_len)
        if not suspicious:
            break
        n_swaps_pass = 0
        for sy0, sy1 in suspicious:
            # Anchors: medians of clean context before and after the segment
            pre_seg = xs[max(y0, sy0 - context_window):sy0]
            pre_v = pre_seg[pre_seg >= 0]
            post_seg = xs[sy1 + 1:min(y1 + 1, sy1 + 1 + context_window)]
            post_v = post_seg[post_seg >= 0]
            if len(pre_v) < 5 or len(post_v) < 5:
                continue  # not enough clean context to predict
            x_pre = float(np.median(pre_v))
            x_post = float(np.median(post_v))
            n_rows = sy1 - sy0 + 1
            ys_idx = np.arange(n_rows, dtype=np.float64)
            predicted = x_pre + (x_post - x_pre) * ys_idx / max(1, n_rows - 1)

            new_seg = xs[sy0:sy1 + 1].copy()
            row_changed = False
            for i in range(n_rows):
                y = sy0 + i
                px = int(round(predicted[i]))
                lo = max(0, px - line_tol)
                hi = min(W, px + line_tol + 1)
                row = dark[y, lo:hi]
                if row.any():
                    idxs = np.where(row)[0] + lo
                    new_x = int(idxs[np.argmin(np.abs(idxs - px))])
                    if new_x != new_seg[i]:
                        new_seg[i] = new_x
                        row_changed = True
                else:
                    if new_seg[i] != -1:
                        new_seg[i] = -1
                        row_changed = True
            if row_changed:
                xs[sy0:sy1 + 1] = new_seg
                n_swaps_pass += 1
        n_swaps_total += n_swaps_pass
        if n_swaps_pass == 0:
            break

    # Re-fill any -1 rows in [y0, y1] by linear interpolation
    ys_known = [y for y in range(y0, y1 + 1) if xs[y] >= 0]
    if ys_known:
        xs_known = [xs[y] for y in ys_known]
        for y in range(y0, y1 + 1):
            if xs[y] < 0:
                xs[y] = int(np.interp(y, ys_known, xs_known))
    return xs, (y0, y1), n_swaps_total


def trace_from_seed_refined(img_rgb: np.ndarray, seed_x: int, cfg: Dict
                            ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Convenience: trace_from_seed + refine_multipass."""
    xs, (y0, y1) = trace_from_seed(img_rgb, seed_x, cfg)
    if cfg.get('multipass_passes', 4) > 0:
        xs, (y0, y1), _ = refine_multipass(img_rgb, xs, y0, y1, cfg)
    return xs, (y0, y1)


def find_vertical_line(img_rgb: np.ndarray, cfg: Optional[Dict] = None,
                       seed_x_override: Optional[int] = None
                       ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Top-level convenience. Auto-seeds unless seed_x_override is given.
    Includes multi-pass refinement when `cfg['multipass_passes'] > 0`."""
    if cfg is None:
        cfg = DEFAULT_CONFIG
    seed_x = int(seed_x_override) if seed_x_override is not None else find_seed_x(img_rgb, cfg)
    xs, (y0, y1) = trace_from_seed_refined(img_rgb, seed_x, cfg)
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

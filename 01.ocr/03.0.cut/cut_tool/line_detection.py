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
    # Isolation-skeleton + polyfit (the primary line-finder for printed
    # dividers). Build a binary mask where each pixel survives only if its
    # column has many vertical neighbours within `iso_vk_rows` AND the
    # flanking columns at +/- `iso_flank_offset` px each have at most
    # `iso_flank_max` such neighbours -- i.e., the divider is dark surrounded
    # by light gutter, while text columns (which have many dark neighbours
    # left and right) get rejected. The strict params produce few but very
    # confident anchor pixels; if they're too sparse, we fall back to medium
    # params, then to the dual-window trace.
    'iso_vk_rows': 20,             # half-height of the column-density window
    'iso_inner_min': 35,           # min dark count in column window for survival
    'iso_flank_offset': 6,         # px to flanking columns
    'iso_flank_max': 6,            # max dark count allowed in either flank
    'iso_strict_min': 200,         # need this many anchors before trying medium
    'iso_min_anchors': 100,        # below this -> total fallback to dual-window
    'iso_inlier_band': 3,          # px band around modal x for initial inliers
    'iso_residual_threshold': 2.5, # max polyfit residual (px) to keep an inlier
    'iso_extrap_threshold': 1.5,   # mean residual <= this -> extrapolate to full page
    'iso_min_span_frac': 0.30,     # inliers must span at least this fraction of page
    # Threshold sweep: try these dark_thresholds for mask construction (the
    # one giving the highest mode count wins). Lets us handle pages whose
    # dividers are darker than the global default.
    'iso_threshold_sweep': (110, 125, 140),
    # Medium-mask fallback parameters (looser than strict).
    'iso_medium_vk_rows': 10,
    'iso_medium_inner_min': 15,
    'iso_medium_flank_max': 8,
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
    """Single-call trace API used by the GUI for manual re-trace.

    Tries iso-polyfit first (which is the primary line finder). Iso-polyfit
    has its own internal fallback to dual-window+multipass when anchors are
    too sparse, so this function is end-to-end robust."""
    xs, (y0, y1), _src, _info = trace_iso_polyfit(img_rgb, seed_x, cfg)
    return xs, (y0, y1)


# ---------------------------------------------------------------------------
# Isolation-skeleton mask + polynomial line fit
# ---------------------------------------------------------------------------
def build_isolation_mask(dark: np.ndarray, vk_rows: int, inner_min: int,
                          flank_offset: int, flank_max: int) -> np.ndarray:
    """A pixel survives if its column has at least `inner_min` dark pixels in
    the window y +/- vk_rows AND the flanking columns at +/- flank_offset px
    each have at most `flank_max` such dark pixels.

    The first condition keeps tall vertical structures (the divider, but
    also tall letter strokes). The flank conditions reject tall structures
    that have other tall dark structures immediately adjacent (text columns,
    where letter strokes cluster horizontally).

    Net effect: keeps ISOLATED tall vertical structures, which is exactly
    what a printed/inked divider looks like in a two-column scan.
    """
    H, W = dark.shape
    cs = np.cumsum(dark.astype(np.int32), axis=0)
    vert_count = np.zeros_like(dark, dtype=np.int32)
    for y in range(H):
        y0_ = max(0, y - vk_rows)
        y1_ = min(H - 1, y + vk_rows)
        vert_count[y] = cs[y1_] - (cs[y0_ - 1] if y0_ > 0 else 0)

    flank_left = np.zeros_like(vert_count)
    flank_right = np.zeros_like(vert_count)
    if flank_offset < W:
        flank_left[:, flank_offset:] = vert_count[:, :W - flank_offset]
        flank_right[:, :W - flank_offset] = vert_count[:, flank_offset:]

    inner_ok = vert_count >= inner_min
    flank_ok = (flank_left <= flank_max) & (flank_right <= flank_max)
    return dark & inner_ok & flank_ok


def _gather_iso_anchors(chan: np.ndarray, seed_x: int, max_drift: int,
                         vk: int, inner: int, off: int, flmax: int,
                         thresholds: Tuple[int, ...]
                         ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[int]]:
    """Try each threshold in `thresholds`; for each build the isolation mask
    and collect anchors (the dark pixel closest to seed_x within +/- max_drift
    on each row). Return the anchor set whose modal x had the highest count
    (strongest single-x signal). Returns (ys, xs, threshold) or (None, None, None)
    if no threshold produced any anchors.
    """
    H, W = chan.shape
    lo, hi = max(0, seed_x - max_drift), min(W, seed_x + max_drift + 1)
    best_y: Optional[np.ndarray] = None
    best_x: Optional[np.ndarray] = None
    best_mode_count = -1
    best_thr: Optional[int] = None
    for thr in thresholds:
        dark = chan <= thr
        mask = build_isolation_mask(dark, vk, inner, off, flmax)
        anchors_y: List[int] = []
        anchors_x: List[int] = []
        for y in range(H):
            row = mask[y, lo:hi]
            if row.any():
                xs_b = np.where(row)[0] + lo
                anchors_y.append(y)
                anchors_x.append(int(xs_b[np.argmin(np.abs(xs_b - seed_x))]))
        if not anchors_y:
            continue
        ax = np.array(anchors_x, dtype=np.int32)
        _, counts = np.unique(ax, return_counts=True)
        mode_count = int(counts.max())
        if mode_count > best_mode_count:
            best_mode_count = mode_count
            best_y = np.array(anchors_y, dtype=np.int32)
            best_x = ax
            best_thr = thr
    return best_y, best_x, best_thr


def _trace_dual_then_refine(img_rgb: np.ndarray, seed_x: int, cfg: Dict
                             ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Internal: dual-window trace + multipass refinement, with no iso path.
    Used as a hard fallback inside `trace_iso_polyfit` to avoid infinite
    recursion via `trace_from_seed_refined`."""
    xs, (y0, y1) = trace_from_seed(img_rgb, seed_x, cfg)
    if cfg.get('multipass_passes', 4) > 0 and y1 >= y0:
        xs, (y0, y1), _ = refine_multipass(img_rgb, xs, y0, y1, cfg)
    return xs, (y0, y1)


def trace_iso_polyfit(img_rgb: np.ndarray, seed_x: int, cfg: Dict
                       ) -> Tuple[np.ndarray, Tuple[int, int], str, str]:
    """Strict-mask -> medium-mask cascade -> RANSAC-style polyfit (deg 1 or 2)
    -> extrapolate (or bound to anchor span if fit is loose).

    Returns (xs, (y0, y1), source_label, info_str). source_label is one of:
      'iso-polyfit-strict', 'iso-polyfit-medium', 'iso-fallback'.
    """
    chan = to_single_channel(img_rgb, cfg.get('gray_mode', 'gray'))
    H, W = chan.shape
    max_drift = cfg.get('max_drift_from_seed', 15)
    thresholds = cfg.get('iso_threshold_sweep', (110, 125, 140))

    # Strict mask
    y_s, x_s, thr_s = _gather_iso_anchors(
        chan, seed_x, max_drift,
        cfg.get('iso_vk_rows', 20),
        cfg.get('iso_inner_min', 35),
        cfg.get('iso_flank_offset', 6),
        cfg.get('iso_flank_max', 6),
        thresholds,
    )
    if y_s is not None and len(y_s) >= cfg.get('iso_strict_min', 200):
        ys, xs_a = y_s.astype(np.float64), x_s.astype(np.float64)
        chosen_thr = thr_s
        mask_label = 'strict'
    else:
        # Medium fallback
        y_m, x_m, thr_m = _gather_iso_anchors(
            chan, seed_x, max_drift,
            cfg.get('iso_medium_vk_rows', 10),
            cfg.get('iso_medium_inner_min', 15),
            cfg.get('iso_flank_offset', 6),
            cfg.get('iso_medium_flank_max', 8),
            thresholds,
        )
        if y_m is None or len(y_m) < cfg.get('iso_min_anchors', 100):
            xs_, (y0_, y1_) = _trace_dual_then_refine(img_rgb, seed_x, cfg)
            return xs_, (y0_, y1_), 'iso-fallback', 'too few anchors'
        ys, xs_a = y_m.astype(np.float64), x_m.astype(np.float64)
        chosen_thr = thr_m
        mask_label = 'medium'

    # Inliers within band of mode
    unique, counts = np.unique(xs_a.astype(np.int32), return_counts=True)
    dom_x = int(unique[np.argmax(counts)])
    inliers = np.abs(xs_a - dom_x) <= cfg.get('iso_inlier_band', 3)
    if inliers.sum() < 30:
        inliers = np.abs(xs_a - dom_x) <= 6
    iy = ys[inliers]
    ix = xs_a[inliers]

    # Span check
    if len(iy) == 0 or (iy[-1] - iy[0] + 1) / H < cfg.get('iso_min_span_frac', 0.30):
        xs_, (y0_, y1_) = _trace_dual_then_refine(img_rgb, seed_x, cfg)
        return xs_, (y0_, y1_), 'iso-fallback', 'inlier span too small'

    # Polyfit deg 1 vs 2 with iterative outlier rejection
    res_thresh = cfg.get('iso_residual_threshold', 2.5)
    best: Optional[Tuple[float, np.ndarray, int, float, int]] = None
    for deg in (1, 2):
        cur_y, cur_x = iy.copy(), ix.copy()
        for _ in range(3):
            if len(cur_y) < deg + 1:
                break
            cf = np.polyfit(cur_y, cur_x, deg)
            res = np.abs(cur_x - np.polyval(cf, cur_y))
            keep = res <= res_thresh
            if keep.sum() < deg + 1 or keep.sum() == len(cur_y):
                break
            cur_y, cur_x = cur_y[keep], cur_x[keep]
        if len(cur_y) < deg + 1:
            continue
        cf = np.polyfit(cur_y, cur_x, deg)
        mres = float(np.mean(np.abs(cur_x - np.polyval(cf, cur_y))))
        # Penalise degree 2 slightly so we prefer simple straight fits when
        # the residual is comparable.
        score = mres + (0.10 if deg == 2 else 0.0)
        if best is None or score < best[0]:
            best = (score, cf, deg, mres, len(cur_y))
    if best is None:
        xs_, (y0_, y1_) = _trace_dual_then_refine(img_rgb, seed_x, cfg)
        return xs_, (y0_, y1_), 'iso-fallback', 'polyfit failed'

    _, coeffs, deg, residual, n_inliers = best

    # Extrapolate to full page if fit is tight, else bound to anchor span
    full_xs = np.full(H, -1, dtype=np.int32)
    extrap_thresh = cfg.get('iso_extrap_threshold', 1.5)
    if residual < extrap_thresh:
        y0, y1 = 0, H - 1
    else:
        y0 = max(0, int(iy.min()) - 50)
        y1 = min(H - 1, int(iy.max()) + 50)
    for y in range(y0, y1 + 1):
        x_pred = int(round(np.polyval(coeffs, y)))
        if abs(x_pred - seed_x) > max_drift:
            continue
        full_xs[y] = x_pred

    info = f'{mask_label} deg={deg} resid={residual:.2f} n={n_inliers} thr={chosen_thr}'
    return full_xs, (y0, y1), f'iso-polyfit-{mask_label}', info


def find_vertical_line(img_rgb: np.ndarray, cfg: Optional[Dict] = None,
                       seed_x_override: Optional[int] = None
                       ) -> Tuple[np.ndarray, Tuple[int, int], int]:
    """Top-level convenience. Auto-seeds unless `seed_x_override` is given.

    Uses isolation-skeleton + polynomial fit by default. Falls back to the
    dual-window trace + multipass refinement when the iso approach can't
    establish enough anchors (e.g., pages with no printed divider, just a
    column gap).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    seed_x = (int(seed_x_override) if seed_x_override is not None
              else find_seed_x(img_rgb, cfg))
    xs, (y0, y1), _src, _info = trace_iso_polyfit(img_rgb, seed_x, cfg)
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

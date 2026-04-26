# import_svg_overlays.py — COMPLETE FIXED VERSION
"""Import edited SVG overlays and update overlays.json with corrected lines.

Usage:
    python import_svg_overlays.py <svg_folder> [source_image_folder]

Reads edited SVG files, extracts the corrected split line paths,
and updates overlays.json in the source image folder. Then you can
re-run the notebook to apply the corrections.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

from line_detection import (
    load_json_safely,
    save_json,
    xs_to_serializable,
)

OVERLAYS_FILENAME = 'overlays.json'


# ---------------------------------------------------------------------------
# Path drawing helpers
# ---------------------------------------------------------------------------
def _record_point(xs: np.ndarray, x: float, y: float) -> None:
    """Record a single point in the xs array."""
    row = int(round(y))
    if 0 <= row < len(xs):
        xs[row] = int(round(x))


def _rasterize_line(xs: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> None:
    """Rasterize a line segment, filling every row between y1 and y2."""
    dy = abs(y2 - y1)
    if dy < 0.5:
        _record_point(xs, x1, y1)
        _record_point(xs, x2, y2)
        return
    steps = max(int(dy), 1)
    for t in range(steps + 1):
        frac = t / steps
        x = x1 + (x2 - x1) * frac
        y = y1 + (y2 - y1) * frac
        _record_point(xs, x, y)


def _rasterize_bezier(
    xs: np.ndarray,
    x0: float, y0: float,
    cp1x: float, cp1y: float,
    cp2x: float, cp2y: float,
    x3: float, y3: float,
) -> None:
    """Rasterize a cubic bezier curve, filling every row."""
    dy = abs(y3 - y0)
    if dy < 0.5:
        _record_point(xs, x0, y0)
        _record_point(xs, x3, y3)
        return
    
    samples = max(int(dy * 3), 30)
    prev_y = int(round(y0))
    prev_x = x0
    _record_point(xs, x0, y0)
    
    for t in np.linspace(0, 1, samples):
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        x = mt3 * x0 + 3 * mt2 * t * cp1x + 3 * mt * t2 * cp2x + t3 * x3
        y = mt3 * y0 + 3 * mt2 * t * cp1y + 3 * mt * t2 * cp2y + t3 * y3
        
        curr_y = int(round(y))
        # Fill any skipped rows
        if abs(curr_y - prev_y) > 1:
            step = 1 if curr_y > prev_y else -1
            for interp_y in range(prev_y + step, curr_y, step):
                frac = (interp_y - prev_y) / (curr_y - prev_y) if curr_y != prev_y else 0
                interp_x = prev_x + (x - prev_x) * frac
                _record_point(xs, interp_x, interp_y)
        
        _record_point(xs, x, y)
        prev_y = curr_y
        prev_x = x


# ---------------------------------------------------------------------------
# SVG path 'd' string parser (handles M, m, L, l, C, c, Z, z)
# ---------------------------------------------------------------------------
def parse_svg_path_d(d_string: str, image_height: int) -> np.ndarray:
    """Parse SVG path 'd' attribute — handles all common commands.
    
    Supports: M, m, L, l, H, h, V, v, C, c, Z, z
    Also handles implicit lineto after moveto (SVG spec).
    """
    xs = np.full(image_height, -1, dtype=np.int32)
    d_string = d_string.strip()
    
    # Tokenize: command letters (all standard ones) and numbers
    tokens = re.findall(
        r'[MLHVCSQTAZmlhvcsqtaz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d_string
    )
    
    cur_x, cur_y = 0.0, 0.0
    start_x, start_y = 0.0, 0.0
    last_cmd = None
    i = 0
    
    while i < len(tokens):
        tok = tokens[i]
        
        # Check if token is a command letter
        if tok in 'MLHVCSQTAZmlhvcsqtaz':
            cmd = tok
            i += 1
            last_cmd = cmd
        else:
            # Implicit command repetition
            if last_cmd in ('M', 'm'):
                cmd = 'L' if last_cmd == 'M' else 'l'
            elif last_cmd in ('L', 'l', 'H', 'h', 'V', 'v', 'C', 'c'):
                cmd = last_cmd
            else:
                # Unknown context: skip this number and try to recover
                i += 1
                continue
            # Don't increment i; we'll read the numbers with the current cmd
        
        # --- Moveto ---
        if cmd == 'M':
            cur_x = float(tokens[i]); cur_y = float(tokens[i+1]); i += 2
            start_x, start_y = cur_x, cur_y
            _record_point(xs, cur_x, cur_y)
        elif cmd == 'm':
            cur_x += float(tokens[i]); cur_y += float(tokens[i+1]); i += 2
            start_x, start_y = cur_x, cur_y
            _record_point(xs, cur_x, cur_y)
        
        # --- Lineto (absolute) ---
        elif cmd == 'L':
            while i < len(tokens) and not is_cmd(tokens[i]):
                x = float(tokens[i]); y = float(tokens[i+1]); i += 2
                _rasterize_line(xs, cur_x, cur_y, x, y)
                cur_x, cur_y = x, y
        elif cmd == 'l':
            while i < len(tokens) and not is_cmd(tokens[i]):
                dx = float(tokens[i]); dy = float(tokens[i+1]); i += 2
                x = cur_x + dx; y = cur_y + dy
                _rasterize_line(xs, cur_x, cur_y, x, y)
                cur_x, cur_y = x, y
        
        # --- Horizontal lineto ---
        elif cmd == 'H':
            while i < len(tokens) and not is_cmd(tokens[i]):
                x = float(tokens[i]); i += 1
                _rasterize_line(xs, cur_x, cur_y, x, cur_y)
                cur_x = x
        elif cmd == 'h':
            while i < len(tokens) and not is_cmd(tokens[i]):
                cur_x += float(tokens[i]); i += 1
                _rasterize_line(xs, cur_x - float(tokens[i-1]), cur_y, cur_x, cur_y)
                # Actually easier: record line from before to after
                # We'll adjust: simplified:
                x_before = cur_x - float(tokens[i-1])
                _rasterize_line(xs, x_before, cur_y, cur_x, cur_y)
        
        # --- Vertical lineto ---
        elif cmd == 'V':
            while i < len(tokens) and not is_cmd(tokens[i]):
                y = float(tokens[i]); i += 1
                _rasterize_line(xs, cur_x, cur_y, cur_x, y)
                cur_y = y
        elif cmd == 'v':
            while i < len(tokens) and not is_cmd(tokens[i]):
                cur_y += float(tokens[i]); i += 1
                y_before = cur_y - float(tokens[i-1])
                _rasterize_line(xs, cur_x, y_before, cur_x, cur_y)
        
        # --- Cubic bezier (absolute) ---
        elif cmd == 'C':
            while i + 5 < len(tokens) and not is_cmd(tokens[i]):
                cp1x = float(tokens[i]);   cp1y = float(tokens[i+1])
                cp2x = float(tokens[i+2]); cp2y = float(tokens[i+3])
                x    = float(tokens[i+4]); y    = float(tokens[i+5])
                i += 6
                _rasterize_bezier(xs, cur_x, cur_y, cp1x, cp1y, cp2x, cp2y, x, y)
                cur_x, cur_y = x, y
        elif cmd == 'c':
            while i + 5 < len(tokens) and not is_cmd(tokens[i]):
                cp1x = cur_x + float(tokens[i]);   cp1y = cur_y + float(tokens[i+1])
                cp2x = cur_x + float(tokens[i+2]); cp2y = cur_y + float(tokens[i+3])
                x    = cur_x + float(tokens[i+4]); y    = cur_y + float(tokens[i+5])
                i += 6
                _rasterize_bezier(xs, cur_x, cur_y, cp1x, cp1y, cp2x, cp2y, x, y)
                cur_x, cur_y = x, y
        
        # --- Close path ---
        elif cmd in ('Z', 'z'):
            _rasterize_line(xs, cur_x, cur_y, start_x, start_y)
            cur_x, cur_y = start_x, start_y
    
    return xs


def is_cmd(token: str) -> bool:
    """Return True if token is an SVG command letter."""
    return token in 'MLHVCSQTAZmlhvcsqtaz'

# ---------------------------------------------------------------------------
# SVG content parsers — extract line from full SVG file
# ---------------------------------------------------------------------------
def parse_svg_polyline(svg_content: str, image_height: int) -> np.ndarray | None:
    """Parse <polyline id="...split..." points="x,y x,y ..."> from SVG content."""
    match = re.search(
        r'<polyline[^>]*\bid\s*=\s*["\'][^"\']*split[^"\']*["\'][^>]*\bpoints\s*=\s*["\']([^"\']+)["\']',
        svg_content, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    
    xs = np.full(image_height, -1, dtype=np.int32)
    points_str = match.group(1)
    
    for pair in points_str.strip().split():
        parts = pair.split(',')
        if len(parts) == 2:
            try:
                x = int(round(float(parts[0])))
                y = int(round(float(parts[1])))
                if 0 <= y < image_height:
                    xs[y] = x
            except ValueError:
                pass
    
    valid_count = int((xs >= 0).sum())
    return xs if valid_count > 10 else None


def parse_svg_split_path(svg_content: str, image_height: int) -> np.ndarray | None:
    """Parse <path id="...split..." d="..."> from SVG content."""
    match = re.search(
        r'<path[^>]*\bid\s*=\s*["\'][^"\']*split[^"\']*["\'][^>]*\bd\s*=\s*["\']([^"\']+)["\']',
        svg_content, re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    
    d_string = match.group(1)
    if not d_string.strip():
        return None
    
    return parse_svg_path_d(d_string, image_height)


def parse_svg_any_line(svg_content: str, image_height: int) -> np.ndarray | None:
    """Try to extract split line from SVG: polyline first, then path."""
    result = parse_svg_polyline(svg_content, image_height)
    if result is not None:
        return result
    return parse_svg_split_path(svg_content, image_height)


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------
def import_overlays(
    svg_folder: Path,
    source_image_folder: Optional[Path] = None,
) -> None:
    """Import edited SVG overlays and update overlays.json."""
    
    svg_folder = Path(svg_folder)
    if not svg_folder.is_dir():
        print(f'Not a directory: {svg_folder}')
        return
    
    # Find SVG files (prefer *_overlay.svg, fallback to *.svg)
    svg_files = sorted(svg_folder.glob('*_overlay.svg'))
    if not svg_files:
        svg_files = sorted(svg_folder.glob('*.svg'))
    
    if not svg_files:
        print(f'No SVG files found in {svg_folder}')
        return
    
    # Determine source image folder
    if source_image_folder is None:
        source_image_folder = svg_folder.parent
    source_image_folder = Path(source_image_folder)
    
    overlays_path = source_image_folder / OVERLAYS_FILENAME
    
    # Load existing overlays
    overlays_data = load_json_safely(overlays_path)
    existing_entries = overlays_data.get('entries', {})
    
    updated_entries: Dict = {}
    stats = {'success': 0, 'skipped': 0, 'errors': 0}
    
    print(f'Importing {len(svg_files)} SVG file(s)...')
    
    for svg_path in svg_files:
        # Get image stem: "gold_00014_overlay.svg" → "gold_00014"
        stem = svg_path.stem.replace('_overlay', '')
        
        # Find matching original image file
        orig_filename = None
        orig_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
            for e in (ext, ext.upper()):
                candidate = source_image_folder / f'{stem}{e}'
                if candidate.exists():
                    orig_filename = candidate.name
                    orig_path = candidate
                    break
            if orig_filename:
                break
        
        if orig_filename is None:
            print(f'  SKIP {svg_path.name}: no matching image found')
            stats['skipped'] += 1
            continue
        
        # Get image height
        try:
            img = Image.open(orig_path)
            img_h = img.height
        except Exception as e:
            print(f'  ERROR {svg_path.name}: cannot open {orig_path}: {e}')
            stats['errors'] += 1
            continue
        
        # Read SVG content
        try:
            svg_content = svg_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f'  ERROR {svg_path.name}: cannot read file: {e}')
            stats['errors'] += 1
            continue
        
        # Parse the line
        xs = parse_svg_any_line(svg_content, img_h)
        
        if xs is None:
            print(f'  SKIP {svg_path.name}: no parseable split line found')
            stats['skipped'] += 1
            continue
        
        # Count valid points
        valid_count = int((xs >= 0).sum())
        if valid_count < 10:
            print(f'  SKIP {svg_path.name}: only {valid_count} valid points')
            stats['skipped'] += 1
            continue
        
        # Determine y range from actual valid rows
        valid_rows = np.where(xs >= 0)[0]
        y0 = int(valid_rows[0])
        y1 = int(valid_rows[-1])
        
        # Store entry
        updated_entries[orig_filename] = {
            'y0': y0,
            'y1': y1,
            'xs': xs_to_serializable(xs),
            'autogen': False,  # Mark as manually edited
        }
        
        print(f'  OK  {orig_filename}: {valid_count} points, rows {y0}-{y1}')
        stats['success'] += 1
    
    # Save results
    if updated_entries:
        config_used = overlays_data.get('config_used') or {}
        
        save_json(
            overlays_path,
            {
                'config_used': config_used,
                'entries': {**existing_entries, **updated_entries},
            },
        )
        print(f'\nSaved to: {overlays_path}')
        print(f'Results: {stats["success"]} imported, {stats["skipped"]} skipped, {stats["errors"]} errors')
        print(f'Re-run the notebook to apply your corrections!')
    else:
        print('\nNo entries were updated.')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python import_svg_overlays.py <svg_folder> [source_image_folder]')
        print('Example: python import_svg_overlays.py BBL/gld/svg_overlays BBL/gld')
        sys.exit(1)
    
    svg_dir = Path(sys.argv[1])
    src_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    import_overlays(svg_dir, src_dir)
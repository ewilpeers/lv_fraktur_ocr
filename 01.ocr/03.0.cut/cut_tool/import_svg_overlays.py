# import_svg_overlays.py
"""Import edited SVG overlays and update overlays.json with corrected lines.

Usage:
    python import_svg_overlays.py <svg_folder> [source_image_folder]

Reads edited SVG files, extracts the corrected split line paths,
and updates overlays.json in the source image folder. Then you can
re-run the notebook to apply the corrections.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from line_detection import (
    load_json_safely,
    save_json,
    xs_to_serializable,
)

from correction_tool import (
    OVERLAYS_FILENAME
)

# XML namespace handling
SVG_NS = 'http://www.w3.org/2000/svg'


def parse_svg_path_d(d_string: str, image_height: int) -> np.ndarray:
    """Parse an SVG path 'd' attribute and rasterize it to pixel columns per row.
    
    Handles M (moveto), L (lineto), C (cubic bezier) commands.
    Returns xs array where xs[y] = x coordinate or -1 if no line at that row.
    
    For bezier curves, samples at sub-pixel resolution to ensure smooth
    interpolation.
    """
    xs = np.full(image_height, -1, dtype=np.int32)
    
    # Normalize the path string: remove extra whitespace, split commands
    d_string = d_string.strip()
    
    # Tokenize: split on command letters, keeping the letters
    tokens = re.findall(r'[MLCZmlcz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d_string)
    
    current_x = 0.0
    current_y = 0.0
    start_x = 0.0
    start_y = 0.0
    
    i = 0
    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        
        if cmd == 'M':
            # Absolute moveto
            x = float(tokens[i]); y = float(tokens[i + 1])
            i += 2
            current_x, current_y = x, y
            start_x, start_y = x, y
            _record_point(xs, x, y)
            
        elif cmd == 'm':
            # Relative moveto
            x = float(tokens[i]); y = float(tokens[i + 1])
            i += 2
            current_x += x; current_y += y
            start_x, start_y = current_x, current_y
            _record_point(xs, current_x, current_y)
            
        elif cmd == 'L':
            # Absolute lineto
            while i < len(tokens) and not tokens[i].isalpha():
                x = float(tokens[i]); y = float(tokens[i + 1])
                i += 2
                _rasterize_line(xs, current_x, current_y, x, y)
                current_x, current_y = x, y
                
        elif cmd == 'l':
            # Relative lineto
            while i < len(tokens) and not tokens[i].isalpha():
                dx = float(tokens[i]); dy = float(tokens[i + 1])
                i += 2
                x = current_x + dx; y = current_y + dy
                _rasterize_line(xs, current_x, current_y, x, y)
                current_x, current_y = x, y
                
        elif cmd == 'C':
            # Absolute cubic bezier
            while i < len(tokens) and not tokens[i].isalpha():
                cp1x = float(tokens[i]); cp1y = float(tokens[i + 1])
                cp2x = float(tokens[i + 2]); cp2y = float(tokens[i + 3])
                x = float(tokens[i + 4]); y = float(tokens[i + 5])
                i += 6
                _rasterize_bezier(xs, current_x, current_y, cp1x, cp1y, cp2x, cp2y, x, y)
                current_x, current_y = x, y
                
        elif cmd == 'c':
            # Relative cubic bezier
            while i < len(tokens) and not tokens[i].isalpha():
                cp1x = current_x + float(tokens[i])
                cp1y = current_y + float(tokens[i + 1])
                cp2x = current_x + float(tokens[i + 2])
                cp2y = current_y + float(tokens[i + 3])
                x = current_x + float(tokens[i + 4])
                y = current_y + float(tokens[i + 5])
                i += 6
                _rasterize_bezier(xs, current_x, current_y, cp1x, cp1y, cp2x, cp2y, x, y)
                current_x, current_y = x, y
                
        elif cmd in ('Z', 'z'):
            # Close path: line back to start
            _rasterize_line(xs, current_x, current_y, start_x, start_y)
            current_x, current_y = start_x, start_y
    
    return xs


def _record_point(xs: np.ndarray, x: float, y: float) -> None:
    """Record a single point in the xs array."""
    row = int(round(y))
    if 0 <= row < len(xs):
        xs[row] = int(round(x))


def _rasterize_line(xs: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> None:
    """Rasterize a line segment into the xs array."""
    dy = abs(y2 - y1)
    if dy < 0.5:
        # Nearly horizontal: just record endpoints
        _record_point(xs, x1, y1)
        _record_point(xs, x2, y2)
        return
    
    # Sample at every row
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
    """Rasterize a cubic bezier curve into the xs array.
    
    Samples densely enough to capture every row the curve passes through.
    """
    # Estimate arc length for sampling
    dy = abs(y3 - y0)
    if dy < 0.5:
        _record_point(xs, x0, y0)
        _record_point(xs, x3, y3)
        return
    
    # Adaptive sampling: more points for longer curves
    samples = max(int(dy * 3), 30)
    
    prev_y = int(round(y0))
    _record_point(xs, x0, y0)
    
    for t in np.linspace(0, 1, samples):
        # Cubic bezier formula
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        
        x = mt3 * x0 + 3 * mt2 * t * cp1x + 3 * mt * t2 * cp2x + t3 * x3
        y = mt3 * y0 + 3 * mt2 * t * cp1y + 3 * mt * t2 * cp2y + t3 * y3
        
        curr_y = int(round(y))
        # If we skipped rows, interpolate them
        if abs(curr_y - prev_y) > 1:
            step = 1 if curr_y > prev_y else -1
            for interp_y in range(prev_y + step, curr_y, step):
                frac = (interp_y - prev_y) / (curr_y - prev_y) if curr_y != prev_y else 0
                interp_x = x0 + (x - x0) * frac  # Approximate
                _record_point(xs, interp_x, interp_y)
        
        _record_point(xs, x, y)
        prev_y = curr_y
        x0, y0 = x, y  # Update for interpolation tracking


def extract_path_from_svg(svg_path: Path) -> Optional[Tuple[str, float, float]]:
    """Extract the split line path from an edited SVG.
    
    Returns (path_d_string, y_start, y_end) or None if no valid path found.
    Looks for the path with id='split-line'.
    """
    try:
        content = svg_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f'  ERROR reading {svg_path.name}: {e}')
        return None
    
    # Try to find path with id="split-line"
    # Pattern: <path ... id="split-line" ... d="..." ... />
    pattern = r'<path[^>]*\bid\s*=\s*["\']split-line["\'][^>]*\bd\s*=\s*["\']([^"\']+)["\']'
    match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
    
    if not match:
        # Try finding any path with 'split' in the id
        pattern = r'<path[^>]*\bid\s*=\s*["\'][^"\']*split[^"\']*["\'][^>]*\bd\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
    
    if not match:
        # Last resort: first path element
        pattern = r'<path[^>]*\bd\s*=\s*["\']([^"\']+)["\']'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
    
    if not match:
        print(f'  WARNING: No path found in {svg_path.name}')
        return None
    
    path_d = match.group(1)
    
    # Extract coordinates to find y range
    coords = re.findall(r'[-+]?\d*\.?\d+', path_d)
    y_coords = []
    for i, val in enumerate(coords):
        # In M x y, L x y, C ... x y format, y is every 2nd value
        if i % 2 == 1 and i > 0:  # Skip the first x (for M command)
            y_coords.append(float(val))
    
    if not y_coords:
        return None
    
    y_start = min(y_coords)
    y_end = max(y_coords)
    
    return path_d, y_start, y_end


def import_overlays(
    svg_folder: Path,
    source_image_folder: Optional[Path] = None,
) -> None:
    """Import edited SVG overlays and update overlays.json."""
    
    svg_folder = Path(svg_folder)
    if not svg_folder.is_dir():
        print(f'Not a directory: {svg_folder}')
        return
    
    svg_files = sorted(svg_folder.glob('*_overlay.svg'))
    if not svg_files:
        svg_files = sorted(svg_folder.glob('*.svg'))
    
    if not svg_files:
        print(f'No SVG files found in {svg_folder}')
        return
    
    # Determine source image folder
    if source_image_folder is None:
        # Assume SVGs were exported from parent folder's overlays
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
        # Extract original image filename from SVG name
        # e.g., "gold_00013_overlay.svg" → "gold_00013.jpg"
        stem = svg_path.stem.replace('_overlay', '')
        
        # Find matching original image
        orig_filename = None
        orig_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
            candidate = source_image_folder / f'{stem}{ext}'
            if candidate.exists():
                orig_filename = candidate.name
                orig_path = candidate
                break
            # Try uppercase
            candidate = source_image_folder / f'{stem}{ext.upper()}'
            if candidate.exists():
                orig_filename = candidate.name
                orig_path = candidate
                break
        
        if orig_filename is None:
            print(f'  SKIP {svg_path.name}: no matching image found')
            stats['skipped'] += 1
            continue
        
        # Get image dimensions
        try:
            img = Image.open(orig_path)
            img_h = img.height
        except Exception as e:
            print(f'  ERROR {svg_path.name}: cannot open {orig_path}: {e}')
            stats['errors'] += 1
            continue
        
        # Extract path from SVG
        result = extract_path_from_svg(svg_path)
        if result is None:
            stats['skipped'] += 1
            continue
        
        path_d, y_start, y_end = result
        
        # Rasterize the path
        try:
            xs = parse_svg_path_d(path_d, img_h)
        except Exception as e:
            print(f'  ERROR {svg_path.name}: failed to parse path: {e}')
            stats['errors'] += 1
            continue
        
        # Count valid points
        valid_count = (xs >= 0).sum()
        if valid_count < 10:
            print(f'  WARNING {svg_path.name}: only {valid_count} valid points, skipping')
            stats['skipped'] += 1
            continue
        
        # Update entry
        y0 = max(0, int(y_start))
        y1 = min(img_h - 1, int(y_end))
        
        updated_entries[orig_filename] = {
            'y0': y0,
            'y1': y1,
            'xs': xs_to_serializable(xs),
            'autogen': False,  # Mark as manually edited
        }
        
        print(f'  ✓ {orig_filename}: {valid_count} valid points, rows {y0}-{y1}')
        stats['success'] += 1
    
    if updated_entries:
        # Merge with existing (keep untouched entries)
        config_used = overlays_data.get('config_used', {})
        if config_used is None:
            config_used = {}
        
        save_json(
            overlays_path,
            {
                'config_used': config_used,
                'entries': {**existing_entries, **updated_entries},
            }
        )
        print(f'\nUpdated {overlays_path}')
        print(f'  {stats["success"]} imported, {stats["skipped"]} skipped, {stats["errors"]} errors')
        print(f'\nNow re-run the notebook to apply the corrected lines!')
    else:
        print(f'\nNo entries updated.')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python import_svg_overlays.py <svg_folder> [source_image_folder]')
        sys.exit(1)
    
    svg_dir = Path(sys.argv[1])
    src_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    import_overlays(svg_dir, src_dir)
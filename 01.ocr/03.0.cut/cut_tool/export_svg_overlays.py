# export_svg_overlays.py
"""Export detected split lines as SVG overlays for editing in Inkscape.

Usage:
    python export_svg_overlays.py <source_folder> [output_folder]

For each image in source_folder that has entries in overlays.json,
creates an SVG with the image embedded and the split line as an editable
bezier path. Edit the path in Inkscape, save as plain SVG, then use
import_svg_overlays.py to apply the corrections.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from line_detection import (
    DEFAULT_CONFIG,
    load_json_safely,
    xs_from_serializable,
)

from correction_tool import (
    OVERLAYS_FILENAME,
    IMG_EXTS,
    STARTS_FILENAME,
)

# Inkscape-friendly namespace
SVG_NS = 'http://www.w3.org/2000/svg'
INKSCAPE_NS = 'http://www.inkscape.org/namespaces/inkscape'
SODIPODI_NS = 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd'

# Distinctive color for the split line in SVG
LINE_COLOR = '#FF0000'
LINE_WIDTH = 3.0
LINE_OPACITY = 0.8


def xs_to_svg_path(xs: np.ndarray, y0: int, y1: int) -> str:
    """Convert detected line array to an SVG path string.
    
    Generates a smooth bezier path that approximates the detected curve
    while being fully editable in Inkscape.
    """
    # Collect valid points within the detected range
    points: List[Tuple[int, int]] = []
    for y in range(y0, y1 + 1):
        x = int(xs[y])
        if x >= 0:
            points.append((x, y))
    
    if len(points) < 2:
        return ''
    
    # Simplify: use every Nth point for smoother curves, but keep endpoints
    # and points where direction changes significantly
    stride = max(1, len(points) // 50)  # Max ~50 nodes for editability
    simplified = [points[0]]
    last_angle = None
    
    for i in range(1, len(points) - 1, stride):
        prev = points[max(0, i - stride)]
        curr = points[i]
        next_pt = points[min(len(points) - 1, i + stride)]
        
        # Calculate direction change
        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = next_pt[0] - curr[0]
        dy2 = next_pt[1] - curr[1]
        
        angle1 = np.arctan2(dy1, dx1) if (dx1 or dy1) else 0
        angle2 = np.arctan2(dy2, dx2) if (dx2 or dy2) else 0
        angle_diff = abs(angle1 - angle2)
        
        # Keep point if direction changes significantly
        if angle_diff > 0.1 or i == 1:  # ~5.7 degrees
            simplified.append(curr)
            last_angle = angle2
    
    simplified.append(points[-1])
    
    # Build SVG path with cubic bezier segments for smooth curves
    if len(simplified) <= 2:
        # Simple polyline for straight or near-straight lines
        d = f'M {simplified[0][0]},{simplified[0][1]}'
        for pt in simplified[1:]:
            d += f' L {pt[0]},{pt[1]}'
        return d
    
    # Bezier interpolation for smoother curves
    d = f'M {simplified[0][0]},{simplified[0][1]}'
    
    for i in range(len(simplified) - 1):
        x1, y1 = simplified[i]
        x2, y2 = simplified[i + 1]
        
        # Calculate control points for smooth cubic bezier
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Control points slightly offset for natural curve
        cp1_x = x1 + (mid_x - x1) * 0.4
        cp1_y = y1 + (mid_y - y1) * 0.4
        cp2_x = x2 - (x2 - mid_x) * 0.4
        cp2_y = y2 - (y2 - mid_y) * 0.4
        
        d += f' C {cp1_x:.1f},{cp1_y:.1f} {cp2_x:.1f},{cp2_y:.1f} {x2},{y2}'
    
    return d


def create_svg_overlay(
    image_path: Path,
    xs: np.ndarray,
    y0: int,
    y1: int,
    output_path: Path,
    seed_x: Optional[int] = None,
) -> None:
    """Create an SVG file with the image embedded and the split line as an editable path."""
    
    # Load image to get dimensions
    img = Image.open(image_path)
    W, H = img.size
    
    # Embed image as base64
    import base64
    import io
    
    buffer = io.BytesIO()
    # Use PNG for lossless embedding
    img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
    img_mime = 'image/png'
    
    # Generate the path
    path_d = xs_to_svg_path(xs, y0, y1)
    
    if not path_d:
        print(f'  WARNING: No valid path for {image_path.name}')
        return
    
    # Build SVG document
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="{SVG_NS}"
     xmlns:inkscape="{INKSCAPE_NS}"
     xmlns:sodipodi="{SODIPODI_NS}"
     width="{W}"
     height="{H}"
     viewBox="0 0 {W} {H}"
     version="1.1">
  
  <sodipodi:namedview
      id="namedview1"
      pagecolor="#ffffff"
      bordercolor="#666666"
      borderopacity="1.0"
      inkscape:pageopacity="0.0"
      inkscape:pageshadow="2"
      inkscape:zoom="{max(W, H) / 1200:.4f}"
      inkscape:cx="{W / 2}"
      inkscape:cy="{H / 2}"
      inkscape:window-width="1600"
      inkscape:window-height="900"
      inkscape:window-x="0"
      inkscape:window-y="0"
      inkscape:window-maximized="1"
      inkscape:current-layer="layer1" />
  
  <!-- Background image -->
  <image
      id="background"
      x="0"
      y="0"
      width="{W}"
      height="{H}"
      xlink:href="data:{img_mime};base64,{img_b64}"
      sodipodi:insensitive="true"
      inkscape:locked="true" />
  
  <!-- Split line layer (editable) -->
  <g
      inkscape:groupmode="layer"
      id="layer1"
      inkscape:label="Split Line"
      sodipodi:insensitive="false">
    
    <path
        id="split-line"
        d="{path_d}"
        fill="none"
        stroke="{LINE_COLOR}"
        stroke-width="{LINE_WIDTH}"
        opacity="{LINE_OPACITY}"
        inkscape:label="Split Line (edit this)"
        sodipodi:nodetypes="{'c' * (len(path_d.split('C')) - 1) if 'C' in path_d else ''}" />
    
    <!-- Guide line: vertical at seed_x -->
    {
    f'''<line
        id="seed-guide"
        x1="{seed_x}"
        y1="{y0}"
        x2="{seed_x}"
        y2="{y1}"
        stroke="#00FF00"
        stroke-width="1.5"
        stroke-dasharray="10,5"
        opacity="0.5"
        inkscape:label="Seed X guide ({seed_x})"
        sodipodi:insensitive="true" />'''
    if seed_x is not None else ''
    }
  </g>
</svg>'''
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_content, encoding='utf-8')
    print(f'  ✓ {output_path.name} ({len(path_d.split("C"))} curve segments)')


def export_overlays(
    source_folder: Path,
    output_folder: Optional[Path] = None,
) -> None:
    """Export SVG overlays for all images with detected lines."""
    
    source_folder = Path(source_folder)
    if output_folder is None:
        output_folder = source_folder / 'svg_overlays'
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load overlays
    overlays_path = source_folder / OVERLAYS_FILENAME
    starts_path = source_folder / STARTS_FILENAME
    
    overlays_data = load_json_safely(overlays_path)
    starts_data = load_json_safely(starts_path)
    
    overlay_entries = overlays_data.get('entries', {})
    start_entries = starts_data.get('entries', {})
    
    if not overlay_entries:
        print(f'No overlay entries found in {overlays_path}')
        return
    
    # Find matching image files
    image_files = sorted(
        p for p in source_folder.iterdir()
        if p.suffix.lower() in IMG_EXTS and p.name in overlay_entries
    )
    
    if not image_files:
        print('No matching image files with overlay entries found.')
        return
    
    print(f'Exporting {len(image_files)} SVG overlay(s)...')
    
    for img_path in image_files:
        entry = overlay_entries[img_path.name]
        xs = xs_from_serializable(entry['xs'])
        y0 = int(entry['y0'])
        y1 = int(entry['y1'])
        
        start_entry = start_entries.get(img_path.name, {})
        seed_x = start_entry.get('seed_x')
        
        svg_path = output_folder / f'{img_path.stem}_overlay.svg'
        create_svg_overlay(img_path, xs, y0, y1, svg_path, seed_x)
    
    print(f'\nDone! SVGs saved to: {output_folder}')
    print(f'Open them in Inkscape, edit the red split line, and save.')
    print(f'Then run: python import_svg_overlays.py {output_folder}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python export_svg_overlays.py <source_folder> [output_folder]')
        sys.exit(1)
    
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    export_overlays(src, out)
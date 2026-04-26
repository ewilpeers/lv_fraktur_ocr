# diagnose_svg.py
"""Diagnose what's in an edited SVG overlay file."""
import re
import sys
from pathlib import Path

def diagnose_svg(svg_path: str) -> None:
    path = Path(svg_path)
    if not path.exists():
        print(f'File not found: {path}')
        return
    
    content = path.read_text(encoding='utf-8')
    print(f'File: {path.name}')
    print(f'Size: {len(content)} bytes')
    print()
    
    # Find all path elements
    paths = re.findall(r'<path[^>]*/?>', content, re.IGNORECASE)
    print(f'Found {len(paths)} path element(s):')
    
    for i, p in enumerate(paths):
        # Extract attributes
        id_match = re.search(r'\bid\s*=\s*["\']([^"\']+)["\']', p)
        d_match = re.search(r'\bd\s*=\s*["\']([^"\']*)["\']', p)
        
        id_val = id_match.group(1) if id_match else '(no id)'
        d_val = d_match.group(1) if d_match else '(no d attribute)'
        
        # Show first 200 chars of path data
        d_preview = d_val[:200] + '...' if len(d_val) > 200 else d_val
        
        print(f'\n  Path #{i+1}: id="{id_val}"')
        print(f'  d="{d_preview}"')
        
        # Count coordinates
        coords = re.findall(r'[-+]?\d*\.?\d+', d_val)
        print(f'  Coordinates: {len(coords)} values (≈{len(coords)//2} points)')
        
        # Show first few commands
        commands = re.findall(r'[MLHVCSQTAZmlhvcsqtaz]', d_val)[:10]
        print(f'  Commands: {" ".join(commands)}')
    
    # Check for groups
    groups = re.findall(r'<g[^>]*>', content, re.IGNORECASE)
    print(f'\nGroups: {len(groups)}')
    
    # Check if path is inside a transformed group
    if re.search(r'transform\s*=\s*["\'][^"\']*["\']', content):
        print('\n⚠️  WARNING: transform attributes found!')
        print('Inkscape may have applied transforms that shift coordinates.')
        print('Try: Object → Unset Transforms (or Ctrl+Alt+T) before saving.')
    
    # Check for sodipodi:nodetypes
    nodetypes = re.search(r'sodipodi:nodetypes\s*=\s*["\']([^"\']*)["\']', content)
    if nodetypes:
        print(f'\nNode types: {nodetypes.group(1)[:100]}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python diagnose_svg.py <path_to_svg>')
        sys.exit(1)
    diagnose_svg(sys.argv[1])
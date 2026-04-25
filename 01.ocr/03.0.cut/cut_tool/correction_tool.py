"""Manual correction tool for the vertical-line splitter.

Lets you:
  * pick a folder of scans,
  * navigate with Left/Right keys (or buttons),
  * see the auto-detected red overlay line,
  * adjust the seed x by typing a +/- offset (or by clicking on the image),
  * watch live mouse coordinates (page x,y AND delta from the current overlay),
  * "Save all" -> write starts.json and overlays.json to the chosen folder.

When you change the seed, the trace is re-run on the spot using the SAME
algorithm `find_vertical_line` uses in the notebook, so the overlay you
correct here is exactly what the notebook will get when it reads the JSONs.
"""
from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

from line_detection import (
    DEFAULT_CONFIG,
    find_vertical_line,
    load_json_safely,
    save_json,
    trace_from_seed,
    xs_from_serializable,
    xs_to_serializable,
)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
STARTS_FILENAME = 'starts.json'
OVERLAYS_FILENAME = 'overlays.json'


class CorrectionApp:
    def __init__(self, root: tk.Tk, folder: Path) -> None:
        self.root = root
        self.folder = Path(folder)
        self.cfg = dict(DEFAULT_CONFIG)

        # Per-image state, keyed by filename:
        #   {'seed_x': int, 'autogen': bool, 'xs': np.ndarray, 'y0': int, 'y1': int,
        #    'viewed': bool, 'orig_shape': (H, W)}
        self.entries: Dict[str, Dict] = {}

        # Load existing JSONs into entries (any image listed there counts as
        # already-viewed for save purposes -- we have data for it).
        starts_path = self.folder / STARTS_FILENAME
        overlays_path = self.folder / OVERLAYS_FILENAME
        starts = load_json_safely(starts_path)
        overlays = load_json_safely(overlays_path)

        files = sorted(p.name for p in self.folder.iterdir()
                       if p.suffix.lower() in IMG_EXTS)
        self.files = files
        if not files:
            messagebox.showerror('No images', f'No supported images in {self.folder}')
            root.destroy()
            return

        for fn in files:
            e: Dict = {'viewed': False}
            if fn in starts.get('entries', {}):
                s = starts['entries'][fn]
                e['seed_x'] = int(s['seed_x'])
                e['autogen'] = bool(s.get('autogen', True))
                e['viewed'] = True
            if fn in overlays.get('entries', {}):
                o = overlays['entries'][fn]
                e['xs'] = xs_from_serializable(o['xs'])
                e['y0'] = int(o['y0'])
                e['y1'] = int(o['y1'])
                e['overlay_autogen'] = bool(o.get('autogen', True))
                e['viewed'] = True
            self.entries[fn] = e

        self.cur_idx = 0
        self.display_scale = 1.0  # original_px / displayed_px
        self.current_pil_img: Optional[Image.Image] = None
        self.current_arr: Optional[np.ndarray] = None  # original RGB array
        self.tk_img: Optional[ImageTk.PhotoImage] = None

        self._build_ui()
        self._load_current(initial=True)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.title(f'Vertical-line correction tool -- {self.folder}')
        self.root.geometry('1100x900')

        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)

        self.btn_prev = ttk.Button(top, text='\u2190 Prev', command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT)
        self.btn_next = ttk.Button(top, text='Next \u2192', command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_file = ttk.Label(top, text='', font=('TkDefaultFont', 10, 'bold'))
        self.lbl_file.pack(side=tk.LEFT)

        ttk.Label(top, text='   Seed x:').pack(side=tk.LEFT)
        self.var_seed = tk.StringVar(value='')
        self.ent_seed = ttk.Entry(top, textvariable=self.var_seed, width=6)
        self.ent_seed.pack(side=tk.LEFT)
        self.ent_seed.bind('<Return>', lambda _e: self._apply_seed_entry())
        ttk.Button(top, text='Apply', command=self._apply_seed_entry).pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text='   Shift \u00b1 px:').pack(side=tk.LEFT)
        self.var_shift = tk.StringVar(value='')
        self.ent_shift = ttk.Entry(top, textvariable=self.var_shift, width=5)
        self.ent_shift.pack(side=tk.LEFT)
        self.ent_shift.bind('<Return>', lambda _e: self._apply_shift())
        ttk.Button(top, text='Shift', command=self._apply_shift).pack(side=tk.LEFT, padx=2)

        ttk.Button(top, text='Re-detect (auto)', command=self._redetect_auto
                   ).pack(side=tk.LEFT, padx=(12, 2))

        ttk.Button(top, text='Save all', command=self.save_all
                   ).pack(side=tk.RIGHT)

        # Image canvas
        self.canvas = tk.Canvas(self.root, bg='#222', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<Configure>', lambda _e: self._render())

        # Status bar
        status = ttk.Frame(self.root)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.lbl_pos = ttk.Label(status, text='', anchor='w')
        self.lbl_pos.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.lbl_status = ttk.Label(status, text='', anchor='e')
        self.lbl_status.pack(side=tk.RIGHT, padx=6)

        # Keyboard nav
        self.root.bind('<Left>',  lambda _e: self.prev_image())
        self.root.bind('<Right>', lambda _e: self.next_image())
        self.root.bind('<Control-s>', lambda _e: self.save_all())

    # ------------------------------------------------------------------
    # Image (de)load + render
    # ------------------------------------------------------------------
    def _load_current(self, initial: bool = False) -> None:
        fn = self.files[self.cur_idx]
        path = self.folder / fn
        self.current_pil_img = Image.open(path).convert('RGB')
        self.current_arr = np.array(self.current_pil_img)
        H, W = self.current_arr.shape[:2]

        e = self.entries[fn]
        e['orig_shape'] = (H, W)

        # If we don't yet have a seed/curve for this file, auto-detect now.
        # (This counts as the first time the image is "viewed".)
        if 'xs' not in e or 'seed_x' not in e:
            xs, (y0, y1), seed_x = find_vertical_line(self.current_arr, self.cfg)
            e['xs'] = xs
            e['y0'] = y0
            e['y1'] = y1
            e.setdefault('seed_x', seed_x)
            e.setdefault('autogen', True)
            e.setdefault('overlay_autogen', True)
        e['viewed'] = True

        self.var_seed.set(str(e['seed_x']))
        self.var_shift.set('')
        self.lbl_file.config(
            text=f'[{self.cur_idx+1}/{len(self.files)}] {fn}   ({W}\u00d7{H})'
        )
        self._render()

    def _render(self) -> None:
        if self.current_pil_img is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        W, H = self.current_pil_img.size
        scale = min(cw / W, ch / H)
        if scale <= 0 or scale != scale:  # NaN guard
            scale = 1.0
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        self.display_scale = W / new_w  # original_px per displayed_px

        # Resample image
        disp = self.current_pil_img.resize((new_w, new_h), Image.LANCZOS)
        # Burn the overlay into a copy
        e = self.entries[self.files[self.cur_idx]]
        xs = e.get('xs')
        y0 = e.get('y0', 0)
        y1 = e.get('y1', -1)
        if xs is not None and y1 >= y0:
            disp_arr = np.array(disp)
            for y in range(y0, y1 + 1):
                x_orig = int(xs[y])
                if x_orig < 0:
                    continue
                dy = int(y / self.display_scale)
                dx = int(x_orig / self.display_scale)
                if 0 <= dy < new_h:
                    lo = max(0, dx - 1)
                    hi = min(new_w, dx + 2)
                    disp_arr[dy, lo:hi] = (255, 0, 0)
            disp = Image.fromarray(disp_arr)

        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        # Center the image in the canvas
        self._img_offset = ((cw - new_w) // 2, (ch - new_h) // 2)
        self.canvas.create_image(self._img_offset[0], self._img_offset[1],
                                 anchor='nw', image=self.tk_img)
        self._update_status()

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------
    def _canvas_to_image(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        ox, oy = self._img_offset
        rx = cx - ox
        ry = cy - oy
        if rx < 0 or ry < 0:
            return None
        x = int(rx * self.display_scale)
        y = int(ry * self.display_scale)
        H, W = self.current_arr.shape[:2]
        if 0 <= x < W and 0 <= y < H:
            return x, y
        return None

    def _on_mouse_move(self, ev: tk.Event) -> None:
        coords = self._canvas_to_image(ev.x, ev.y)
        if coords is None:
            self.lbl_pos.config(text='')
            return
        x, y = coords
        e = self.entries[self.files[self.cur_idx]]
        xs = e.get('xs')
        if xs is not None and 0 <= y < len(xs) and xs[y] >= 0:
            dx = x - int(xs[y])
            self.lbl_pos.config(text=f'page (x={x}, y={y})   |   \u0394 from line: {dx:+d} px')
        else:
            self.lbl_pos.config(text=f'page (x={x}, y={y})   |   (no line at this row)')

    def _on_click(self, ev: tk.Event) -> None:
        coords = self._canvas_to_image(ev.x, ev.y)
        if coords is None:
            return
        x, _y = coords
        # Click sets new absolute seed_x and re-traces
        self._set_seed_x(x, mark_manual=True)

    # ------------------------------------------------------------------
    # Seed updates
    # ------------------------------------------------------------------
    def _set_seed_x(self, new_seed_x: int, mark_manual: bool) -> None:
        fn = self.files[self.cur_idx]
        e = self.entries[fn]
        H, W = self.current_arr.shape[:2]
        new_seed_x = max(0, min(W - 1, int(new_seed_x)))

        xs, (y0, y1) = trace_from_seed(self.current_arr, new_seed_x, self.cfg)
        e['xs'] = xs
        e['y0'] = y0
        e['y1'] = y1
        e['seed_x'] = new_seed_x
        if mark_manual:
            e['autogen'] = False
            e['overlay_autogen'] = False
        self.var_seed.set(str(new_seed_x))
        self.var_shift.set('')
        self._render()
        if y1 < y0:
            self._flash_status(f'Re-traced from x={new_seed_x}: NO LINE FOUND', warn=True)
        else:
            self._flash_status(f'Re-traced from x={new_seed_x}: rows {y0}-{y1}')

    def _apply_seed_entry(self) -> None:
        try:
            v = int(self.var_seed.get())
        except ValueError:
            self._flash_status('Seed x must be an integer', warn=True)
            return
        self._set_seed_x(v, mark_manual=True)

    def _apply_shift(self) -> None:
        try:
            d = int(self.var_shift.get())
        except ValueError:
            self._flash_status('Shift must be an integer (e.g. -12 or 8)', warn=True)
            return
        e = self.entries[self.files[self.cur_idx]]
        self._set_seed_x(int(e['seed_x']) + d, mark_manual=True)

    def _redetect_auto(self) -> None:
        # Recompute seed via auto-detection, mark autogen.
        xs, (y0, y1), seed_x = find_vertical_line(self.current_arr, self.cfg)
        fn = self.files[self.cur_idx]
        e = self.entries[fn]
        e['xs'] = xs
        e['y0'] = y0
        e['y1'] = y1
        e['seed_x'] = seed_x
        e['autogen'] = True
        e['overlay_autogen'] = True
        self.var_seed.set(str(seed_x))
        self.var_shift.set('')
        self._render()
        self._flash_status(f'Auto-redetect: seed x={seed_x}, rows {y0}-{y1}')

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def prev_image(self) -> None:
        if self.cur_idx > 0:
            self.cur_idx -= 1
            self._load_current()

    def next_image(self) -> None:
        if self.cur_idx < len(self.files) - 1:
            self.cur_idx += 1
            self._load_current()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    def save_all(self) -> None:
        starts_data = {
            'config_used': self.cfg,
            'entries': {},
        }
        overlays_data = {
            'config_used': self.cfg,
            'entries': {},
        }

        for fn, e in self.entries.items():
            if not e.get('viewed'):
                continue  # rule (5): unviewed images get no entry
            if 'seed_x' in e:
                starts_data['entries'][fn] = {
                    'seed_x': int(e['seed_x']),
                    'autogen': bool(e.get('autogen', True)),
                }
            if 'xs' in e and e.get('y1', -1) >= e.get('y0', 0):
                overlays_data['entries'][fn] = {
                    'y0': int(e['y0']),
                    'y1': int(e['y1']),
                    'xs': xs_to_serializable(e['xs']),
                    'autogen': bool(e.get('overlay_autogen', e.get('autogen', True))),
                }

        save_json(self.folder / STARTS_FILENAME, starts_data)
        save_json(self.folder / OVERLAYS_FILENAME, overlays_data)
        self._flash_status(
            f'Saved {len(starts_data["entries"])} starts, '
            f'{len(overlays_data["entries"])} overlays -> {self.folder}'
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def _flash_status(self, msg: str, warn: bool = False) -> None:
        self.lbl_status.config(text=msg, foreground=('red' if warn else 'black'))

    def _update_status(self) -> None:
        e = self.entries[self.files[self.cur_idx]]
        flag = 'manual' if not e.get('autogen', True) else 'auto'
        self._flash_status(f'seed_x={e.get("seed_x", "?")} ({flag})')


def main() -> None:
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        # Show a folder picker first
        root_pre = tk.Tk()
        root_pre.withdraw()
        folder = filedialog.askdirectory(title='Pick scan folder')
        root_pre.destroy()
        if not folder:
            return
        folder = Path(folder)

    if not folder.is_dir():
        print(f'Not a folder: {folder}', file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    CorrectionApp(root, folder)
    root.mainloop()


if __name__ == '__main__':
    main()

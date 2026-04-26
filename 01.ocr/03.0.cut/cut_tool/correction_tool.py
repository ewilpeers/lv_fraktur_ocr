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
    trace_from_seed_refined,
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
        self.display_scale = 1.0  # original_px / displayed_px (combined fit*zoom)
        self.fit_scale = 1.0      # original_px / displayed_px at fit-to-window
        self.zoom = 1.0           # user zoom multiplier on top of fit
        self.pan_x = 0.0          # pan offsets in canvas pixels
        self.pan_y = 0.0
        self._panning = False
        self._pan_start = (0, 0)
        self.current_pil_img: Optional[Image.Image] = None
        self.current_arr: Optional[np.ndarray] = None  # original RGB array
        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self._img_offset = (0, 0)

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
        ttk.Button(top, text='Fit', command=self._reset_view
                   ).pack(side=tk.LEFT, padx=2)

        ttk.Button(top, text='Save all', command=self.save_all
                   ).pack(side=tk.RIGHT)

        # Image canvas
        self.canvas = tk.Canvas(self.root, bg='#222', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Button-1>', self._on_click)
        self.canvas.bind('<ButtonRelease-1>', self._on_click_release)
        self.canvas.bind('<Configure>', lambda _e: self._render())
        # Zoom: mouse wheel (Linux uses Button-4/Button-5; Windows/Mac use <MouseWheel>)
        self.canvas.bind('<MouseWheel>', self._on_wheel)
        self.canvas.bind('<Button-4>', lambda e: self._on_wheel_step(e, +1))
        self.canvas.bind('<Button-5>', lambda e: self._on_wheel_step(e, -1))
        # Pan: middle-button drag, or shift+left-drag
        self.canvas.bind('<Button-2>', self._on_pan_start)
        self.canvas.bind('<B2-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_pan_end)
        self.canvas.bind('<Shift-Button-1>', self._on_pan_start)
        self.canvas.bind('<Shift-B1-Motion>', self._on_pan_drag)
        self.canvas.bind('<Shift-ButtonRelease-1>', self._on_pan_end)

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
        self.root.bind('<KeyPress-0>', lambda _e: self._reset_view())
        self.root.bind('<KeyPress-plus>',  lambda _e: self._zoom_at_center(+1))
        self.root.bind('<KeyPress-equal>', lambda _e: self._zoom_at_center(+1))
        self.root.bind('<KeyPress-minus>', lambda _e: self._zoom_at_center(-1))

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
        # Reset view on image change
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._render()

    def _render(self) -> None:
        if self.current_pil_img is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        W, H = self.current_pil_img.size

        # fit_scale: original_px per displayed_px when fitted (>=1.0 typical)
        fit = max(W / cw, H / ch, 1e-6)
        if fit <= 0 or fit != fit:
            fit = 1.0
        self.fit_scale = fit
        # Effective scale used for display: smaller value = bigger on screen.
        # display_scale = fit_scale / zoom -> at zoom=1 we fit the window;
        # at zoom=2 we double the on-screen size.
        self.display_scale = self.fit_scale / max(self.zoom, 1e-3)

        # Size of the entire image at the current effective scale
        full_disp_w = max(1, int(round(W / self.display_scale)))
        full_disp_h = max(1, int(round(H / self.display_scale)))

        # Compute placement: at zoom <= 1, image fits entirely; we ignore pan.
        if self.zoom <= 1.0 + 1e-6:
            self.pan_x = 0.0
            self.pan_y = 0.0
            ox = (cw - full_disp_w) // 2
            oy = (ch - full_disp_h) // 2
            disp = self.current_pil_img.resize((full_disp_w, full_disp_h),
                                               Image.LANCZOS)
            disp_arr = np.array(disp)
            # Burn overlay
            self._burn_overlay(disp_arr, full_disp_w, full_disp_h)
            self.tk_img = ImageTk.PhotoImage(Image.fromarray(disp_arr))
            self.canvas.delete('all')
            self._img_offset = (ox, oy)
            self._visible_src_origin = (0, 0)  # source x0,y0 currently rendered
            self._rendered_src_size = (W, H)   # source w,h currently rendered
            self.canvas.create_image(ox, oy, anchor='nw', image=self.tk_img)
            self._update_status()
            return

        # Zoomed in: render only the visible portion of the source.
        # Clamp pan so we don't scroll past the image.
        max_pan_x = max(0.0, full_disp_w - cw)
        max_pan_y = max(0.0, full_disp_h - ch)
        self.pan_x = max(0.0, min(max_pan_x, self.pan_x))
        self.pan_y = max(0.0, min(max_pan_y, self.pan_y))

        # Visible region in displayed coords (within the full displayed image)
        vis_x0 = int(self.pan_x)
        vis_y0 = int(self.pan_y)
        vis_w = min(cw, full_disp_w - vis_x0)
        vis_h = min(ch, full_disp_h - vis_y0)
        # Map back to source pixel coords
        src_x0 = int(round(vis_x0 * self.display_scale))
        src_y0 = int(round(vis_y0 * self.display_scale))
        src_x1 = int(round((vis_x0 + vis_w) * self.display_scale))
        src_y1 = int(round((vis_y0 + vis_h) * self.display_scale))
        src_x0 = max(0, src_x0); src_y0 = max(0, src_y0)
        src_x1 = min(W, max(src_x0 + 1, src_x1))
        src_y1 = min(H, max(src_y0 + 1, src_y1))

        crop = self.current_pil_img.crop((src_x0, src_y0, src_x1, src_y1))
        disp = crop.resize((max(1, vis_w), max(1, vis_h)), Image.LANCZOS)
        disp_arr = np.array(disp)
        self._visible_src_origin = (src_x0, src_y0)
        self._rendered_src_size = (src_x1 - src_x0, src_y1 - src_y0)
        self._burn_overlay(disp_arr, vis_w, vis_h,
                           src_x0=src_x0, src_y0=src_y0)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(disp_arr))
        self.canvas.delete('all')
        # When zoomed, we draw at the canvas origin
        self._img_offset = (0, 0)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img)
        self._update_status()

    def _burn_overlay(self, disp_arr: np.ndarray, disp_w: int, disp_h: int,
                      src_x0: int = 0, src_y0: int = 0) -> None:
        """Draw the red overlay line into disp_arr in place. src_x0/src_y0 are
        the source-pixel origin of the rendered region (0,0 if not zoomed)."""
        e = self.entries[self.files[self.cur_idx]]
        xs = e.get('xs')
        y0 = e.get('y0', 0)
        y1 = e.get('y1', -1)
        if xs is None or y1 < y0:
            return
        ds = self.display_scale
        for y in range(y0, y1 + 1):
            x_orig = int(xs[y])
            if x_orig < 0:
                continue
            dy = int(round((y - src_y0) / ds))
            dx = int(round((x_orig - src_x0) / ds))
            if 0 <= dy < disp_h:
                lo = max(0, dx - 1)
                hi = min(disp_w, dx + 2)
                if hi > lo:
                    disp_arr[dy, lo:hi] = (255, 0, 0)

    # ------------------------------------------------------------------
    # Coordinate handling
    # ------------------------------------------------------------------
    def _canvas_to_image(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        ox, oy = self._img_offset
        rx = cx - ox
        ry = cy - oy
        if rx < 0 or ry < 0:
            return None
        # Convert canvas-relative (rx, ry) -> source pixels.
        # When zoomed: visible region is anchored at _visible_src_origin in
        # source space, with display_scale = source_px / display_px.
        sox, soy = getattr(self, '_visible_src_origin', (0, 0))
        x = int(sox + rx * self.display_scale)
        y = int(soy + ry * self.display_scale)
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
        # Record press position; act only on release if the user didn't drag.
        self._click_press = (ev.x, ev.y)

    def _on_click_release(self, ev: tk.Event) -> None:
        press = getattr(self, '_click_press', None)
        if press is None:
            return
        self._click_press = None
        if abs(ev.x - press[0]) > 3 or abs(ev.y - press[1]) > 3:
            return  # was a drag, not a click
        coords = self._canvas_to_image(ev.x, ev.y)
        if coords is None:
            return
        x, _y = coords
        self._set_seed_x(x, mark_manual=True)

    # ------------------------------------------------------------------
    # Zoom + pan
    # ------------------------------------------------------------------
    ZOOM_FACTOR = 1.25
    ZOOM_MIN = 1.0
    ZOOM_MAX = 16.0

    def _on_wheel(self, ev: tk.Event) -> None:
        # Windows / macOS deliver delta in MouseWheel
        steps = 1 if getattr(ev, 'delta', 0) > 0 else -1
        self._on_wheel_step(ev, steps)

    def _on_wheel_step(self, ev: tk.Event, direction: int) -> None:
        # Zoom centered on the cursor: the source pixel under the cursor
        # should remain under the cursor after the zoom change.
        coords = self._canvas_to_image(ev.x, ev.y)
        if coords is None:
            anchor = None
        else:
            anchor = coords  # (src_x, src_y)
        new_zoom = self.zoom * (self.ZOOM_FACTOR if direction > 0
                                else 1.0 / self.ZOOM_FACTOR)
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, new_zoom))
        if abs(new_zoom - self.zoom) < 1e-6:
            return
        # Pre-zoom display_scale and pan
        old_disp_scale = self.display_scale
        new_disp_scale = self.fit_scale / max(new_zoom, 1e-3)
        if anchor is not None and new_zoom > 1.0 + 1e-6:
            src_x, src_y = anchor
            # Cursor canvas position under the new scale, with pan adjusted to
            # keep src_x, src_y aligned with ev.x, ev.y.
            self.pan_x = src_x / new_disp_scale - ev.x
            self.pan_y = src_y / new_disp_scale - ev.y
        elif new_zoom <= 1.0 + 1e-6:
            self.pan_x = 0.0
            self.pan_y = 0.0
        self.zoom = new_zoom
        self._render()

    def _zoom_at_center(self, direction: int) -> None:
        # Synthesize a 'wheel event' at canvas center for keyboard zoom
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        ev = type('E', (), {'x': cw // 2, 'y': ch // 2, 'delta': 0})()
        self._on_wheel_step(ev, direction)

    def _reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._render()

    def _on_pan_start(self, ev: tk.Event) -> None:
        self._panning = True
        self._pan_start = (ev.x, ev.y, self.pan_x, self.pan_y)
        self.canvas.config(cursor='fleur')

    def _on_pan_drag(self, ev: tk.Event) -> None:
        if not self._panning:
            return
        sx, sy, p0x, p0y = self._pan_start
        dx = ev.x - sx
        dy = ev.y - sy
        # Drag the image: moving mouse right shifts pan_x left (image moves right)
        self.pan_x = p0x - dx
        self.pan_y = p0y - dy
        self._render()

    def _on_pan_end(self, _ev: tk.Event) -> None:
        self._panning = False
        self.canvas.config(cursor='')

    # ------------------------------------------------------------------
    # Seed updates
    # ------------------------------------------------------------------
    def _set_seed_x(self, new_seed_x: int, mark_manual: bool) -> None:
        fn = self.files[self.cur_idx]
        e = self.entries[fn]
        H, W = self.current_arr.shape[:2]
        new_seed_x = max(0, min(W - 1, int(new_seed_x)))

        xs, (y0, y1) = trace_from_seed_refined(self.current_arr, new_seed_x, self.cfg)
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

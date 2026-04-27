"""
Glück Bible verse-JSON editor.

GUI for stepping through per-page JSON files in
  C:\\Users\\user\\Pictures\\BBL\\lv_fraktur_ocr\\gluck_1694_verses
alongside the corresponding IIIF page image from the BSB endpoint.

Two edit modes (radio toggle, session-global, defaults to "full_list"):
  - full_list : edit only the substring between [ and ] of the full_list field
                (preserves file's existing formatting outside that range)
  - raw       : edit the entire file as text

Save-on-navigate: pressing Next/Prev/Jump/Save validates the current edit
as JSON; on failure, navigation is blocked and the error is shown.

Image: scroll-wheel zoom, middle-click drag to pan when larger than viewport.
"""

import json
import re
import io
import os
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

from PIL import Image, ImageTk


# ------------------- configuration -------------------
JSON_DIR = Path(r"C:\Users\user\Pictures\BBL\lv_fraktur_ocr\gluck_1694_verses")
CACHE_DIR = Path(r"C:\Users\user\Pictures\BBL\.iiif_cache")
IIIF_URL_TEMPLATE = (
    "https://api.digitale-sammlungen.de/iiif/image/v2/"
    "bsb10914821_{padded}/full/1200,/0/default.jpg"
)
PAGE_FILE_TEMPLATE = "{padded}.json"   # e.g. 00055.json
PAD_WIDTH = 5

# View modes
VIEW_FULL_LIST = "full_list"
VIEW_RAW = "raw"


# ------------------- helpers -------------------

def padded(n: int) -> str:
    return f"{n:0{PAD_WIDTH}d}"


def page_num_from_filename(name: str):
    """Extract the integer page number from a filename like '00055.json'."""
    stem = Path(name).stem
    if stem.isdigit():
        return int(stem)
    return None


def list_existing_pages(directory: Path):
    """Return a sorted list of page numbers for which a JSON file exists."""
    if not directory.exists():
        return []
    nums = []
    for f in directory.iterdir():
        if f.suffix.lower() != ".json":
            continue
        n = page_num_from_filename(f.name)
        if n is not None:
            nums.append(n)
    return sorted(nums)


def find_full_list_range(text: str):
    """
    Locate the '[' and matching ']' for the "full_list": [ ... ] field
    in a JSON document, returning (open_bracket_idx, close_bracket_idx).
    Returns None if not found.
    Operates on the raw text — no JSON parsing — so formatting is preserved.
    """
    # Match  "full_list" : [   (whitespace flexible)
    m = re.search(r'"full_list"\s*:\s*\[', text)
    if not m:
        return None
    start = m.end() - 1   # position of the '['
    # Walk forward, tracking depth, ignoring brackets inside strings
    depth = 0
    i = start
    in_string = False
    string_quote = None
    escape = False
    while i < len(text):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == string_quote:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                string_quote = ch
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    return start, i
        i += 1
    return None


def fetch_image_bytes(page_num: int) -> bytes:
    """Fetch IIIF image with on-disk cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"bsb10914821_{padded(page_num)}_1200.jpg"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    url = IIIF_URL_TEMPLATE.format(padded=padded(page_num))
    req = Request(url, headers={"User-Agent": "gluck-editor/1.0"})
    with urlopen(req, timeout=30) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return data


# ------------------- the app -------------------

class GluckEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Glück Bible verse-JSON editor")
        self.geometry("1500x900")

        # state
        self.pages = list_existing_pages(JSON_DIR)
        if not self.pages:
            messagebox.showerror(
                "No JSONs found",
                f"No .json files found in:\n{JSON_DIR}\n\nExiting.",
            )
            self.destroy()
            return

        self.current_page = self.pages[0]
        self.view_mode = tk.StringVar(value=VIEW_FULL_LIST)
        self.dirty = False
        self.loaded_text = ""        # the raw file text loaded on disk
        self.editor_text_origin = "" # the text the editor currently displays, as loaded
        self.fl_range = None         # (start, end) of full_list contents in loaded_text

        # image state
        self.original_pil = None
        self.zoom = 1.0
        self.tk_image = None
        self.pan_start = None
        # Sticky view across navigation: pan stored as fractions in [0,1]
        # of the *scrollable region* (so it adapts to differing page sizes).
        # zoom is global. Initialized lazily on first image load.
        self.sticky_zoom = None
        self.sticky_pan_x = 0.0
        self.sticky_pan_y = 0.0
        # Suppress saving sticky state during programmatic scrolls right
        # after loading a new image.
        self._suppress_pan_capture = False

        self._build_ui()
        self._bind_keys()
        self._load_page(self.current_page)

    # ----- UI construction -----
    def _build_ui(self):
        # Top bar: navigation and view toggle
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        ttk.Button(top, text="◀ Prev", command=self.prev_page).pack(side=tk.LEFT)
        ttk.Button(top, text="Next ▶", command=self.next_page).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Save", command=self.save_only).pack(side=tk.LEFT, padx=4)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(top, text="Jump to page:").pack(side=tk.LEFT)
        self.jump_entry = ttk.Entry(top, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=4)
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_entry())
        ttk.Button(top, text="Go", command=self.jump_to_entry).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(top, text="View:").pack(side=tk.LEFT)
        ttk.Radiobutton(
            top, text="full_list only", value=VIEW_FULL_LIST,
            variable=self.view_mode, command=self._view_changed,
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            top, text="raw JSON", value=VIEW_RAW,
            variable=self.view_mode, command=self._view_changed,
        ).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.page_label = ttk.Label(top, text="", font=("TkDefaultFont", 10, "bold"))
        self.page_label.pack(side=tk.LEFT, padx=8)

        self.dirty_label = ttk.Label(top, text="", foreground="orange")
        self.dirty_label.pack(side=tk.LEFT, padx=8)

        # Status bar (errors etc.)
        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self, textvariable=self.status_var, foreground="red", anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

        # Main split: image on left, editor on right
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Image area with scrollbars
        img_frame = ttk.Frame(main)
        main.add(img_frame, weight=3)

        self.canvas = tk.Canvas(img_frame, bg="gray20", highlightthickness=0)

        # Wrap scrollbar -> canvas commands so dragging the bars also captures sticky pan
        def _xview(*args):
            self.canvas.xview(*args)
            self._capture_pan()
        def _yview(*args):
            self.canvas.yview(*args)
            self._capture_pan()
        hbar = ttk.Scrollbar(img_frame, orient=tk.HORIZONTAL, command=_xview)
        vbar = ttk.Scrollbar(img_frame, orient=tk.VERTICAL, command=_yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_img_id = None

        # zoom + pan bindings
        self.canvas.bind("<MouseWheel>", self._on_zoom)        # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_zoom)          # Linux up
        self.canvas.bind("<Button-5>", self._on_zoom)          # Linux down
        self.canvas.bind("<ButtonPress-2>", self._pan_start)   # middle button
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_end)

        # Editor area
        edit_frame = ttk.Frame(main)
        main.add(edit_frame, weight=2)

        self.editor = tk.Text(edit_frame, wrap=tk.NONE, undo=True,
                              font=("Consolas", 10))
        ev = ttk.Scrollbar(edit_frame, orient=tk.VERTICAL, command=self.editor.yview)
        eh = ttk.Scrollbar(edit_frame, orient=tk.HORIZONTAL, command=self.editor.xview)
        self.editor.configure(yscrollcommand=ev.set, xscrollcommand=eh.set)
        ev.pack(side=tk.RIGHT, fill=tk.Y)
        eh.pack(side=tk.BOTTOM, fill=tk.X)
        self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # mark dirty on edit
        self.editor.bind("<<Modified>>", self._on_modified)

    def _bind_keys(self):
        self.bind_all("<Control-s>", lambda e: self.save_only())
        self.bind_all("<Alt-Right>", lambda e: self.next_page())
        self.bind_all("<Alt-Left>", lambda e: self.prev_page())

    # ----- navigation -----
    def _load_page(self, page_num: int):
        self.current_page = page_num
        path = JSON_DIR / PAGE_FILE_TEMPLATE.format(padded=padded(page_num))
        if not path.exists():
            self.status_var.set(f"page {page_num}: file does not exist")
            return

        text = path.read_text(encoding="utf-8")
        self.loaded_text = text
        self.fl_range = find_full_list_range(text)

        # Choose what to put in editor based on view_mode
        self._populate_editor_from_loaded()

        # update page label
        self.page_label.config(text=f"Page {page_num}    ({path.name})")
        self.status_var.set("")
        self._set_dirty(False)

        # load image
        self._load_image(page_num)

    def _populate_editor_from_loaded(self):
        """Set editor contents from self.loaded_text per current view_mode."""
        mode = self.view_mode.get()
        if mode == VIEW_FULL_LIST and self.fl_range is not None:
            start, end = self.fl_range
            # contents BETWEEN brackets (exclusive on both sides)
            inner = self.loaded_text[start + 1: end]
            new_text = inner
        else:
            new_text = self.loaded_text
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", new_text)
        self.editor_text_origin = new_text
        self.editor.edit_modified(False)

    def _view_changed(self):
        # If user toggles view while dirty, we need to either save or revert.
        # Try save: if invalid, block view change.
        if self.dirty:
            ok, err = self._validate_and_save_to_disk()
            if not ok:
                # revert radio to opposite of just-clicked
                other = VIEW_RAW if self.view_mode.get() == VIEW_FULL_LIST else VIEW_FULL_LIST
                self.view_mode.set(other)
                messagebox.showerror("Save failed", f"Cannot switch view:\n\n{err}")
                return
        # Reload from disk so loaded_text matches what is on disk; then repopulate editor
        self._load_page(self.current_page)

    def next_page(self):
        if not self._save_before_navigate():
            return
        idx = self._index_of(self.current_page)
        if idx is None or idx + 1 >= len(self.pages):
            self.status_var.set("Already at last page.")
            return
        self._load_page(self.pages[idx + 1])

    def prev_page(self):
        if not self._save_before_navigate():
            return
        idx = self._index_of(self.current_page)
        if idx is None or idx - 1 < 0:
            self.status_var.set("Already at first page.")
            return
        self._load_page(self.pages[idx - 1])

    def jump_to_entry(self):
        val = self.jump_entry.get().strip()
        if not val.isdigit():
            self.status_var.set("Jump value must be an integer page number.")
            return
        target = int(val)
        # find target or skip to next existing >= target
        existing = [p for p in self.pages if p >= target]
        if not existing:
            self.status_var.set(f"No JSON file at or after page {target}.")
            return
        if not self._save_before_navigate():
            return
        self._load_page(existing[0])

    def save_only(self):
        ok, err = self._validate_and_save_to_disk()
        if not ok:
            self.status_var.set(err)
        else:
            self.status_var.set("Saved.")

    def _index_of(self, page_num):
        try:
            return self.pages.index(page_num)
        except ValueError:
            return None

    def _save_before_navigate(self) -> bool:
        """Return True if it's OK to navigate, False if blocked."""
        if not self.dirty:
            return True
        ok, err = self._validate_and_save_to_disk()
        if not ok:
            messagebox.showerror(
                "Cannot navigate",
                f"Save failed (JSON invalid). Fix or revert before navigating.\n\n{err}",
            )
            self.status_var.set(err)
            return False
        return True

    # ----- save / validation -----
    def _validate_and_save_to_disk(self):
        """
        Returns (True, '') on success, (False, error_message) on failure.
        Splices the editor's content back into the original file text without
        reformatting any byte outside the edited region.
        """
        editor_text = self.editor.get("1.0", "end-1c")
        mode = self.view_mode.get()

        if mode == VIEW_FULL_LIST and self.fl_range is not None:
            # Validate: editor_text must form a valid JSON array of ints when wrapped in [...]
            try:
                arr = json.loads("[" + editor_text + "]")
            except json.JSONDecodeError as e:
                return False, f"Invalid full_list JSON: {e}"
            if not isinstance(arr, list) or not all(isinstance(x, int) for x in arr):
                return False, "full_list must be a JSON array of integers."
            # Splice
            start, end = self.fl_range
            new_full_text = (
                self.loaded_text[: start + 1]
                + editor_text
                + self.loaded_text[end:]
            )
            # As a safety net, ensure the whole result is still parseable JSON
            try:
                json.loads(new_full_text)
            except json.JSONDecodeError as e:
                return False, f"After splicing, the file is no longer valid JSON: {e}"
        else:
            # Raw view: editor_text *is* the new full file text
            try:
                json.loads(editor_text)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"
            new_full_text = editor_text

        # Write to disk
        path = JSON_DIR / PAGE_FILE_TEMPLATE.format(padded=padded(self.current_page))
        try:
            path.write_text(new_full_text, encoding="utf-8")
        except OSError as e:
            return False, f"Disk write failed: {e}"

        # Update in-memory loaded_text and recompute full_list range
        self.loaded_text = new_full_text
        self.fl_range = find_full_list_range(new_full_text)
        self.editor_text_origin = editor_text
        self.editor.edit_modified(False)
        self._set_dirty(False)
        return True, ""

    # ----- image -----
    def _load_image(self, page_num: int):
        try:
            data = fetch_image_bytes(page_num)
        except (URLError, HTTPError, OSError) as e:
            self.status_var.set(f"Image fetch failed: {e}")
            self.original_pil = None
            self.canvas.delete("all")
            return
        try:
            self.original_pil = Image.open(io.BytesIO(data))
            self.original_pil.load()
        except Exception as e:
            self.status_var.set(f"Image decode failed: {e}")
            self.original_pil = None
            return
        # Apply sticky zoom (use 1.0 only on the very first image of the session)
        if self.sticky_zoom is None:
            self.sticky_zoom = 1.0
        self.zoom = self.sticky_zoom
        self._render_image()
        # Apply sticky pan as fractions of the (new) scrollregion.
        # Tk's xview_moveto expects a fraction in [0,1] of the scrollregion.
        # The fraction maps to "left edge of viewport" — same convention we
        # used to capture, so identical fractions yield equivalent placement.
        self._suppress_pan_capture = True
        self.canvas.xview_moveto(self.sticky_pan_x)
        self.canvas.yview_moveto(self.sticky_pan_y)
        # release the suppression after Tk has processed pending events
        self.after_idle(self._release_pan_suppression)

    def _release_pan_suppression(self):
        self._suppress_pan_capture = False

    def _capture_pan(self):
        """Save current scroll fractions to sticky state."""
        if self._suppress_pan_capture or self.original_pil is None:
            return
        # xview/yview return (left, right) fractions; we store the left edge
        x_frac = self.canvas.xview()[0]
        y_frac = self.canvas.yview()[0]
        self.sticky_pan_x = x_frac
        self.sticky_pan_y = y_frac

    def _render_image(self):
        if self.original_pil is None:
            return
        w, h = self.original_pil.size
        zw = max(1, int(w * self.zoom))
        zh = max(1, int(h * self.zoom))
        if (zw, zh) != self.original_pil.size:
            resized = self.original_pil.resize((zw, zh), Image.LANCZOS)
        else:
            resized = self.original_pil
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, zw, zh))

    def _on_zoom(self, event):
        # determine direction
        if hasattr(event, "delta") and event.delta != 0:
            direction = 1 if event.delta > 0 else -1
        else:
            direction = 1 if event.num == 4 else -1
        factor = 1.15 if direction > 0 else 1 / 1.15

        # Get cursor position in canvas coords for cursor-anchored zoom
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        old_zoom = self.zoom
        new_zoom = max(0.1, min(8.0, self.zoom * factor))
        if new_zoom == old_zoom:
            return
        self.zoom = new_zoom
        self.sticky_zoom = new_zoom
        self._render_image()
        # Adjust scroll so the cursor stays on the same image point
        ratio = new_zoom / old_zoom
        new_cx = cx * ratio
        new_cy = cy * ratio
        # how much to scroll so (new_cx, new_cy) is under (event.x, event.y)
        sw = max(1, int(self.original_pil.size[0] * new_zoom))
        sh = max(1, int(self.original_pil.size[1] * new_zoom))
        self.canvas.xview_moveto(max(0, (new_cx - event.x) / sw))
        self.canvas.yview_moveto(max(0, (new_cy - event.y) / sh))
        self._capture_pan()

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.pan_start = (event.x, event.y)

    def _pan_drag(self, event):
        if self.pan_start is None:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._capture_pan()

    def _pan_end(self, event):
        self.pan_start = None
        self._capture_pan()

    # ----- editor dirty tracking -----
    def _on_modified(self, event=None):
        # Tk fires <<Modified>> on any change, including programmatic ones; gate via our flag.
        if self.editor.edit_modified():
            current = self.editor.get("1.0", "end-1c")
            if current != self.editor_text_origin:
                self._set_dirty(True)
            else:
                self._set_dirty(False)
            self.editor.edit_modified(False)

    def _set_dirty(self, val: bool):
        self.dirty = val
        self.dirty_label.config(text="● unsaved changes" if val else "")


def main():
    if not JSON_DIR.exists():
        # Helpful early diagnostic
        print(f"Warning: JSON_DIR does not exist: {JSON_DIR}")
    app = GluckEditor()
    app.mainloop()


if __name__ == "__main__":
    main()

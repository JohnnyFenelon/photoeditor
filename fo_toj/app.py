"""FO_TOJ desktop GUI (CustomTkinter).

Implements the wireframe in PRD section 7:
    Header (folder picker) -> Action bar (AI scan + progress) ->
    Filter bar (dropdown + save) -> Preview canvas.

Threading model: the AI/heuristic scoring loop runs on a worker thread so the
Tk event loop stays responsive (PRD non-functional: no "Not Responding").
The worker never touches Tk directly; it pushes events onto a thread-safe queue
that the main thread drains via ``after`` polling.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Optional

import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox

from . import __app_name__, __version__
from . import image_io
from .filters import FILTER_NAMES, apply_filter
from .scoring import Scorer

# Preview is rendered from a downscaled copy to keep RAM and redraws cheap;
# filters for the final save are applied to the full-resolution original.
PREVIEW_MAX_SIZE = 1400

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# --- Worker -> UI messages -------------------------------------------------
@dataclass
class ProgressMsg:
    current: int
    total: int
    filename: str


@dataclass
class DoneMsg:
    best_path: Optional[str]
    best_score: float
    scored: int
    skipped: int
    backend: str


@dataclass
class ErrorMsg:
    text: str


class FoTojApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title(f"{__app_name__} — AI Photo Culling & Filters  v{__version__}")
        self.geometry("980x760")
        self.minsize(820, 620)

        # --- state ---
        self.folder: Optional[str] = None
        self.image_paths: list[str] = []
        self.best_path: Optional[str] = None
        self.base_full: Optional[Image.Image] = None   # full-res selected image
        self.base_preview: Optional[Image.Image] = None  # downscaled for display
        self.filtered_preview: Optional[Image.Image] = None
        self.current_filter: str = "Original"
        self._ctk_image: Optional[ctk.CTkImage] = None
        self._scanning = False

        self.scorer: Optional[Scorer] = None
        self.events: "queue.Queue" = queue.Queue()

        self._build_ui()
        self.after(80, self._poll_events)
        self.bind("<Configure>", self._on_resize)
        self._last_canvas_size = (0, 0)

    # ====================================================================== UI
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)  # preview row expands

        # --- Header: folder selection ---
        header = ctk.CTkFrame(self, corner_radius=10)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(
            header, text="📁  Select Folder", width=150, command=self.on_select_folder
        ).grid(row=0, column=0, padx=12, pady=12)

        self.folder_label = ctk.CTkLabel(
            header, text="No folder selected", anchor="w"
        )
        self.folder_label.grid(row=0, column=1, sticky="ew", padx=8)

        # --- Action bar: AI scan + progress ---
        action = ctk.CTkFrame(self, corner_radius=10)
        action.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        action.grid_columnconfigure(1, weight=1)

        self.scan_btn = ctk.CTkButton(
            action,
            text="🤖  AI: Find Best Photo",
            width=210,
            height=40,
            fg_color="#1f9d55",
            hover_color="#178a49",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.on_find_best,
            state="disabled",
        )
        self.scan_btn.grid(row=0, column=0, padx=12, pady=(12, 4))

        self.progress = ctk.CTkProgressBar(action)
        self.progress.set(0)
        self.progress.grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 4))
        self.progress.grid_remove()  # hidden until scanning starts

        self.status_label = ctk.CTkLabel(action, text="Ready", anchor="w")
        self.status_label.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=14, pady=(0, 10)
        )

        # --- Filter bar ---
        fbar = ctk.CTkFrame(self, corner_radius=10)
        fbar.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        fbar.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(fbar, text="Apply Filter:").grid(
            row=0, column=0, padx=(12, 6), pady=12
        )
        self.filter_var = ctk.StringVar(value="Original")
        self.filter_menu = ctk.CTkOptionMenu(
            fbar,
            values=FILTER_NAMES,
            variable=self.filter_var,
            width=180,
            command=self.on_filter_change,
        )
        self.filter_menu.grid(row=0, column=1, padx=6, pady=12)
        self.filter_menu.configure(state="disabled")

        self.apply_btn = ctk.CTkButton(
            fbar, text="Apply", width=80, command=lambda: self.on_filter_change(self.filter_var.get()),
            state="disabled",
        )
        self.apply_btn.grid(row=0, column=2, padx=6, pady=12)

        self.save_btn = ctk.CTkButton(
            fbar,
            text="💾  Save Result",
            width=150,
            fg_color="#d97706",
            hover_color="#b45309",
            command=self.on_save,
            state="disabled",
        )
        self.save_btn.grid(row=0, column=4, padx=12, pady=12, sticky="e")

        # --- Preview canvas ---
        preview_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#2b2b2b")
        preview_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(6, 6))
        preview_frame.grid_columnconfigure(0, weight=1)
        preview_frame.grid_rowconfigure(0, weight=1)

        self.preview_label = ctk.CTkLabel(
            preview_frame,
            text="Image preview will appear here",
            text_color="#888888",
            font=ctk.CTkFont(size=16),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # --- Footer / score readout ---
        self.score_label = ctk.CTkLabel(
            self, text="", anchor="w", text_color="#9aa0a6"
        )
        self.score_label.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 10))

    # ================================================================ actions
    def on_select_folder(self) -> None:
        if self._scanning:
            return
        folder = filedialog.askdirectory(title="Select a folder of photos")
        if not folder:
            return
        self.folder = folder
        self.set_status("Scanning folder…")
        self.image_paths = image_io.scan_folder(folder)
        count = len(self.image_paths)
        shown = folder if len(folder) < 60 else "…" + folder[-57:]
        self.folder_label.configure(text=f"Folder: {shown}   ({count} images found)")

        if count == 0:
            self.scan_btn.configure(state="disabled")
            self.set_status("No supported images found in this folder.")
            messagebox.showinfo(
                __app_name__,
                "No supported images were found.\n\n"
                "Supported: JPG, PNG, TIFF, BMP, WEBP"
                + (", and RAW files" if image_io.HAS_RAWPY else ""),
            )
        else:
            self.scan_btn.configure(state="normal")
            self.set_status(f"Ready — {count} images. Click 'Find Best Photo'.")

    def on_find_best(self) -> None:
        if self._scanning or not self.image_paths:
            return
        self._scanning = True
        self._set_busy(True)
        self.progress.grid()
        self.progress.set(0)
        self.set_status("Loading quality model…")

        worker = threading.Thread(target=self._scan_worker, daemon=True)
        worker.start()

    def _scan_worker(self) -> None:
        """Runs off the main thread: score every image, report the winner."""
        try:
            if self.scorer is None:
                self.scorer = Scorer(prefer_ai=True)

            paths = list(self.image_paths)
            total = len(paths)
            best_path: Optional[str] = None
            best_score = float("-inf")
            scored = 0
            skipped = 0

            for i, path in enumerate(paths, start=1):
                self.events.put(ProgressMsg(i, total, os.path.basename(path)))
                try:
                    from .scoring import SCORE_MAX_SIZE

                    img = image_io.load_image(path, max_size=SCORE_MAX_SIZE)
                    score = self.scorer.score(img)
                    scored += 1
                    if score > best_score:
                        best_score = score
                        best_path = path
                except Exception:
                    skipped += 1  # corrupt/unsupported -> skip, never crash
                    continue

            self.events.put(
                DoneMsg(
                    best_path=best_path,
                    best_score=best_score if best_path else 0.0,
                    scored=scored,
                    skipped=skipped,
                    backend=self.scorer.model_name,
                )
            )
        except Exception:
            self.events.put(ErrorMsg(traceback.format_exc()))

    # =============================================================== UI events
    def _poll_events(self) -> None:
        """Drain worker messages on the main thread."""
        try:
            while True:
                msg = self.events.get_nowait()
                if isinstance(msg, ProgressMsg):
                    self._handle_progress(msg)
                elif isinstance(msg, DoneMsg):
                    self._handle_done(msg)
                elif isinstance(msg, ErrorMsg):
                    self._handle_error(msg)
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _handle_progress(self, msg: ProgressMsg) -> None:
        self.progress.set(msg.current / max(1, msg.total))
        self.set_status(
            f"Analyzing {msg.current}/{msg.total}: {msg.filename}"
        )

    def _handle_done(self, msg: DoneMsg) -> None:
        self._scanning = False
        self._set_busy(False)
        self.progress.set(1.0)
        self.progress.grid_remove()

        if not msg.best_path:
            self.set_status("Could not score any images.")
            messagebox.showwarning(
                __app_name__,
                f"No images could be scored. Skipped {msg.skipped} file(s).",
            )
            return

        backend_kind = "AI" if "pyiqa" in msg.backend else "heuristic"
        self.set_status(
            f"Best photo: {os.path.basename(msg.best_path)}  "
            f"(scored {msg.scored}, skipped {msg.skipped})"
        )
        self.score_label.configure(
            text=(
                f"Engine: {msg.backend} [{backend_kind}]   •   "
                f"Best score: {msg.best_score:.3f}"
            )
        )
        self._load_selected(msg.best_path)

    def _handle_error(self, msg: ErrorMsg) -> None:
        self._scanning = False
        self._set_busy(False)
        self.progress.grid_remove()
        self.set_status("An error occurred during analysis.")
        messagebox.showerror(__app_name__, f"Analysis failed:\n\n{msg.text}")

    # =============================================================== preview
    def _load_selected(self, path: str) -> None:
        """Load the winning image full-res + a downscaled preview copy."""
        try:
            self.base_full = image_io.load_image(path)  # full resolution
            self.base_preview = image_io.downscale(self.base_full, PREVIEW_MAX_SIZE)
            self.best_path = path
        except Exception as exc:
            messagebox.showerror(__app_name__, f"Could not open image:\n{exc}")
            return

        self.filter_var.set("Original")
        self.current_filter = "Original"
        self.filter_menu.configure(state="normal")
        self.apply_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self._render_filter("Original")

    def on_filter_change(self, name: str) -> None:
        if self.base_preview is None:
            return
        self.current_filter = name
        self._render_filter(name)

    def _render_filter(self, name: str) -> None:
        """Apply *name* to the preview copy and display it."""
        if self.base_preview is None:
            return
        self.filtered_preview = apply_filter(self.base_preview, name)
        self._display(self.filtered_preview)

    def _display(self, img: Image.Image) -> None:
        """Fit *img* into the preview area, preserving aspect ratio."""
        self.update_idletasks()
        avail_w = max(200, self.preview_label.winfo_width() - 16)
        avail_h = max(200, self.preview_label.winfo_height() - 16)

        w, h = img.size
        scale = min(avail_w / w, avail_h / h, 1.0)
        disp_size = (max(1, int(w * scale)), max(1, int(h * scale)))

        self._ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=disp_size)
        self.preview_label.configure(image=self._ctk_image, text="")

    def _on_resize(self, event) -> None:
        # Re-fit the current preview when the window size changes meaningfully.
        if self.filtered_preview is None:
            return
        size = (self.winfo_width(), self.winfo_height())
        if abs(size[0] - self._last_canvas_size[0]) > 20 or abs(
            size[1] - self._last_canvas_size[1]
        ) > 20:
            self._last_canvas_size = size
            self._display(self.filtered_preview)

    # =================================================================== save
    def on_save(self) -> None:
        if self.base_full is None:
            return
        default_name = "FO_TOJ_result"
        if self.best_path:
            stem = os.path.splitext(os.path.basename(self.best_path))[0]
            default_name = f"{stem}_{self.current_filter.replace(' & ', '_').replace(' ', '')}"

        path = filedialog.asksaveasfilename(
            title="Save filtered image",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
        )
        if not path:
            return

        try:
            # Apply the chosen filter to the FULL-resolution image for export.
            self.set_status("Rendering full-resolution image…")
            self.update_idletasks()
            final = apply_filter(self.base_full, self.current_filter)

            ext = os.path.splitext(path)[1].lower()
            if ext == ".png":
                final.save(path, format="PNG")
            else:
                if ext not in (".jpg", ".jpeg"):
                    path += ".jpg"
                final.save(path, format="JPEG", quality=95, subsampling=0)

            self.set_status(f"Saved: {path}")
            messagebox.showinfo(__app_name__, f"Image saved successfully:\n{path}")
        except Exception as exc:
            self.set_status("Save failed.")
            messagebox.showerror(__app_name__, f"Could not save image:\n{exc}")

    # ================================================================ helpers
    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.scan_btn.configure(state=state if self.image_paths else "disabled")
        # During scan, keep filter/save disabled; they re-enable on load.
        if busy:
            self.filter_menu.configure(state="disabled")
            self.apply_btn.configure(state="disabled")
            self.save_btn.configure(state="disabled")


def main() -> None:
    app = FoTojApp()
    app.mainloop()


if __name__ == "__main__":
    main()

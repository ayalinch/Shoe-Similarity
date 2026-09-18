#!/usr/bin/env python3
"""
app.py — Sole Match Finder (desktop GUI)

Two ways to use it:

  1. "Library Search" tab — point at a folder of reference sole photos,
     upload one query photo, and see the closest matches ranked by
     similarity, with a detail view for each candidate.

  2. "Compare Two Photos" tab — pick exactly two sole photos directly
     (no folder / library needed) and see whether they're a high enough
     match, with the same full comparison view.

Run with:
    python app.py

First run in a new environment:
    pip install -r requirements.txt
"""

from __future__ import annotations

import csv
import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from engine import (
    IDENTICAL_THRESH,
    LIKELY_THRESH,
    compute_similarity,
    extract_features,
    label_for_score,
)
from library import MatchResult, ShoeLibrary
from viz import build_pair_card

CONFIG_DIR = Path.home() / ".shoe_matcher"
CONFIG_PATH = CONFIG_DIR / "config.json"

THUMB = 84             # results-list thumbnail size (px)
PREVIEW = 260           # query photo preview size (px)
COMPARE_PREVIEW = 200   # compare-tab thumbnail size (px)

LABEL_COLORS = {
    "Identical": "#1f9d55",
    "Likely match": "#d18b1a",
    "No match": "#b0384a",
}


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg))
    except OSError:
        pass


def cv2_to_photoimage(bgr_img, max_size=None):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if max_size:
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img)


class DetailWindow(tk.Toplevel):
    """Full comparison card for one pair of photos, with a Save button.

    `title_a` lets callers label the left-hand photo (defaults to "Query
    photo" for the library-search flow); the right-hand label always comes
    from match_result.name.
    """

    def __init__(self, master, query_features, match_result, title_a="Query photo"):
        super().__init__(master)
        self.match_result = match_result
        self.title(f"Comparison — {title_a} vs {match_result.name}")
        self.geometry("960x760")

        card_bgr = build_pair_card(
            query_features, match_result.features,
            match_result.score, match_result.sub,
            title_a=title_a, title_b=match_result.name,
        )
        self._card_bgr = card_bgr

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, bg="#101010", highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        photo = cv2_to_photoimage(card_bgr)
        self._photo_ref = photo  # keep alive
        img_label = tk.Label(canvas, image=photo, bg="#101010")
        canvas.create_window((0, 0), window=img_label, anchor="nw")
        img_label.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))

        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill="x", side="bottom")
        ttk.Button(btn_bar, text="Save comparison image...",
                   command=self._save).pack(side="right", padx=8, pady=8)

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg")],
            initialfile=f"match_{self.match_result.name}.jpg",
        )
        if not path:
            return
        cv2.imwrite(path, self._card_bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
        messagebox.showinfo("Saved", f"Saved to {path}")


class ResultRow(ttk.Frame):
    def __init__(self, master, rank, match_result, on_view):
        super().__init__(master, padding=6)
        self.match_result = match_result

        thumb_bgr = match_result.features["img_sq"]
        photo = cv2_to_photoimage(thumb_bgr, max_size=THUMB)
        self._photo_ref = photo
        tk.Label(self, image=photo).grid(row=0, column=0, rowspan=2, padx=(0, 10))

        ttk.Label(self, text=f"#{rank}  {match_result.name}",
                  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=1, sticky="w")

        pct = int(round(match_result.score * 100))
        color = LABEL_COLORS.get(match_result.label, "#888888")
        sub_frame = ttk.Frame(self)
        sub_frame.grid(row=1, column=1, sticky="w")
        tk.Label(sub_frame, text=f"{pct}%", font=("TkDefaultFont", 9)).pack(side="left")
        tk.Label(sub_frame, text=f"  {match_result.label}", fg=color,
                 font=("TkDefaultFont", 9, "bold")).pack(side="left")

        ttk.Button(self, text="View comparison",
                   command=lambda: on_view(match_result)).grid(row=0, column=2, rowspan=2, padx=8)

        self.columnconfigure(1, weight=1)
        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, columnspan=3,
                                                        sticky="ew", pady=(6, 0))


class ScrollableResults(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                         lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>",
                          lambda e: self.canvas.itemconfigure(self._window, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-2, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(2, "units"))

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 60), "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sole Match Finder")
        self.geometry("1060x800")
        self.minsize(880, 640)

        self.library: ShoeLibrary | None = None
        self.query_path: Path | None = None
        self.query_features: dict | None = None
        self._task_queue: queue.Queue = queue.Queue()
        self._results = []

        # -- Compare-two-photos tab state --
        self.cmp_path = {"a": None, "b": None}
        self.cmp_features = {"a": None, "b": None}
        self.cmp_widgets = {"a": {}, "b": {}}
        self.cmp_match_result: MatchResult | None = None

        self._build_ui()

        cfg = load_config()
        last_folder = cfg.get("library_folder")
        if last_folder and Path(last_folder).is_dir():
            self._set_library_folder(Path(last_folder), auto_load=True)

        self.after(100, self._poll_queue)

    # ---------------------------------------------------------------- UI

    def _build_ui(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        library_tab = ttk.Frame(notebook)
        compare_tab = ttk.Frame(notebook)
        notebook.add(library_tab, text="Library Search")
        notebook.add(compare_tab, text="Compare Two Photos")

        self._build_library_tab(library_tab)
        self._build_compare_tab(compare_tab)

    def _build_library_tab(self, root):
        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Reference library folder:").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar(value="(none selected)")
        ttk.Label(top, textvariable=self.folder_var, foreground="#555").grid(
            row=0, column=1, sticky="w", padx=8)
        ttk.Button(top, text="Choose folder...", command=self._choose_folder).grid(
            row=0, column=2, padx=4)
        self.build_btn = ttk.Button(top, text="Build / refresh library",
                                     command=self._build_library, state="disabled")
        self.build_btn.grid(row=0, column=3, padx=4)

        self.progress = ttk.Progressbar(top, mode="determinate", length=200, maximum=100)
        self.progress.grid(row=0, column=4, padx=(8, 4))
        self.progress_pct_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.progress_pct_var, width=5).grid(row=0, column=5)
        self.status_var = tk.StringVar(value="No library loaded.")
        ttk.Label(top, textvariable=self.status_var, foreground="#555").grid(
            row=1, column=0, columnspan=5, sticky="w", pady=(4, 0))
        top.columnconfigure(1, weight=1)

        body = ttk.Frame(root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Left: query photo
        left = ttk.LabelFrame(body, text="Query photo", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self.preview_label = tk.Label(left, text="No photo selected", width=32, height=14,
                                       bg="#1c1c1c", fg="#888")
        self.preview_label.pack(pady=(0, 8))
        ttk.Button(left, text="Choose photo...", command=self._choose_photo).pack(fill="x")
        self.query_name_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.query_name_var, foreground="#555",
                  wraplength=220).pack(pady=(6, 10))

        ttk.Label(left, text="Show top:").pack(anchor="w")
        self.topk_var = tk.IntVar(value=15)
        ttk.Spinbox(left, from_=1, to=100, textvariable=self.topk_var, width=6).pack(anchor="w")

        self.find_btn = ttk.Button(left, text="Find matches", command=self._find_matches,
                                    state="disabled")
        self.find_btn.pack(fill="x", pady=(12, 4))
        ttk.Button(left, text="Export results to CSV...",
                   command=self._export_csv).pack(fill="x")

        legend = ttk.LabelFrame(left, text="Thresholds", padding=8)
        legend.pack(fill="x", pady=(14, 0))
        tk.Label(legend, text=f"Identical  ≥ {int(IDENTICAL_THRESH*100)}%",
                 fg=LABEL_COLORS["Identical"]).pack(anchor="w")
        tk.Label(legend, text=f"Likely     ≥ {int(LIKELY_THRESH*100)}%",
                 fg=LABEL_COLORS["Likely match"]).pack(anchor="w")
        tk.Label(legend, text=f"No match  < {int(LIKELY_THRESH*100)}%",
                 fg=LABEL_COLORS["No match"]).pack(anchor="w")

        # Right: results
        right = ttk.LabelFrame(body, text="Matches", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)
        self.results_panel = ScrollableResults(right)
        self.results_panel.grid(row=0, column=0, sticky="nsew")
        self.empty_hint = ttk.Label(
            self.results_panel.inner,
            text="Choose a library folder and a query photo, then click"
                 " “Find matches”.",
            foreground="#777", padding=20)
        self.empty_hint.pack(anchor="w")

    def _build_compare_tab(self, root):
        ttk.Label(
            root,
            text="Pick any two sole photos and see how closely they match —"
                 " no library folder needed.",
            foreground="#555", padding=(10, 10, 10, 4),
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        self._build_compare_slot(root, "a", "Photo A").grid(
            row=1, column=0, sticky="nsew", padx=(10, 5), pady=(0, 10))
        self._build_compare_slot(root, "b", "Photo B").grid(
            row=1, column=1, sticky="nsew", padx=(5, 10), pady=(0, 10))

        action = ttk.Frame(root, padding=(10, 0))
        action.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.compare_btn = ttk.Button(
            action, text="Compare these two photos",
            command=self._run_compare, state="disabled")
        self.compare_btn.pack(side="left")

        result = ttk.LabelFrame(root, text="Result", padding=14)
        result.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.cmp_result_var = tk.StringVar(value="Choose two photos, then click Compare.")
        self.cmp_result_label = tk.Label(
            result, textvariable=self.cmp_result_var,
            font=("TkDefaultFont", 15, "bold"), fg="#888")
        self.cmp_result_label.pack(anchor="w")
        self.cmp_view_btn = ttk.Button(
            result, text="View full comparison...",
            command=self._open_compare_detail, state="disabled")
        self.cmp_view_btn.pack(anchor="w", pady=(8, 0))

    def _build_compare_slot(self, root, slot, title):
        frame = ttk.LabelFrame(root, text=title, padding=10)

        preview = tk.Label(frame, text="No photo selected", width=26, height=12,
                            bg="#1c1c1c", fg="#888")
        preview.pack(pady=(0, 8))
        ttk.Button(frame, text="Choose photo...",
                   command=lambda: self._choose_compare_photo(slot)).pack(fill="x")
        name_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=name_var, foreground="#555",
                  wraplength=220).pack(pady=(6, 0))

        self.cmp_widgets[slot] = {"preview": preview, "name_var": name_var}
        return frame

    # ---------------------------------------------------------- library

    def _choose_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder of reference shoe photos")
        if folder:
            self._set_library_folder(Path(folder))

    def _set_library_folder(self, folder: Path, auto_load: bool = False):
        self.library = ShoeLibrary(folder)
        self.library.load_cache()
        self.folder_var.set(str(folder))
        self.build_btn.config(state="normal")
        n = len(self.library)
        if n:
            self.status_var.set(f"{n} image(s) indexed (from a previous run)."
                                 " Click “Build / refresh library” to pick up changes.")
        else:
            self.status_var.set("Folder selected. Click “Build / refresh library” to index it.")
        save_config({"library_folder": str(folder)})
        self._update_find_btn()

    def _build_library(self):
        if self.library is None:
            return
        self.build_btn.config(state="disabled")
        images = self.library.list_images()
        if not images:
            messagebox.showwarning(
                "No images found",
                f"No image files found in:\n{self.library.folder}")
            self.build_btn.config(state="normal")
            return
        self._reset_progress()
        self.status_var.set("Indexing library...")

        def worker():
            def on_progress(done, total, name):
                self._task_queue.put(("progress", done, total, f"Indexing {done}/{total}: {name}"))
            try:
                added, skipped, removed = self.library.build(progress_callback=on_progress)
                self._task_queue.put(("library_done", added, skipped, removed))
            except Exception as exc:  # surface errors instead of a silent hang
                self._task_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------- query photo

    def _choose_photo(self):
        path = filedialog.askopenfilename(
            title="Choose a photo of the shoe/boot sole",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self.query_path = Path(path)
        self.query_name_var.set(self.query_path.name)
        self.status_var.set("Processing query photo...")
        self.find_btn.config(state="disabled")

        def worker():
            feats = extract_features(self.query_path)
            self._task_queue.put(("query_done", feats))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------------------------------------- matching

    def _update_find_btn(self):
        ready = bool(self.library is not None and len(self.library) and self.query_features is not None)
        self.find_btn.config(state="normal" if ready else "disabled")

    def _find_matches(self):
        if self.library is None or self.query_features is None:
            return
        top_k = self.topk_var.get()
        self.find_btn.config(state="disabled")
        self._reset_progress()
        self.status_var.set("Scoring against library...")

        def worker():
            def on_progress(done, total, name):
                self._task_queue.put(("progress", done, total, f"Comparing {done}/{total}: {name}"))
            try:
                results = self.library.query(self.query_features, top_k=top_k,
                                              progress_callback=on_progress)
                self._task_queue.put(("matches_done", results))
            except Exception as exc:
                self._task_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, results):
        self._results = results
        self.results_panel.clear()
        if not results:
            ttk.Label(self.results_panel.inner, text="No matches found.",
                      foreground="#777", padding=20).pack(anchor="w")
            return
        for rank, r in enumerate(results, start=1):
            ResultRow(self.results_panel.inner, rank, r, self._open_detail).pack(
                fill="x", expand=True)

    def _open_detail(self, match_result):
        DetailWindow(self, self.query_features, match_result)

    def _export_csv(self):
        if not self._results:
            messagebox.showinfo("Nothing to export", "Run “Find matches” first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="shoe_matches.csv")
        if not path:
            return
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "filename", "score_pct", "label"])
            for rank, r in enumerate(self._results, start=1):
                w.writerow([rank, r.name, int(round(r.score * 100)), r.label])
        messagebox.showinfo("Exported", f"Saved to {path}")

    # ---------------------------------------------------------- compare two photos

    def _choose_compare_photo(self, slot):
        path = filedialog.askopenfilename(
            title=f"Choose photo {slot.upper()}",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        path = Path(path)
        self.cmp_path[slot] = path
        self.cmp_features[slot] = None
        self.cmp_widgets[slot]["name_var"].set(f"{path.name}  (processing...)")
        self._set_compare_ready()
        self.cmp_view_btn.config(state="disabled")
        self.cmp_result_var.set("Choose two photos, then click Compare.")
        self.cmp_result_label.config(fg="#888")

        def worker():
            feats = extract_features(path)
            self._task_queue.put(("cmp_feat_done", slot, path, feats))

        threading.Thread(target=worker, daemon=True).start()

    def _set_compare_ready(self):
        ready = self.cmp_features["a"] is not None and self.cmp_features["b"] is not None
        self.compare_btn.config(state="normal" if ready else "disabled")

    def _run_compare(self):
        feats_a, feats_b = self.cmp_features["a"], self.cmp_features["b"]
        if feats_a is None or feats_b is None:
            return
        self.compare_btn.config(state="disabled")
        self.cmp_view_btn.config(state="disabled")
        self.cmp_result_var.set("Comparing...")
        self.cmp_result_label.config(fg="#888")

        def worker():
            try:
                score, sub = compute_similarity(feats_a, feats_b)
                self._task_queue.put(("cmp_done", score, sub))
            except Exception as exc:
                self._task_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_compare_detail(self):
        if self.cmp_match_result is None:
            return
        title_a = self.cmp_path["a"].name if self.cmp_path["a"] else "Photo A"
        DetailWindow(self, self.cmp_features["a"], self.cmp_match_result, title_a=title_a)

    # ---------------------------------------------------------- event loop

    def _reset_progress(self):
        self.progress.config(mode="determinate", maximum=100, value=0)
        self.progress_pct_var.set("0%")

    def _set_progress_pct(self, done, total):
        pct = int(round(done * 100 / total)) if total else 100
        self.progress.config(value=pct)
        self.progress_pct_var.set(f"{pct}%")

    def _poll_queue(self):
        try:
            while True:
                msg = self._task_queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    _, done, total, status_text = msg
                    self._set_progress_pct(done, total)
                    self.status_var.set(status_text)
                elif kind == "library_done":
                    _, added, skipped, removed = msg
                    self._set_progress_pct(1, 1)
                    n = len(self.library)
                    self.status_var.set(
                        f"Library ready: {n} image(s) indexed "
                        f"({added} processed, {skipped} unchanged, {removed} removed).")
                    self.build_btn.config(state="normal")
                    self._update_find_btn()
                elif kind == "query_done":
                    _, feats = msg
                    self.query_features = feats
                    if feats is None:
                        self.status_var.set(
                            "Could not find a usable tread pattern in that photo — "
                            "try a clearer, more direct shot of the sole.")
                        self.preview_label.config(image="", text="No usable tread found",
                                                   compound="center")
                    else:
                        photo = cv2_to_photoimage(feats["img_sq"], max_size=PREVIEW)
                        self.preview_label.image = photo
                        self.preview_label.config(image=photo, text="")
                        self.status_var.set("Query photo ready.")
                    self._update_find_btn()
                elif kind == "matches_done":
                    _, results = msg
                    self._set_progress_pct(1, 1)
                    self._show_results(results)
                    self.status_var.set(f"Found {len(results)} candidate match(es).")
                    self.find_btn.config(state="normal")
                elif kind == "cmp_feat_done":
                    _, slot, path, feats = msg
                    self.cmp_features[slot] = feats
                    widgets = self.cmp_widgets[slot]
                    if feats is None:
                        widgets["name_var"].set(
                            f"{path.name}  — no usable tread found; try a clearer photo.")
                        widgets["preview"].config(image="", text="No usable tread found",
                                                   compound="center")
                    else:
                        photo = cv2_to_photoimage(feats["img_sq"], max_size=COMPARE_PREVIEW)
                        widgets["preview"].image = photo
                        widgets["preview"].config(image=photo, text="")
                        widgets["name_var"].set(path.name)
                    self._set_compare_ready()
                elif kind == "cmp_done":
                    _, score, sub = msg
                    label = label_for_score(score)
                    pct = int(round(score * 100))
                    color = LABEL_COLORS.get(label, "#888888")
                    self.cmp_result_var.set(f"{pct}% match  —  {label}")
                    self.cmp_result_label.config(fg=color)
                    self.cmp_match_result = MatchResult(
                        score=score, sub=sub, label=label,
                        name=self.cmp_path["b"].name if self.cmp_path["b"] else "Photo B",
                        path=self.cmp_path["b"], features=self.cmp_features["b"],
                    )
                    self.cmp_view_btn.config(state="normal")
                    self._set_compare_ready()
                elif kind == "error":
                    self._reset_progress()
                    self.status_var.set("Error.")
                    messagebox.showerror("Error", msg[1])
                    self.build_btn.config(state="normal")
                    self.find_btn.config(state="normal")
                    self._set_compare_ready()
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

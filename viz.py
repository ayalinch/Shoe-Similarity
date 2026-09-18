"""
viz.py — Build a single composited comparison image (BGR numpy array) for
two shoe feature dicts: photos with tread outlined, normalized tread
binaries side by side, the cyan/yellow/white edge overlay, and a bar chart
of every sub-score. Used by the GUI's match-detail view, and can also be
saved straight to a .jpg.
"""

import cv2
import numpy as np

from engine import NORM_H, NORM_W, make_overlay_image, IDENTICAL_THRESH, LIKELY_THRESH

PANEL_W = 420
PAD = 10
HEADER_H = 46
LABEL_H = 22
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_COLOR = (200, 220, 255)

SUB_LABELS = [
    ("pc_e", "Edge alignment (phase corr)"),
    ("pc_t", "Tread alignment (phase corr)"),
    ("ssim", "Structural similarity (SSIM)"),
    ("orb", "Keypoint matches (ORB)"),
    ("iou", "Tread overlap (IoU)"),
    ("zone", "Zone density correlation"),
    ("eoh", "Edge orientation histogram"),
    ("hu", "Shape moments (Hu)"),
    ("fd", "Contour shape (Fourier)"),
]


def _tight_fit(src, target_w, pad=PAD):
    sh, sw = src.shape[:2]
    sc = (target_w - 2 * pad) / max(sw, 1)
    nw = max(1, int(sw * sc))
    nh = max(1, int(sh * sc))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    panel_h = nh + 2 * pad
    canvas = np.zeros((panel_h, target_w, 3), np.uint8)
    y0 = pad
    x0 = (target_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas, nh


def _label(img, text, x, y, scale=0.42, color=LABEL_COLOR):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def _pad_to_height(panels, target_h):
    out = []
    for p in panels:
        ph = p.shape[0]
        if ph < target_h:
            strip = np.zeros((target_h - ph, p.shape[1], 3), np.uint8)
            p = np.vstack([p, strip])
        out.append(p)
    return out


def _photo_panel(f, target_w):
    photo = f["img_sq"].copy()
    tm = f["tread_mask"]
    ys, xs = np.where(tm > 0)
    if len(ys) > 0:
        margin = 20
        y0c = max(0, ys.min() - margin)
        y1c = min(photo.shape[0], ys.max() + margin)
        x0c = max(0, xs.min() - margin)
        x1c = min(photo.shape[1], xs.max() + margin)
        photo_crop = photo[y0c:y1c, x0c:x1c]
        tm_crop = tm[y0c:y1c, x0c:x1c]
    else:
        photo_crop = photo
        tm_crop = tm
    photo_crop = photo_crop.copy()
    photo_crop[cv2.Canny(tm_crop, 50, 150) > 0] = (0, 210, 0)
    return _tight_fit(photo_crop, target_w)


def build_pair_card(fa, fb, score, sub, title_a="Query", title_b=None):
    title_b = title_b or fb.get("name", "match")
    n = 2
    total_w = PANEL_W * n + 4 * (n - 1)

    header = np.full((HEADER_H, total_w, 3), 16, np.uint8)
    pct = int(round(score * 100))
    if score >= IDENTICAL_THRESH:
        label, sc_color = "IDENTICAL", (60, 200, 60)
    elif score >= LIKELY_THRESH:
        label, sc_color = "LIKELY MATCH", (40, 170, 230)
    else:
        label, sc_color = "NO MATCH", (90, 90, 220)
    title = f"{title_a}  vs  {title_b}    {pct}%  [{label}]"
    cv2.putText(header, title, (10, 30), FONT, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(header, title, (10, 30), FONT, 0.62, sc_color, 1, cv2.LINE_AA)

    divider = np.full((4, total_w, 3), 45, np.uint8)

    # Photo row
    panel_a, ha = _photo_panel(fa, PANEL_W)
    panel_b, hb = _photo_panel(fb, PANEL_W)
    max_h = max(panel_a.shape[0], panel_b.shape[0])
    panel_a, panel_b = _pad_to_height([panel_a, panel_b], max_h)
    photo_row = np.hstack([panel_a, np.full((max_h, 4, 3), 45, np.uint8), panel_b])
    label_strip = np.full((LABEL_H, total_w, 3), 22, np.uint8)
    _label(label_strip, title_a, 8, 16)
    _label(label_strip, title_b[:60], PANEL_W + 12, 16)

    # Binary tread row
    bin_a, _ = _tight_fit(fa["tread_bw"], PANEL_W)
    flip_used = sub.get("flip") == "flip"
    bin_b_src = fb["tread_bw_flip"] if flip_used else fb["tread_bw"]
    bin_b, _ = _tight_fit(bin_b_src, PANEL_W)
    max_hb = max(bin_a.shape[0], bin_b.shape[0])
    bin_a, bin_b = _pad_to_height([bin_a, bin_b], max_hb)
    bin_row = np.hstack([bin_a, np.full((max_hb, 4, 3), 45, np.uint8), bin_b])
    bin_label_strip = np.full((LABEL_H, total_w, 3), 22, np.uint8)
    _label(bin_label_strip, "Normalized tread (what the algorithm compares)", 8, 16)
    if flip_used:
        _label(bin_label_strip, "(mirrored match — best orientation)", PANEL_W + 12, 16)

    # Overlay row
    edge_b_src = fb["edge_pat_flip"] if flip_used else fb["edge_pat"]
    overlay = make_overlay_image(fa["edge_pat"], edge_b_src)
    overlay_disp = cv2.resize(overlay, (NORM_W * 2, NORM_H * 2), interpolation=cv2.INTER_NEAREST)
    overlay_canvas = np.zeros((overlay_disp.shape[0] + 2 * PAD, total_w, 3), np.uint8)
    x0 = (total_w - overlay_disp.shape[1]) // 2
    overlay_canvas[PAD:PAD + overlay_disp.shape[0], x0:x0 + overlay_disp.shape[1]] = overlay_disp
    overlay_label_strip = np.full((LABEL_H, total_w, 3), 22, np.uint8)
    _label(overlay_label_strip,
           "Edge overlay:  yellow = query only   cyan = match only   white = both (agreement)",
           8, 16)

    # Score bar chart
    bar_h = 20
    bar_gap = 4
    chart_pad = 14
    chart_h = chart_pad * 2 + len(SUB_LABELS) * (bar_h + bar_gap)
    chart = np.full((chart_h, total_w, 3), 18, np.uint8)
    max_bar_w = total_w - 340
    for i, (key, desc) in enumerate(SUB_LABELS):
        val = sub.get(key, 0.0)
        y = chart_pad + i * (bar_h + bar_gap)
        _label(chart, desc, 8, y + bar_h - 6, 0.40, (210, 210, 210))
        bar_x0 = 240
        w_full = max(1, int(max_bar_w * min(max(val, 0.0), 1.0)))
        color = (60, 200, 60) if val >= 0.6 else (40, 170, 230) if val >= 0.35 else (90, 90, 200)
        cv2.rectangle(chart, (bar_x0, y + 2), (bar_x0 + max_bar_w, y + bar_h - 2), (50, 50, 50), -1)
        cv2.rectangle(chart, (bar_x0, y + 2), (bar_x0 + w_full, y + bar_h - 2), color, -1)
        _label(chart, f"{val:.2f}", bar_x0 + max_bar_w + 8, y + bar_h - 6, 0.40, (220, 220, 220))

    card = np.vstack([
        header, divider,
        label_strip, photo_row, divider,
        bin_label_strip, bin_row, divider,
        overlay_label_strip, overlay_canvas, divider,
        chart,
    ])
    return card

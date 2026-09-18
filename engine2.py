"""
engine.py — Shoe / boot sole similarity engine.

This is the core computer-vision matching logic, lifted from the original
"Sole Similarity Analyzer v12" batch script and generalized so it can score
any two shoe photos against each other (not just a fixed dataset with known
ground truth). Nothing here talks to the filesystem, a GUI, or Excel — it's
pure "given an image, extract features" / "given two feature sets, score
similarity".

Pipeline per image:
  1. Resize onto a fixed square canvas (scale-invariant).
  2. Find the boot/shoe outline (find_boot_contour).
  3. Isolate the tread/outsole region within that outline, using texture
     energy + edge density + a bottom-of-shoe position bias
     (extract_tread_region) — this is what keeps the comparison from being
     thrown off by the upper (laces, leather, fabric).
  4. Crop to the tread, rotate it to a canonical orientation via PCA
     (align_to_canonical), and resize to a fixed NORM_H x NORM_W patch.
  5. From that normalized patch, derive several color-invariant
     representations (adaptive-threshold binary tread, Canny edge pattern,
     CLAHE-enhanced grayscale) plus vector descriptors (ORB keypoints,
     zone density grid, edge-orientation histogram, Hu moments, Fourier
     shape descriptor) — each captures a different aspect of the tread
     pattern, so no single noisy metric can dominate.

Similarity between two images is a weighted blend of 9 sub-scores, computed
in both normal and left-right mirrored orientation (soles are often
photographed from either foot), taking whichever orientation scores higher.

Thresholds (carried over from the original, tuned on a labeled dataset):
  score >= IDENTICAL_THRESH  -> "identical" tread pattern
  score >= LIKELY_THRESH     -> "likely same mold" / worth a human look
  below LIKELY_THRESH        -> no meaningful match
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────
#  Tunables
# ─────────────────────────────────────────────────────────────────────────

IDENTICAL_THRESH = 0.72   # >= this: treated as the same sole/mold
LIKELY_THRESH = 0.58      # >= this (but below IDENTICAL): "likely" match

TARGET = 400               # working canvas size (square) before tread isolation
NORM_H, NORM_W = 256, 128  # normalized tread patch size after alignment

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────────────────────────────────
#  Boot outline + tread-only mask
# ─────────────────────────────────────────────────────────────────────────

def find_boot_contour(gray, img_bgr):
    """Find the full shoe/boot outline against its background."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 15, 60)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    filled = cv2.dilate(closed, k, iterations=2)
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < h * w * 0.08 or area > h * w * 0.95:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        score = area
        if abs((x + cw / 2) / w - 0.5) < 0.4 and abs((y + ch / 2) / h - 0.5) < 0.4:
            score *= 1.0
        else:
            score *= 0.3
        if score > best_score:
            best_score = score
            best = c
    if best is not None:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [best], -1, 255, -1)
        return mask
    # Fallback: HSV distance from border/background color
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bw = 15
    border_pixels = np.concatenate([
        hsv[:bw, :, :].reshape(-1, 3), hsv[-bw:, :, :].reshape(-1, 3),
        hsv[:, :bw, :].reshape(-1, 3), hsv[:, -bw:, :].reshape(-1, 3)
    ])
    bg_v = float(np.median(border_pixels[:, 2]))
    bg_s = float(np.median(border_pixels[:, 1]))
    vd = np.abs(hsv[:, :, 2].astype(np.float32) - bg_v)
    sd = np.abs(hsv[:, :, 1].astype(np.float32) - bg_s)
    fg = ((vd + sd * 0.5) > 25).astype(np.uint8) * 255
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n_lab > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == largest).astype(np.uint8) * 255
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.04):h - int(h * 0.04), int(w * 0.04):w - int(w * 0.04)] = 255
    return mask


def extract_tread_region(gray, img_bgr, boot_mask):
    """
    From the full boot mask, extract ONLY the tread/outsole region.

    The tread has high-frequency texture (lug pattern); the upper is
    smoother (leather/fabric/lacing). Combine local texture energy with
    edge density, biased toward the bottom of the shoe, then Otsu-threshold
    within the boot mask.
    """
    h, w = gray.shape

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    lap = cv2.Laplacian(enhanced, cv2.CV_32F, ksize=3)
    lap_sq = lap * lap
    win = 21
    local_energy = cv2.blur(lap_sq, (win, win))
    local_energy = cv2.bitwise_and(local_energy.astype(np.float32),
                                    local_energy.astype(np.float32),
                                    mask=boot_mask)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(blur, 25, 80)
    edges = cv2.bitwise_and(edges, boot_mask)
    edge_density = cv2.blur(edges.astype(np.float32), (31, 31))

    e_max = local_energy.max()
    if e_max > 0:
        local_energy /= e_max
    d_max = edge_density.max()
    if d_max > 0:
        edge_density /= d_max

    texture_map = (local_energy * 0.5 + edge_density * 0.5)
    texture_map = cv2.bitwise_and(texture_map, texture_map, mask=boot_mask)

    ys_boot, xs_boot = np.where(boot_mask > 0)
    if len(ys_boot) < 100:
        return boot_mask  # fallback

    y_top, y_bot = ys_boot.min(), ys_boot.max()
    boot_h = y_bot - y_top

    vert_weight = np.zeros((h, w), dtype=np.float32)
    for row in range(y_top, y_bot + 1):
        frac = (row - y_top) / max(boot_h, 1)
        vert_weight[row, :] = 0.3 + 0.7 * frac

    weighted_texture = texture_map * vert_weight

    wt_vals = weighted_texture[boot_mask > 0]
    if len(wt_vals) < 100:
        return boot_mask

    wt_norm = (weighted_texture * 255).astype(np.uint8)
    wt_norm = cv2.bitwise_and(wt_norm, boot_mask)
    _, tread_mask = cv2.threshold(wt_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tread_mask = cv2.morphologyEx(tread_mask, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    tread_mask = cv2.morphologyEx(tread_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(tread_mask)
    if n_lab > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        tread_mask = (labels == largest).astype(np.uint8) * 255

    tread_area = np.sum(tread_mask > 0)
    boot_area = np.sum(boot_mask > 0)
    if tread_area < boot_area * 0.15:
        cutoff_y = y_top + int(boot_h * 0.45)
        tread_mask = boot_mask.copy()
        tread_mask[:cutoff_y, :] = 0
        tread_mask = cv2.erode(tread_mask,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)

    return tread_mask


# ─────────────────────────────────────────────────────────────────────────
#  Canonical orientation
# ─────────────────────────────────────────────────────────────────────────

def align_to_canonical(patch, patch_mask=None):
    """Rotate the sole patch so its long axis is vertical, heavier end down."""
    h, w = patch.shape[:2]
    if patch_mask is None:
        patch_mask = (patch > 10).astype(np.uint8) * 255 if len(patch.shape) == 2 else \
            (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 10).astype(np.uint8) * 255

    ys, xs = np.where(patch_mask > 0)
    if len(xs) < 50:
        return patch

    coords = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float32)
    cov = np.cov(coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    principal = eigenvectors[:, np.argmax(eigenvalues)]
    angle = np.degrees(np.arctan2(principal[1], principal[0]))

    rot_angle = 90 - angle
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot_angle, 1.0)

    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(patch, M, (new_w, new_h))
    rot_mask = cv2.warpAffine(patch_mask, M, (new_w, new_h))

    rh = rotated.shape[0]
    top_mass = np.sum(rot_mask[:rh // 2, :] > 0)
    bot_mass = np.sum(rot_mask[rh // 2:, :] > 0)
    if top_mass > bot_mass * 1.2:
        rotated = cv2.rotate(rotated, cv2.ROTATE_180)
        rot_mask = cv2.rotate(rot_mask, cv2.ROTATE_180)

    ys2, xs2 = np.where(rot_mask > 0)
    if len(xs2) < 10:
        return patch
    y0, y1 = ys2.min(), ys2.max()
    x0, x1 = xs2.min(), xs2.max()
    margin = 5
    y0 = max(0, y0 - margin)
    y1 = min(rotated.shape[0], y1 + margin)
    x0 = max(0, x0 - margin)
    x1 = min(rotated.shape[1], x1 + margin)

    return rotated[y0:y1 + 1, x0:x1 + 1]


# ─────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────

def extract_features(img_path):
    """
    Run the full pipeline on one image file and return a feature dict,
    or None if the image couldn't be loaded / no usable tread was found.
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    return extract_features_from_array(img, name=img_path.name, stem=img_path.stem, path=img_path)


def extract_features_from_array(img, name="query", stem="query", path=None):
    """Same pipeline as extract_features but starting from an in-memory BGR image."""
    if img is None:
        return None
    h0, w0 = img.shape[:2]
    if h0 == 0 or w0 == 0:
        return None
    scale = TARGET / max(h0, w0)
    img = cv2.resize(img, (max(1, int(w0 * scale)), max(1, int(h0 * scale))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_sq = np.zeros((TARGET, TARGET, 3), dtype=np.uint8)
    gray_sq = np.zeros((TARGET, TARGET), dtype=np.uint8)
    img_sq[:h, :w] = img
    gray_sq[:h, :w] = gray
    gray = gray_sq
    img = img_sq
    h = w = TARGET

    boot_mask = find_boot_contour(gray, img)
    tread_mask = extract_tread_region(gray, img, boot_mask)

    ys, xs = np.where(tread_mask > 0)
    if len(xs) < 100:
        ys, xs = np.where(boot_mask > 0)
        if len(xs) < 100:
            return None
        tread_mask = boot_mask

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    tread_crop_gray = gray[y0:y1 + 1, x0:x1 + 1].copy()
    tread_crop_mask = tread_mask[y0:y1 + 1, x0:x1 + 1]
    tread_crop_gray[tread_crop_mask == 0] = 0

    aligned_gray = align_to_canonical(tread_crop_gray, tread_crop_mask)
    norm_gray = cv2.resize(aligned_gray, (NORM_W, NORM_H))

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    norm_enhanced = clahe.apply(norm_gray)

    blur = cv2.GaussianBlur(norm_enhanced, (5, 5), 0)
    tread_bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 25, 3)
    tread_bw = cv2.morphologyEx(tread_bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    edge_pat = cv2.Canny(cv2.GaussianBlur(norm_enhanced, (3, 3), 0), 30, 90)
    edge_pat = cv2.dilate(edge_pat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1)

    tread_bw_flip = cv2.flip(tread_bw, 1)
    edge_pat_flip = cv2.flip(edge_pat, 1)
    norm_gray_flip = cv2.flip(norm_enhanced, 1)

    orb = cv2.ORB_create(nfeatures=500, edgeThreshold=8, patchSize=16, fastThreshold=5)
    kp_e, desc_e = orb.detectAndCompute(edge_pat, None)
    kp_ef, desc_ef = orb.detectAndCompute(edge_pat_flip, None)
    kp_t, desc_t = orb.detectAndCompute(tread_bw, None)
    kp_tf, desc_tf = orb.detectAndCompute(tread_bw_flip, None)

    n_vz, n_hz = 4, 2
    zh, zw = NORM_H // n_vz, NORM_W // n_hz
    zone_density = np.zeros(n_vz * n_hz, dtype=np.float32)
    for zi in range(n_vz):
        for zj in range(n_hz):
            r0, r1 = zi * zh, (zi + 1) * zh
            c0, c1 = zj * zw, (zj + 1) * zw
            zone = tread_bw[r0:r1, c0:c1]
            zone_density[zi * n_hz + zj] = np.mean(zone > 0)

    gx = cv2.Sobel(norm_enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_enhanced, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx) * 180 / np.pi % 180
    n_bins = 18
    eoh = np.zeros(n_bins, dtype=np.float32)
    bin_width = 180.0 / n_bins
    for i in range(n_bins):
        in_bin = (ori >= i * bin_width) & (ori < (i + 1) * bin_width) & (mag > 10)
        eoh[i] = np.sum(mag[in_bin])
    if eoh.sum() > 0:
        eoh /= eoh.sum()

    hu = cv2.HuMoments(cv2.moments(tread_bw)).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    cnts_bw, _ = cv2.findContours(tread_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fourier_desc = np.zeros(20, dtype=np.float32)
    if cnts_bw:
        cnts_sorted = sorted(cnts_bw, key=cv2.contourArea, reverse=True)[:10]
        all_points = np.vstack(cnts_sorted) if len(cnts_sorted) > 0 else cnts_sorted[0]
        pts = all_points.reshape(-1, 2).astype(np.float32)
        if len(pts) > 20:
            indices = np.linspace(0, len(pts) - 1, 64).astype(int)
            pts_resampled = pts[indices]
            z = pts_resampled[:, 0] + 1j * pts_resampled[:, 1]
            fft = np.fft.fft(z)
            fd = np.abs(fft[1:21])
            if fd[0] > 0:
                fd /= fd[0]
            fourier_desc[:len(fd)] = fd.astype(np.float32)

    edges_full = cv2.Canny(cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0), 35, 110)
    edges_full = cv2.bitwise_and(edges_full, tread_mask)

    return dict(
        path=path, name=name, stem=stem,
        norm_gray=norm_enhanced, norm_gray_flip=norm_gray_flip,
        tread_bw=tread_bw, tread_bw_flip=tread_bw_flip,
        edge_pat=edge_pat, edge_pat_flip=edge_pat_flip,
        desc_e=desc_e, desc_ef=desc_ef,
        desc_t=desc_t, desc_tf=desc_tf,
        zone_density=zone_density, eoh=eoh, hu_log=hu_log,
        fourier_desc=fourier_desc,
        img_sq=img, boot_mask=boot_mask, tread_mask=tread_mask,
        edges=edges_full,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Similarity
# ─────────────────────────────────────────────────────────────────────────

def orb_match_score(desc_a, desc_b):
    if desc_a is None or desc_b is None:
        return 0.0
    if len(desc_a) < 5 or len(desc_b) < 5:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0.0
    good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.78 * m[1].distance)
    return good / max(len(desc_a), 1)


def phase_corr_align_and_compare(a, b):
    """Phase-correlation translation alignment, then NCC on the aligned pair."""
    af = a.astype(np.float32)
    bf = b.astype(np.float32)

    shift, response = cv2.phaseCorrelate(af, bf)
    dx, dy = int(round(shift[0])), int(round(shift[1]))

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = b.shape
    b_shifted = cv2.warpAffine(bf, M, (w, h))

    an = af - af.mean()
    bn = b_shifted - b_shifted.mean()
    denom = max(np.std(an) * np.std(bn) * an.size, 1e-9)
    ncc = float(np.sum(an * bn) / denom)

    return max(0.0, (ncc + 1) / 2), response


def ssim_windowed(a, b, win_size=11):
    """Mean SSIM over sliding windows."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    af = a.astype(np.float64)
    bf = b.astype(np.float64)

    k = cv2.getGaussianKernel(win_size, 1.5)
    window = k @ k.T

    mu_a = cv2.filter2D(af, -1, window)
    mu_b = cv2.filter2D(bf, -1, window)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.filter2D(af ** 2, -1, window) - mu_a_sq
    sigma_b_sq = cv2.filter2D(bf ** 2, -1, window) - mu_b_sq
    sigma_ab = cv2.filter2D(af * bf, -1, window) - mu_ab

    ssim_map = ((2 * mu_ab + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2))

    return float(np.mean(ssim_map))


def pixel_iou(bw_a, bw_b):
    a = (bw_a > 0).astype(np.uint8)
    b = (bw_b > 0).astype(np.uint8)
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return float(inter / union) if union > 0 else 0.0


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(np.clip(np.dot(a, b) / (na * nb), 0, 1)) if na > 1e-9 and nb > 1e-9 else 0.0


def hist_corr(a, b):
    return float(max(0.0, cv2.compareHist(
        a.reshape(-1, 1).astype(np.float32),
        b.reshape(-1, 1).astype(np.float32), cv2.HISTCMP_CORREL)))


def edge_overlay_score(edge_a, edge_b):
    """
    Overlap ratio of two edge maps (IoU): white/(white+yellow+cyan) in the
    cyan/yellow/white comparison visualization. Shown for the user, not
    used in the final score (too sensitive to minor scale/angle noise).
    """
    af = edge_a.astype(np.float32)
    bf = edge_b.astype(np.float32)
    if np.std(af) < 1e-6 or np.std(bf) < 1e-6:
        return 0.0
    shift, _ = cv2.phaseCorrelate(af, bf)
    dx = int(round(np.clip(shift[0], -NORM_W // 5, NORM_W // 5)))
    dy = int(round(np.clip(shift[1], -NORM_H // 5, NORM_H // 5)))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    b_shifted = cv2.warpAffine(edge_b, M, (edge_b.shape[1], edge_b.shape[0]))
    a_bin = edge_a > 0
    b_bin = b_shifted > 0
    both = np.sum(a_bin & b_bin)
    union = np.sum(a_bin | b_bin)
    return float(both / union) if union > 0 else 0.0


def make_overlay_image(edge_a, edge_b):
    """Cyan/yellow/white overlay image (BGR, NORM_H x NORM_W) for display."""
    af = edge_a.astype(np.float32)
    bf = edge_b.astype(np.float32)
    if np.std(af) > 1e-6 and np.std(bf) > 1e-6:
        shift, _ = cv2.phaseCorrelate(af, bf)
        dx = int(round(np.clip(shift[0], -NORM_W // 5, NORM_W // 5)))
        dy = int(round(np.clip(shift[1], -NORM_H // 5, NORM_H // 5)))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        b_shifted = cv2.warpAffine(edge_b, M, (edge_b.shape[1], edge_b.shape[0]))
    else:
        b_shifted = edge_b
    a_bin = edge_a > 0
    b_bin = b_shifted > 0
    canvas = np.zeros((NORM_H, NORM_W, 3), np.uint8)
    canvas[a_bin & ~b_bin] = (0, 200, 220)     # yellow — A only
    canvas[b_bin & ~a_bin] = (220, 200, 0)     # cyan   — B only
    canvas[a_bin & b_bin] = (220, 220, 220)    # white  — both = match
    return canvas


def compute_similarity(fa, fb):
    """
    Multi-strategy similarity between two feature dicts (as returned by
    extract_features), testing both original and mirrored orientation.
    Returns (score 0..1, sub_scores dict for the winning orientation).
    """
    results = []
    for flip_label, tbw_b, epa_b, ng_b, de_b, dt_b in [
        ("orig", fb["tread_bw"], fb["edge_pat"], fb["norm_gray"],
         fb["desc_e"], fb["desc_t"]),
        ("flip", fb["tread_bw_flip"], fb["edge_pat_flip"], fb["norm_gray_flip"],
         fb["desc_ef"], fb["desc_tf"]),
    ]:
        pc_edge, pc_resp = phase_corr_align_and_compare(
            fa["edge_pat"].astype(np.float32), epa_b.astype(np.float32))

        pc_tread, _ = phase_corr_align_and_compare(
            fa["tread_bw"].astype(np.float32), tbw_b.astype(np.float32))

        ssim_val = ssim_windowed(fa["norm_gray"], ng_b)
        ssim_val = max(0.0, (ssim_val + 1) / 2)

        orb_e = orb_match_score(fa["desc_e"], de_b)
        orb_t = orb_match_score(fa["desc_t"], dt_b)
        s_orb = max(orb_e, orb_t)
        s_orb_norm = min(1.0, s_orb / 0.10)

        s_iou = pixel_iou(fa["tread_bw"], tbw_b)

        zd_corr = float(np.corrcoef(fa["zone_density"], fb["zone_density"])[0, 1])
        s_zone = max(0.0, zd_corr if not np.isnan(zd_corr) else 0.0)

        s_eoh = hist_corr(fa["eoh"], fb["eoh"])

        s_hu = 1.0 - min(1.0, np.sum(np.abs(fa["hu_log"] - fb["hu_log"])) / 40.0)

        s_fd = cosine_sim(fa["fourier_desc"], fb["fourier_desc"])

        s_eo = edge_overlay_score(fa["edge_pat"], epa_b)

        score = (0.20 * pc_edge +
                 0.15 * pc_tread +
                 0.15 * ssim_val +
                 0.12 * s_orb_norm +
                 0.10 * s_iou +
                 0.08 * s_zone +
                 0.08 * s_eoh +
                 0.06 * s_hu +
                 0.06 * s_fd)

        sub = dict(pc_e=round(pc_edge, 3), pc_t=round(pc_tread, 3),
                   ssim=round(ssim_val, 3), orb=round(s_orb_norm, 3),
                   iou=round(s_iou, 3), zone=round(s_zone, 3),
                   eoh=round(s_eoh, 3), hu=round(s_hu, 3), fd=round(s_fd, 3),
                   eo=round(s_eo, 3), flip=flip_label)
        results.append((score, sub))

    best_score, best_sub = max(results, key=lambda x: x[0])
    return round(float(best_score), 4), best_sub


def label_for_score(score):
    if score >= IDENTICAL_THRESH:
        return "Identical"
    if score >= LIKELY_THRESH:
        return "Likely match"
    return "No match"

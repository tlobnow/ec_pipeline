#!/usr/bin/env python
"""
Step 06 — find GFP puncta and filaments, cell by cell and frame by frame.

Outputs (in <pos>/06_gfp/):
    filaments.tif           8-bit movie: the filament structures themselves
    puncta.tif              8-bit movie: the punctum spots
    filament_cells.tif      8-bit movie: outlines of cells that contain a
                            filament, to lay over the image like outlines.tif
    gfp_structures.csv      one row per cell per frame, calls and raw numbers
    gfp_summary.csv         per frame: how many cells in each state
    calibration.png         worked examples, showing why each call was made
    thresholds.png          where the thresholds sit in the data
    filament_montage.pdf    whole field at each timepoint, filaments picked out
    open_filaments.ijm      drag onto Fiji to view the layers together
    meta.json

How a structure is found
------------------------
Each cell is compared against ITSELF, not against a stretched version of
itself. The frame's background is subtracted, the cell's median is taken as
its diffuse pool, and every pixel is expressed as a fraction above that pool.
A pixel joins a structure when it exceeds GFP_STRUCT_MIN_CONTRAST, with a
floor at a few times the cell's own noise.

That distinction matters: rescaling each cell to its own 2nd-98th percentile,
as an earlier version did, stretches the noise of an evenly-glowing cell
across the full range and finds "puncta" in every single cell. Here an even
cell produces nothing at all.

Each connected structure is then judged on its own geometry. Long, thin and
elongated makes a filament; anything else contributes puncta. Lengths are in
microns, taken from the nd2, so the thresholds do not drift as cells grow.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    imread as safe_imread,
    Timer,
    base_parser,
    die,
    env_float_list,
    get_logger,
    load_cell_names,
    load_config,
    load_masks,
    read_units,
    require,
    resolve_timepoints,
    run_safely,
    step_dir,
    write_meta,
)

FEATURES = [
    "diffuse_level",
    "signal_over_bg",
    "bg_sigma",
    "thr_counts",
    "ridge_area_frac",
    "cell_noise_rel",
    "rel_p95",
    "struct_area_frac",
    "n_puncta",
    "punct_max_contrast",
    "fil_length_um",
    "fil_width_um",
    "fil_aspect",
    "fil_contrast",
    "fil_branches",
    "fil_peak_ratio",
    "punct_prominence",
    "gfp_area_frac",
    "shape_circularity",
    "shape_elongation",
    "shape_solidity",
    "punct_axial_pos_max",
    "punct_axial_pos_mean",
    "n_puncta_polar",
    "n_filaments",
]


def pixel_noise(plane, background):
    """
    Camera noise, from the difference between neighbouring background pixels.

    The spread of the background as a whole is NOT noise here: the space
    between cells is full of out-of-focus haze from their neighbours, which
    in this data measures 450-2200 counts and would set the bar several times
    too high. Haze is smooth, so it cancels between adjacent pixels while
    noise does not.
    """
    a = plane.astype(float)
    d = np.diff(a, axis=1)
    valid = background[:, 1:] & background[:, :-1]
    d = d[valid]
    if d.size < 50:
        d = np.diff(a, axis=1).ravel()  # crowded frame, use everything
    if d.size == 0:
        return 1.0
    return max(1.4826 * float(np.median(np.abs(d))) / np.sqrt(2.0), 1.0)


def ridge_mask(excess, cell_mask, thr_abs, bg_sigma, p):
    """
    Find line-like structure regardless of how bright it is.

    Intensity thresholding can only see a filament that stands far enough
    above the cell's diffuse pool. A faint filament never gets there at any
    threshold that is not also picking up noise. A ridge filter asks a
    different question — is this pixel part of a line? — so a faint filament
    is found by its shape, and a smooth bright region is not.
    """
    from skimage.filters import sato

    img = np.where(cell_mask, np.maximum(excess, 0.0), 0.0)
    resp = sato(img, sigmas=p["ridge_sigmas"], black_ridges=False, mode="constant")
    vals = resp[cell_mask]
    if vals.size < 4:
        return np.zeros(cell_mask.shape, bool)
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med)))
    thr = med + p["ridge_k"] * max(mad, 1e-9)
    # Still has to be above the noise, so a ridge cannot be built out of it.
    return (resp > thr) & (excess > 2.0 * bg_sigma) & cell_mask


def shape_descriptors(comp, area, perimeter, L_px, W_px):
    """
    Scale-free descriptors of a structure's shape.

    These need no micron thresholds, so they do not have to be re-tuned when
    the magnification, the cell size or the expression level changes.

        circularity  4*pi*A / P^2.  A round focus approaches 1, a line goes
                     towards 0. This is the single most useful separator of
                     a punctum from a filament.
        elongation   skeleton length / sqrt(area).  A compact blob sits near
                     1, a filament climbs with how long and thin it is.
        extent       fraction of the bounding box that is filled. A diagonal
                     line fills very little of its box.
    """
    circ = (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0.0
    elong = L_px / max(np.sqrt(max(area, 1.0)), 1e-9)
    return {"circularity": float(min(circ, 1.5)), "elongation": float(elong)}


def axial_axis(cell_mask):
    """
    The cell's long axis, taken straight from the mask.

    Derived by principal components of the mask pixels rather than from
    regionprops' orientation angle, whose sign convention is easy to get
    backwards — and a silently reversed axis would put every focus at
    mid-cell.  Returns (centroid, unit vector, half length).
    """
    coords = np.argwhere(cell_mask).astype(float)
    if len(coords) < 3:
        return None
    centre = coords.mean(axis=0)
    centred = coords - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]  # (dy, dx), unit length
    proj = centred @ axis
    half = float(np.abs(proj).max())
    if half <= 0:
        return None
    return centre, axis, half


def axial_position(cell_mask, y, x, cached=None):
    """
    Where a focus sits along the cell: 0 at mid-cell, 1 at a pole.

    Polar localisation is a real and common phenotype, so this is recorded as
    a measurement rather than treated as an artefact to remove. It also lets
    a polar focus be distinguished from a mid-cell one downstream, which is a
    biological question, not a detection problem.
    """
    got = cached if cached is not None else axial_axis(cell_mask)
    if got is None:
        return 0.0
    centre, axis, half = got
    proj = float(np.dot(np.array([y, x], dtype=float) - centre, axis))
    return float(min(abs(proj) / half, 1.5))


def prominence(excess, cell_mask, y, x, diffuse, r_in=3, r_out=7):
    """
    How far a peak stands above its own immediate surroundings.

    Being above the cell's overall threshold is not enough: a cell with
    uneven brightness has broad rises that clear any global bar without
    being foci. A focus is locally prominent — bright against the ring of
    cell just outside it — and a broad rise is not.
    """
    h, w = cell_mask.shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - y) ** 2 + (xx - x) ** 2
    ring = (d2 >= r_in**2) & (d2 <= r_out**2) & cell_mask
    if ring.sum() < 8:
        return float("inf")  # too near the cell end to judge
    return float(excess[y, x] - np.median(excess[ring])) / max(diffuse, 1.0)


def filament_paths(fil_mask, px_um):
    """
    The exact line whose length was reported, as a mask plus its length.

    skeleton_metrics measures the arc length along a structure's skeleton, but
    only returns the number. Drawing that same skeleton makes the measurement
    checkable: a number can only be compared against a line drawn by hand,
    whereas the path shows WHERE the algorithm went — through a branch, around
    a bend, or off along a neighbouring structure.
    """
    from skimage import measure as _m
    from skimage.morphology import skeletonize

    out = np.zeros(fil_mask.shape, bool)
    lengths = []
    if not fil_mask.any():
        return out, lengths
    lab = _m.label(fil_mask)
    for r in _m.regionprops(lab):
        comp = lab == r.label
        sk = skeletonize(comp)
        out |= sk
        L_px, _, _ = skeleton_metrics(comp)
        lengths.append(float(L_px) * px_um)
    return out, lengths


def skeleton_metrics(component):
    """
    Arc length and mean thickness of a structure, following its actual path.

    The second-moment major axis measures the straight line across a shape.
    For a bent or V-shaped filament that is far shorter than the filament
    itself, so the derived width (area / length) comes out too fat and a real
    filament is thrown away for being too thick. Walking the skeleton follows
    the curve, so a bent filament measures as long and thin as a straight one.
    """
    from skimage.morphology import skeletonize

    skel = skeletonize(component)
    coords = np.argwhere(skel)
    if len(coords) == 0:
        return 0.0, float(component.sum()), 0
    if len(coords) == 1:
        return 1.0, float(component.sum()), 0

    present = set(map(tuple, coords))
    length, branch = 0.0, 0
    for y, x in present:
        n = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and (y + dy, x + dx) in present:
                    n += 1
        if n > 2:
            branch += 1
        # count each edge once, weighting diagonals
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (y + dy, x + dx) in present:
                length += 1.0 if (dy == 0 or dx == 0) else 1.41421356
    length = max(length, 1.0)
    return length, float(component.sum()) / length, branch


def _axes(r):
    """regionprops axis lengths, tolerant of the skimage 0.19 -> 2.0 rename."""
    try:
        return float(r.axis_major_length), float(r.axis_minor_length)
    except AttributeError:
        return float(r.major_axis_length), float(r.minor_axis_length)


# ── the detector ────────────────────────────────────────────────────────────
def measure_cell(crop, cell_mask, bg_level, bg_sigma, px_um, p):
    """
    Find the structures in one cell in one frame.

    Returns (features, filament mask, punctum coordinates, rel image), where
    rel is the per-pixel excess over the cell's diffuse pool. The masks come
    back so they can be painted into the movie-sized stacks.
    """
    from skimage import measure
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian

    out = {k: 0.0 for k in FEATURES}
    out["n_puncta"] = 0
    out["n_filaments"] = 0
    empty = np.zeros(cell_mask.shape, bool)

    # Optionally ignore a rim of pixels just inside the outline. A bright
    # neighbour bleeds across the boundary, and that halo lands exactly at
    # the poles, where it is easily mistaken for a focus of this cell.
    if p.get("erode_px", 0) > 0:
        from skimage.morphology import binary_erosion, disk

        eroded = binary_erosion(cell_mask, disk(int(p["erode_px"])))
        if eroded.sum() >= p["min_area_px"] * 2:
            cell_mask = eroded

    pix = crop[cell_mask].astype(float) - bg_level
    if pix.size < 2 * p["min_area_px"]:
        return out, empty, [], None, None

    diffuse = float(np.median(pix))
    if not np.isfinite(diffuse) or diffuse <= 0:
        return out, empty, [], None, None  # cell at or below background

    # The bar a pixel must clear, in raw counts above the cell's pool. It is
    # the larger of two conditions:
    #   * a fraction of the pool          — meaningful contrast in bright cells
    #   * a multiple of the camera noise  — measured OUTSIDE the cells
    #
    # The noise term deliberately comes from the background, not from inside
    # the cell. An in-cell spread counts the structure itself as noise: cells
    # here report 70-900% "noise", which pushes the bar to 1000-3000 counts
    # and throws away every filament except the very brightest.
    thr_abs = max(p["min_contrast"] * diffuse, p["noise_k"] * bg_sigma)

    excess = np.full(cell_mask.shape, -1.0, dtype=float)
    excess[cell_mask] = crop[cell_mask].astype(float) - bg_level - diffuse

    rel = np.full(cell_mask.shape, -1.0, dtype=float)
    rel[cell_mask] = excess[cell_mask] / max(diffuse, 1.0)

    out["diffuse_level"] = diffuse
    out["signal_over_bg"] = diffuse / max(bg_sigma, 1.0)
    # How much of the cell carries GFP above the background at all. This is
    # what separates an untagged cell from an evenly glowing one without
    # relying on the median alone.
    above = (crop[cell_mask].astype(float) - bg_level) > (p["noise_k"] * bg_sigma)
    out["gfp_area_frac"] = float(above.mean())
    out["bg_sigma"] = bg_sigma
    out["thr_counts"] = thr_abs
    out["cell_noise_rel"] = thr_abs / max(diffuse, 1.0)
    out["rel_p95"] = float(np.percentile(rel[cell_mask], 95))

    by_intensity = (excess > thr_abs) & cell_mask
    bright = by_intensity
    if p.get("use_ridge", True):
        rid = ridge_mask(excess, cell_mask, thr_abs, bg_sigma, p)
        out["ridge_area_frac"] = float(rid.sum()) / float(cell_mask.sum())
        bright = bright | rid
    if bright.sum() < p["min_area_px"]:
        return out, empty, [], rel, None

    # Close single-pixel breaks so one filament stays one object rather than
    # a string of fragments, none of which is long enough on its own.
    if p["close_gaps_px"] > 0:
        from skimage.morphology import disk

        try:
            from skimage.morphology import binary_closing
        except ImportError:  # skimage >= 0.28
            from skimage.morphology import closing as binary_closing
        bright = binary_closing(bright, disk(int(p["close_gaps_px"]))) & cell_mask

    lab = measure.label(bright)
    props = [
        r
        for r in measure.regionprops(lab, intensity_image=rel)
        if r.area >= p["min_area_px"]
    ]
    if not props:
        return out, empty, [], rel, None
    out["struct_area_frac"] = sum(r.area for r in props) / float(cell_mask.sum())

    # Peak finding works on the same absolute excess as the thresholding, so
    # the two cannot disagree about what counts as bright.
    smooth = gaussian(np.where(cell_mask, excess, 0.0), 0.6, preserve_range=True)
    fil_mask = np.zeros(cell_mask.shape, bool)
    punct_pts, punct_contrast, best = [], [], None
    axial_all = []
    cell_axis = axial_axis(cell_mask)

    for r in props:
        comp = lab == r.label
        # Did this survive on brightness, or only because the ridge filter
        # picked it up? The ridge filter is there to find lines; anything it
        # found that is not a line is discarded rather than counted as a
        # punctum, otherwise a faint noise ridge invents structure in a cell
        # that is simply glowing evenly.
        from_intensity = bool((comp & by_intensity).any())
        L_px, W_px, branches = skeleton_metrics(comp)
        W_px = max(W_px, 1.0)
        L_um, W_um = L_px * px_um, W_px * px_um
        aspect = L_px / W_px
        sh_pre = shape_descriptors(comp, r.area, r.perimeter, L_px, W_px)
        if p["fil_rule"] == "shape":
            # No micron thresholds: a filament is simply a shape that is far
            # from round and extended for its area.
            is_fil = (
                sh_pre["circularity"] <= p["fil_max_circularity"]
                and sh_pre["elongation"] >= p["fil_min_elongation"]
                and L_um >= p["fil_min_length_um"]
            )
        else:
            is_fil = (
                L_um >= p["fil_min_length_um"]
                and W_um <= p["fil_max_width_um"]
                and aspect >= p["fil_min_aspect"]
            )

        if not (is_fil or from_intensity):
            continue  # ridge-only and not a line: ignore

        sh = shape_descriptors(comp, r.area, r.perimeter, L_px, W_px)
        cand = dict(
            fil_length_um=L_um,
            fil_width_um=W_um,
            fil_aspect=aspect,
            fil_contrast=float(r.intensity_mean),
            fil_branches=branches,
            is_fil=is_fil,
            shape_circularity=sh["circularity"],
            shape_elongation=sh["elongation"],
            shape_solidity=float(r.solidity),
        )
        if best is None or (cand["is_fil"], L_um) > (
            best["is_fil"],
            best["fil_length_um"],
        ):
            best = cand

        if is_fil:
            fil_mask |= comp
            out["n_filaments"] += 1
            # A focus sitting on a filament is still a focus, but ordinary
            # brightness variation along a filament is not. Measured on known
            # cases, a plain filament runs at max/median 1.2-1.3 while a
            # filament carrying a focus reaches 2.0, so the peak has to stand
            # that far above the filament's OWN median to count.
            v = excess[comp]
            med = float(np.median(v))
            out["fil_peak_ratio"] = float(v.max() / med) if med > 0 else 0.0
            if med > 0 and out["fil_peak_ratio"] >= p["punct_on_filament_ratio"]:
                peaks = peak_local_max(
                    smooth,
                    min_distance=3,
                    threshold_abs=med * p["punct_on_filament_ratio"],
                    labels=comp,
                )
                for y, x in peaks[:3]:
                    punct_pts.append((int(y), int(x)))
                    punct_contrast.append(float(excess[y, x]))
            continue

        if r.area > p["punct_max_area_px"]:
            # one large focus, not a dozen noise maxima inside a bright blob
            ys, xs = np.nonzero(comp)
            cands = [(int(ys.mean()), int(xs.mean()))]
        else:
            peaks = peak_local_max(
                smooth, min_distance=2, threshold_abs=thr_abs, labels=comp
            )
            cap = int(np.ceil(r.area / p["punct_unit_area_px"]))
            if len(peaks) == 0:
                ys, xs = np.nonzero(comp)
                cands = [(int(ys.mean()), int(xs.mean()))]
            else:
                cands = [
                    (int(y), int(x)) for y, x in peaks[: max(1, min(len(peaks), cap))]
                ]

        kept = [
            (y, x)
            for y, x in cands
            if prominence(excess, cell_mask, y, x, diffuse) >= p["punct_min_prominence"]
        ]
        if kept:
            for y, x in kept:
                pos = axial_position(cell_mask, y, x, cached=cell_axis)
                out["punct_axial_pos_max"] = max(out["punct_axial_pos_max"], pos)
                axial_all.append(pos)
            best_prom = max(
                prominence(excess, cell_mask, y, x, diffuse) for y, x in kept
            )
            out["punct_prominence"] = max(out["punct_prominence"], best_prom)
            punct_pts.extend(kept)
            punct_contrast.append(float(r.intensity_max))

    out["n_puncta"] = len(punct_pts)
    if punct_contrast:
        out["punct_max_contrast"] = max(punct_contrast)
    if axial_all:
        out["punct_axial_pos_mean"] = float(np.mean(axial_all))
        out["n_puncta_polar"] = int(
            sum(1 for a in axial_all if a >= p["polar_threshold"])
        )
    if best is not None:
        for k in (
            "fil_length_um",
            "fil_width_um",
            "fil_aspect",
            "fil_contrast",
            "fil_branches",
            "shape_circularity",
            "shape_elongation",
            "shape_solidity",
        ):
            out[k] = float(best[k])
    return out, fil_mask, punct_pts, rel, bright


def smooth_states(rows, min_duration, log):
    """
    Stop a call flickering when a measurement sits right on a threshold.

    A structure measured at 0.45 um wide against a 0.45 um limit will cross
    back and forth on noise alone, so the same cell reads punctate, mixed,
    punctate from frame to frame. A new state has to hold for min_duration
    frames before it is adopted.
    """
    from collections import defaultdict

    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["track_id"]].append(r)

    changed = 0
    for track, series in by_cell.items():
        series.sort(key=lambda r: r["frame"])
        raw = [r["state_raw"] for r in series]
        n = len(raw)
        if n < min_duration:
            for r in series:
                r["state"] = r["state_raw"]
            continue

        settled = [raw[0]] * n
        current, run_state, run_start = raw[0], raw[0], 0
        for i in range(n):
            if raw[i] != run_state:
                run_state, run_start = raw[i], i
            if i - run_start + 1 >= min_duration:
                # The run has lasted long enough to be believed. Back-fill it,
                # so the frame the change really happened on is kept rather
                # than reported min_duration frames late.
                current = run_state
                for j in range(run_start, i + 1):
                    settled[j] = current
            elif settled[i] == raw[0] or i > 0:
                settled[i] = current

        for r, s in zip(series, settled):
            r["state"] = s
            if s != r["state_raw"]:
                changed += 1
    if changed:
        log.info(
            f"hysteresis settled {changed} of {len(rows)} cell-frames "
            f"({100 * changed / max(len(rows), 1):.1f}%) that were "
            f"flickering between states"
        )
    return changed


def report_borderline(rows, p, log):
    """Warn when many measurements sit within 10% of a threshold."""
    checks = [
        ("fil_length_um", p["fil_min_length_um"]),
        ("fil_width_um", p["fil_max_width_um"]),
        ("fil_aspect", p["fil_min_aspect"]),
    ]
    for key, thr in checks:
        vals = np.array([r[key] for r in rows if r[key] > 0])
        if vals.size == 0:
            continue
        near = np.abs(vals - thr) <= 0.1 * thr
        if near.mean() > 0.15:
            log.warning(
                f"{near.mean() * 100:.0f}% of measured structures sit within "
                f"10% of the {key} threshold ({thr:g}). The threshold is "
                f"cutting through a population rather than between two — "
                f"check thresholds.png before trusting these calls."
            )


MODULE_VERSION = "0.8.0"

ALL_STATES = ("none", "diffuse", "punctate", "filamentous", "mixed")

STATE_COLOURS = {
    "none": "#777777",
    "diffuse": "#aec7e8",
    "punctate": "#ff7f0e",
    "filamentous": "#2ca02c",
    "mixed": "#9467bd",
}


def state_of(row, min_signal=0.0, min_struct_frac=0.0):
    """
    Which phenotype this cell shows.

        none         no GFP above background — untagged, or not expressing
        diffuse      GFP present, spread evenly
        punctate     one or more foci
        filamentous  one or more line-like structures
        mixed        both

    'none' is separate from 'diffuse' on purpose. Both look featureless, but
    one means the reporter is absent and the other means it is present and
    unassembled, and merging them would make an untagged line look like an
    assembly-free one.
    """
    if row.get("signal_over_bg", 0.0) < min_signal:
        return "none"
    # A cell where the detected structure covers almost none of the cell is
    # not really structured, whatever was found in it. Hand-labelled data put
    # this at the top of the tree, above every shape measure.
    if row.get("struct_area_frac", 1.0) < min_struct_frac and not row["n_filaments"]:
        return "diffuse"
    if row["n_filaments"] and row["n_puncta"]:
        return "mixed"
    if row["n_filaments"]:
        return "filamentous"
    if row["n_puncta"]:
        return "punctate"
    return "diffuse"


# ── figures ─────────────────────────────────────────────────────────────────
def calibration_figure(examples, p, path, log):
    """
    Worked examples: for each cell, the raw crop, what counts as structure,
    and why each piece was or was not called a filament.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    if not examples:
        log.warning("calibration: no cells to show")
        return
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(13, 2.9 * n), squeeze=False)

    for i, ex in enumerate(examples):
        crop, rel, fil, pts, feat, title = (
            ex["crop"],
            ex["rel"],
            ex["fil"],
            ex["pts"],
            ex["feat"],
            ex["title"],
        )
        axes[i][0].imshow(crop, cmap="Greens")
        axes[i][0].set_title(f"{title}\nGFP", fontsize=8)

        show = np.ma.masked_where(rel < -0.5, rel)
        im = axes[i][1].imshow(
            show, cmap="inferno", vmin=0, vmax=max(1.0, float(np.nanmax(rel)))
        )
        axes[i][1].set_title(
            f"excess over the cell's own pool\n"
            f"threshold {feat['thr_counts']:.0f} counts "
            f"({ex['thr']:.2f}x pool)",
            fontsize=8,
        )
        plt.colorbar(im, ax=axes[i][1], fraction=0.046)

        rgb = np.zeros(crop.shape + (3,))
        g = crop.astype(float)
        g = (g - g.min()) / max(float(np.ptp(g)), 1)
        rgb[..., 1] = g * 0.5
        rgb[fil] = (1.0, 0.2, 1.0)  # filaments magenta
        axes[i][2].imshow(rgb)
        for y, x in pts:
            axes[i][2].plot(x, y, "o", mfc="none", mec="cyan", ms=9, mew=1.4)
        axes[i][2].set_title(
            f"filaments magenta, puncta cyan\n"
            f"{feat['n_filaments']} filament(s), "
            f"{feat['n_puncta']} punctum(s)",
            fontsize=8,
        )

        ax = axes[i][3]
        ax.axis("off")
        lines = [
            "longest structure",
            f"  length   {feat['fil_length_um']:.2f} um   "
            f"(needs >= {p['fil_min_length_um']})",
            f"  width    {feat['fil_width_um']:.2f} um   "
            f"(needs <= {p['fil_max_width_um']})",
            f"  aspect   {feat['fil_aspect']:.1f}       "
            f"(needs >= {p['fil_min_aspect']})",
            f"  contrast {feat['fil_contrast']:.2f}",
            f"  branches {int(feat['fil_branches'])}",
            "",
            f"diffuse pool   {feat['diffuse_level']:.0f} counts",
            f"background noise {feat['bg_sigma']:.0f} counts",
            f"threshold      {feat['thr_counts']:.0f} counts above pool",
            f"structure area {feat['struct_area_frac'] * 100:.1f} % of cell",
            "",
            f"CALL: {state_of(feat)}",
        ]
        ax.text(
            0.0,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            transform=ax.transAxes,
        )

        for a in axes[i][:3]:
            a.set_xticks([])
            a.set_yticks([])

    fig.suptitle(
        "GFP structure calibration — a cell is judged against its own "
        "diffuse pool, so an evenly glowing cell yields nothing",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"calibration -> {Path(path).name} ({n} examples)")


def threshold_figure(rows, p, path, log):
    """Where the thresholds sit in the data actually measured."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    arr = lambda k: np.array([r[k] for r in rows if r[k] > 0])

    for ax, key, thr, label, side in [
        (
            axes[0],
            "fil_length_um",
            p["fil_min_length_um"],
            "arc length along the structure (um)",
            "min",
        ),
        (axes[1], "fil_width_um", p["fil_max_width_um"], "mean thickness (um)", "max"),
        (axes[2], "fil_aspect", p["fil_min_aspect"], "arc length / thickness", "min"),
        (
            axes[3],
            "rel_p95",
            p["min_contrast"],
            "95th percentile excess over pool",
            "min",
        ),
    ]:
        v = arr(key)
        if v.size:
            ax.hist(v, bins=60, color="seagreen", edgecolor="white")
        ax.axvline(
            thr,
            color="red",
            ls="--",
            label=f"{'>=' if side == 'min' else '<='} {thr:g}",
        )
        ax.set_xlabel(label, fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cell-frames")
    fig.suptitle(
        "Thresholds against the measured distributions — a threshold "
        "sitting mid-peak is splitting one population, not two",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"thresholds -> {Path(path).name}")


def filament_montage(
    gfp,
    tracked,
    fil_stack,
    per_frame_cells,
    name_of,
    requested_min,
    dt_min,
    px_um,
    path,
    log,
):
    """Whole field at each timepoint, with the filaments picked out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.segmentation import find_boundaries

    T = gfp.shape[0]
    picks, skipped = [], []
    for minute in requested_min:
        f = int(round(minute / dt_min))
        (picks if 0 <= f < T else skipped).append((minute, f))
    if skipped:
        log.info(
            "montage: "
            + ", ".join(f"{m:g}" for m, _ in skipped)
            + " min are past the end of this movie, skipped"
        )
    if not picks:
        log.warning("montage: none of the requested timepoints exist here")
        return

    lo, hi = np.percentile(gfp[[f for _, f in picks]], [1, 99.5])
    ncol = min(3, len(picks))
    nrow = int(np.ceil(len(picks) / ncol))

    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(7 * ncol, 7.4 * nrow), squeeze=False
        )
        for k, (minute, f) in enumerate(picks):
            ax = axes[k // ncol][k % ncol]
            ax.imshow(gfp[f], cmap="gray", vmin=lo, vmax=hi)

            ids = per_frame_cells.get(f, [])
            if ids:
                sel = np.isin(tracked[f], ids)
                edges = find_boundaries(np.where(sel, tracked[f], 0), mode="inner")
                rgba = np.zeros(edges.shape + (4,))
                rgba[edges] = (1.0, 0.85, 0.0, 1.0)  # yellow outlines
                ax.imshow(rgba, interpolation="nearest")

            fmask = fil_stack[f] > 0
            if fmask.any():
                rgba = np.zeros(fmask.shape + (4,))
                rgba[fmask] = (1.0, 0.15, 1.0, 1.0)  # magenta filaments
                ax.imshow(rgba, interpolation="nearest")

            ax.set_title(
                f"{minute:g} min  ·  frame {f}  ·  "
                f"{len(ids)} cell(s) with filaments",
                fontsize=11,
            )
            ax.axis("off")

            width_um = gfp.shape[2] * px_um
            bar_um = next((v for v in (20, 10, 5, 2, 1) if v <= 0.25 * width_um), 1)
            frac = bar_um / width_um
            ax.plot(
                [0.96 - frac, 0.96],
                [0.045, 0.045],
                transform=ax.transAxes,
                color="white",
                lw=3,
                solid_capstyle="butt",
            )
            ax.text(
                0.96 - frac / 2,
                0.055,
                f"{bar_um:g} um",
                transform=ax.transAxes,
                color="white",
                ha="center",
                va="bottom",
                fontsize=9,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

        for k in range(len(picks), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(
            "GFP filaments over time — magenta = filament "
            "structures, yellow = cells containing one",
            fontsize=13,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    log.info(f"filament montage -> {Path(path).name} ({len(picks)} panels)")


def timecourse_figure(summary, dt_min, path, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([s["frame"] for s in summary]) * dt_min
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, colour, label in [
        ("pct_filamentous", "#2ca02c", "filamentous"),
        ("pct_mixed", "#9467bd", "mixed"),
        ("pct_punctate", "#ff7f0e", "punctate"),
        ("pct_diffuse", "#aec7e8", "diffuse"),
    ]:
        ax.plot(t, [s[key] for s in summary], color=colour, lw=2, label=label)
    ax.set(
        xlabel="time (min)", ylabel="% of tracked cells", title="GFP state over time"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"timecourse -> {Path(path).name}")


def build_macro(paths, px_um, dt_min, n_frames) -> str:
    def ij(q):
        return str(q).replace("\\", "/")

    return f"""// =====================================================================
//  GFP structures as layers over the movie.
//  Drag onto Fiji and press Run.
//
//  Channels:  grey = brightfield, green = GFP, magenta = filaments,
//             cyan = cells containing a filament, yellow = puncta
//  Turn layers on and off with Image > Color > Channels Tool.
// =====================================================================

SATURATED = 0.35;

run("Close All");
print("\\\\Clear");
setBatchMode(true);

open("{ij(paths['bf'])}");        rename("bf");
open("{ij(paths['gfp'])}");       rename("gfp");
open("{ij(paths['fil'])}");       rename("filaments");
open("{ij(paths['cells'])}");     rename("filcells");
open("{ij(paths['puncta'])}");    rename("puncta");

prepare("bf");
prepare("gfp");

run("Merge Channels...", "c2=gfp c4=bf c6=filaments c5=filcells c7=puncta create");
rename("gfp structures");

run("Properties...", "channels=5 slices=1 frames={n_frames}"
    + " pixel_width={px_um} pixel_height={px_um} voxel_depth=1"
    + " frame=[{dt_min} min]");
Stack.setDisplayMode("composite");

run("Label...", "format=0 starting=0 interval={dt_min} x=5 y=20 font=14 "
    + "text=min range=1-{n_frames} use overlay");

setBatchMode(false);
Stack.setFrame(1);
print("Filament layer is magenta. Cells containing a filament are outlined "
    + "cyan. Use the Channels Tool to switch layers on and off.");

function prepare(title) {{
    selectWindow(title);
    if (bitDepth() == 8) return;
    n = nSlices;
    f = floor(n / 2) + 1;
    setSlice(f);
    run("Enhance Contrast", "saturated=" + SATURATED);
    getMinAndMax(lo, hi);
    if (hi <= lo) {{ resetMinAndMax(); getMinAndMax(lo, hi); }}
    setMinAndMax(lo, hi);
    setOption("ScaleConversions", true);
    run("8-bit");
    print(title + ": contrast " + lo + "-" + hi + " from frame " + f);
}}
"""


# ── main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = base_parser(__doc__.split("\n")[1]).parse_args(argv)
    cfg = load_config()
    log = get_logger("06_gfp", args.quiet)

    p = cfg["gfp"]
    pos = args.position

    load_dir = step_dir(cfg, pos, "01_load")
    track_dir = step_dir(cfg, pos, "03_track")
    gfp_path = require(
        load_dir / "gfp.tif",
        "the GFP channel",
        "Set LOAD_ND2=TRUE in config.sh and run again.",
    )
    out = step_dir(cfg, pos, "06_gfp", create=True)

    import tifffile
    from skimage import measure
    from skimage.segmentation import find_boundaries

    gfp = safe_imread(gfp_path)
    tracked, is_tracked, mask_file = load_masks(cfg, pos, log)
    if gfp.shape != tracked.shape:
        die(
            f"the GFP movie is {gfp.shape} but the masks are {tracked.shape}.",
            "These must match. Re-run the load and segmentation steps so both "
            "come from the same crop.",
        )
    T = gfp.shape[0]

    px_um, dt_min = read_units(load_dir, log, cfg)

    log.info(
        f"{px_um:.4f} um/px — a filament must be "
        f"{p['fil_min_length_um']} um, i.e. "
        f"{p['fil_min_length_um'] / px_um:.0f} px long and at most "
        f"{p['fil_max_width_um'] / px_um:.0f} px wide"
    )

    lineage_csv = track_dir / "lineage.csv"
    names, _ = load_cell_names(lineage_csv) if lineage_csv.exists() else ({}, {})
    name_of = lambda t: names.get(int(t), str(int(t)))

    fil_stack = np.zeros(gfp.shape, np.uint8)
    punct_stack = np.zeros(gfp.shape, np.uint8)
    cells_stack = np.zeros(gfp.shape, np.uint8)
    # Every pixel the detector counted as structure, before any judgement
    # about shape. This separates the two ways detection goes wrong: the
    # wrong pixels selected, or the right pixels classified wrongly.
    struct_stack = np.zeros(gfp.shape, np.uint8)

    rows, summary, examples_pool = [], [], []
    fil_cells_by_frame = {}

    with Timer(log, "detection"):
        for t in range(T):
            lbl = tracked[t]
            outside = gfp[t][lbl == 0].astype(float)
            bg = float(np.median(outside)) if outside.size else 0.0
            bg_sigma = pixel_noise(gfp[t], lbl == 0)

            fil_ids, counts = [], {s: 0 for s in ALL_STATES}
            for reg in measure.regionprops(lbl):
                y0, x0, y1, x1 = reg.bbox
                cm = lbl[y0:y1, x0:x1] == reg.label
                crop = gfp[t][y0:y1, x0:x1]

                feat, fmask, pts, rel, smask = measure_cell(
                    crop, cm, bg, bg_sigma, px_um, p
                )
                state = state_of(
                    feat, p["min_signal_over_bg"], p["min_struct_area_frac"]
                )
                counts[state] += 1

                if smask is not None and smask.any():
                    struct_stack[t, y0:y1, x0:x1][smask] = 255
                if fmask.any():
                    fil_stack[t, y0:y1, x0:x1][fmask] = 255
                for y, x in pts:
                    punct_stack[t, y0 + y, x0 + x] = 255
                if feat["n_filaments"]:
                    fil_ids.append(int(reg.label))

                rows.append(
                    {
                        "frame": t,
                        "time_min": round(t * dt_min, 2),
                        "track_id": int(reg.label),
                        "cell_name": name_of(reg.label),
                        "area_px": int(reg.area),
                        "gfp_bg": round(bg, 1),
                        "gfp_bg_sigma": round(bg_sigma, 1),
                        "state": state,
                        "state_raw": state,
                        **{k: round(float(feat[k]), 4) for k in FEATURES},
                    }
                )

                if rel is not None and (
                    feat["n_filaments"] or feat["fil_length_um"] > 0
                ):
                    examples_pool.append(
                        {
                            "crop": crop.copy(),
                            "rel": rel,
                            "fil": fmask,
                            "pts": pts,
                            "feat": feat,
                            "thr": feat["thr_counts"] / max(feat["diffuse_level"], 1.0),
                            "title": f"cell {name_of(reg.label)}, "
                            f"{t * dt_min:.0f} min",
                            "is_fil": bool(feat["n_filaments"]),
                            "len": feat["fil_length_um"],
                        }
                    )

            fil_cells_by_frame[t] = []

    # ── settle the flickering, then mark the cells ─────────────────────────
    if is_tracked:
        smooth_states(rows, int(p["state_min_duration"]), log)
    else:
        # Without tracking a cell cannot be followed between frames, so there
        # is no sequence to smooth. Each frame stands alone.
        for r in rows:
            r["state"] = r["state_raw"]
    report_borderline(rows, p, log)

    # Outlines are drawn for the settled call, so the movie agrees with the CSV.
    for r in rows:
        if r["state"] in ("filamentous", "mixed"):
            fil_cells_by_frame.setdefault(r["frame"], []).append(r["track_id"])
    for t in range(T):
        ids = fil_cells_by_frame.get(t, [])
        if ids:
            sel = np.isin(tracked[t], ids)
            cells_stack[t][
                find_boundaries(np.where(sel, tracked[t], 0), mode="inner")
            ] = 255

    # per-frame counts, recomputed from the settled calls
    summary = []
    for t in range(T):
        here = [r for r in rows if r["frame"] == t]
        n = max(len(here), 1)
        counts = {s: sum(1 for r in here if r["state"] == s) for s in ALL_STATES}
        summary.append(
            {
                "frame": t,
                "time_min": round(t * dt_min, 2),
                "n_cells": len(here),
                **{f"n_{k}": v for k, v in counts.items()},
                **{f"pct_{k}": round(100 * v / n, 2) for k, v in counts.items()},
            }
        )

    # ── save ───────────────────────────────────────────────────────────────
    # Dilate the punctum dots so they are visible at normal zoom.
    from scipy.ndimage import binary_dilation

    punct_stack = (
        binary_dilation(punct_stack > 0, np.ones((1, 3, 3), bool)) * 255
    ).astype(np.uint8)

    tifffile.imwrite(out / "structures.tif", struct_stack)
    tifffile.imwrite(out / "filaments.tif", fil_stack)
    tifffile.imwrite(out / "puncta.tif", punct_stack)
    tifffile.imwrite(out / "filament_cells.tif", cells_stack)

    import csv as _csv

    with open(out / "gfp_structures.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out / "gfp_summary.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    n_rows = len(rows)
    pct = lambda s: 100 * sum(1 for r in rows if r["state"] == s) / max(n_rows, 1)
    log.info(
        f"{n_rows} cell-frames:  " + "  ".join(f"{s} {pct(s):.1f}%" for s in ALL_STATES)
    )
    if pct("diffuse") + pct("none") < 2:
        log.warning(
            "almost no cell is called diffuse or none — if that is not "
            "real, raise GFP_STRUCT_MIN_CONTRAST in config.sh"
        )

    # ── figures ────────────────────────────────────────────────────────────
    n_ex = int(p["calibration_cells"])
    pos_ex = sorted([e for e in examples_pool if e["is_fil"]], key=lambda e: -e["len"])[
        : max(1, n_ex // 2)
    ]
    neg_ex = sorted(
        [e for e in examples_pool if not e["is_fil"]], key=lambda e: -e["len"]
    )[: max(1, n_ex - len(pos_ex))]
    for e in neg_ex:
        e["title"] += "  (not called)"
    calibration_figure(pos_ex + neg_ex, p, out / "calibration.png", log)
    threshold_figure(rows, p, out / "thresholds.png", log)
    timecourse_figure(summary, dt_min, out / "gfp_timecourse.png", log)

    requested = resolve_timepoints(
        os.environ.get("TIMEPOINTS_MIN", "0 30 45 60 90 120 150"), T, dt_min, log
    )
    filament_montage(
        gfp,
        tracked,
        fil_stack,
        fil_cells_by_frame,
        name_of,
        requested,
        dt_min,
        px_um,
        out / "filament_montage.pdf",
        log,
    )

    (out / "open_filaments.ijm").write_text(
        build_macro(
            {
                "bf": load_dir / "bf.tif",
                "gfp": gfp_path,
                "fil": out / "filaments.tif",
                "cells": out / "filament_cells.tif",
                "puncta": out / "puncta.tif",
            },
            px_um,
            dt_min,
            T,
        )
    )
    log.info(f"drag this onto Fiji:  {out / 'open_filaments.ijm'}")

    write_meta(
        out,
        "06_gfp",
        dict(p, position=pos),
        {"gfp": gfp_path, "tracked_masks": mask_file},
        {
            "n_cell_frames": n_rows,
            "pixel_size_um": px_um,
            "pct_diffuse": round(pct("diffuse"), 2),
            "pct_punctate": round(pct("punctate"), 2),
            "pct_filamentous": round(pct("filamentous"), 2),
            "pct_mixed": round(pct("mixed"), 2),
        },
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "gfp structures"))

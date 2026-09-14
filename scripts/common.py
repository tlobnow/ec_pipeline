"""
Shared helpers for the pipeline steps.

The config lives in config.sh. run_pipeline.sh sources it with 'set -a', which
makes every setting an environment variable, so these scripts read the same
config the coordinator used — nothing is passed on the command line except the
position number.

Every setting is read through env_str / env_int / env_float / env_bool /
env_int_list. If a setting is missing or malformed the script stops with a
message naming the setting, what was found, and what to write instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import numpy as np
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

MISSING = object()

# Bumped whenever the scripts and tools have to be updated together. The
# tools check it, so a half-updated folder fails with an explanation instead
# of a TypeError deep inside a detection loop — or worse, runs quietly with a
# stale detector.
PIPELINE_VERSION = "0.10.0"


# ── error reporting ─────────────────────────────────────────────────────────
def die(problem: str, fix: str = "") -> "None":
    """Stop with the same error layout the bash coordinator uses."""
    print("", flush=True)
    print("===========================================================")
    print("  STEP FAILED")
    print("===========================================================")
    print(f"  Problem : {problem}")
    if fix:
        print(f"  Fix     : {fix}")
    print("", flush=True)
    sys.exit(1)


def _raw(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        if default is MISSING:
            die(
                f"the setting {name} is missing from the config file.",
                f'Add a line to config.sh, for example  {name}="..."',
            )
        return None
    return value


# ── typed settings ──────────────────────────────────────────────────────────
def env_str(name, default=MISSING, choices=None):
    value = _raw(name, default)
    if value is None:
        return default
    if choices and value not in choices:
        die(
            f"{name} is set to '{value}', which is not allowed.",
            f"Use one of: {', '.join(choices)}",
        )
    return value


def env_int(name, default=MISSING):
    value = _raw(name, default)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        die(
            f"{name} should be a whole number but is set to '{value}'.",
            f"Edit config.sh, for example  {name}=5",
        )


def env_float(name, default=MISSING):
    value = _raw(name, default)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        die(
            f"{name} should be a number but is set to '{value}'.",
            f"Edit config.sh, for example  {name}=2.5",
        )


def env_bool(name, default=MISSING):
    """Accepts TRUE/FALSE (and a few obvious spellings), nothing else."""
    value = _raw(name, default)
    if value is None:
        return default
    v = str(value).strip().lower()
    if v in ("true", "yes", "1"):
        return True
    if v in ("false", "no", "0"):
        return False
    die(
        f"{name} is set to '{value}', which is not TRUE or FALSE.",
        f"Edit config.sh and write  {name}=TRUE  or  {name}=FALSE",
    )


def env_int_list(name, default=None):
    """A space-separated list of numbers, e.g. PREVIEW_FRAMES=\"0 30 60\"."""
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    out = []
    for part in value.replace(",", " ").split():
        try:
            out.append(int(part))
        except ValueError:
            die(
                f"{name} should be a list of whole numbers separated by "
                f"spaces, but contains '{part}'.",
                f'Edit config.sh, for example  {name}="0 30 60"',
            )
    return out


def env_str_list(name, default=None):
    """A space-separated list left as text, e.g. CELL_TRACK_IDS=\"41 41-2\"."""
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default) if default is not None else []
    return [p for p in value.replace(",", " ").split() if p]


def env_float_list(name, default=None):
    """A space-separated list of times, e.g. TIMEPOINTS_MIN=\"0 30 45\"."""
    value = os.environ.get(name, "").strip()
    if not value:
        return list(default) if default is not None else None
    out = []
    for part in value.replace(",", " ").split():
        try:
            out.append(float(part))
        except ValueError:
            die(
                f"{name} should be a list of times in minutes separated by "
                f"spaces, but contains '{part}'.",
                f'Edit config.sh, for example  {name}="0 30 60 120"',
            )
    return out


# ── the config, in one place ────────────────────────────────────────────────
def load_config() -> dict:
    """Read every setting once, so a bad value is caught before any work."""
    if "EXPERIMENT_NAME" not in os.environ:
        die(
            "the configuration was not found in the environment.",
            "Run the pipeline through ./run_pipeline.sh — these scripts read "
            "the config that the coordinator loads.",
        )
    return {
        "experiment": {
            "name": env_str("EXPERIMENT_NAME"),
            "nd2_path": env_str("ND2_PATH"),
            "output_root": env_str("OUTPUT_ROOT"),
            "condition": env_str("CONDITION", "unspecified"),
            "config_dir": env_str("CONFIG_DIR", "."),
        },
        "load": {
            # A number or a channel name from the file, e.g. "PHASE".
            "ch_bf": env_str("CH_BF"),
            "ch_gfp": env_str("CH_GFP"),
            "ch_rfp": env_str("CH_RFP"),
            "crop_half_size": env_int("CROP_HALF_SIZE", 0),
            "input_source": env_str("INPUT_SOURCE", "nd2", choices=("nd2", "split")),
            "split_dir": env_str("SPLIT_DIR", ""),
            "z_method": env_str(
                "Z_METHOD",
                "consensus",
                choices=("consensus", "focus", "fixed", "max", "mean"),
            ),
            "z_plane": env_int("Z_PLANE", 0),
            "z_max_step": env_int("Z_MAX_STEP", 1),
            "z_focus_sample_every": env_int("Z_FOCUS_SAMPLE_EVERY", 5),
            # Empty in config.sh = read it from the nd2. A number = force it.
            "frame_interval_min_override": env_float("FRAME_INTERVAL_MIN", None),
            "pixel_size_um_override": env_float("PIXEL_SIZE_UM", None),
        },
        "segment": {
            "model": env_str("OMNI_MODEL"),
            "mask_threshold": env_float("MASK_THRESHOLD"),
            "flow_threshold": env_float("FLOW_THRESHOLD", 0.0),
            "channel": env_str("SEGMENT_CHANNEL", "bf", choices=("bf", "gfp", "rfp")),
            "use_gpu": env_str("USE_GPU", "auto"),
            "affinity_seg": env_bool("AFFINITY_SEG", True),
            "preview_frames": env_int_list("PREVIEW_FRAMES", None),
        },
        "track": {
            "config_file": env_str(
                "BTRACK_CONFIG_FULL", env_str("BTRACK_CONFIG", "btrack_config.json")
            ),
            "search_radius": env_float("SEARCH_RADIUS", 20),
            "optimize": env_bool("OPTIMIZE_TRACKS", True),
            "relabel_method": env_str(
                "RELABEL_METHOD", "refs", choices=("refs", "centroid")
            ),
            "centroid_tolerance_px": env_int("CENTROID_TOLERANCE_PX", 8),
            "min_track_length": env_int("MIN_TRACK_LENGTH", 0),
            "method": env_str("TRACK_METHOD", "overlap", choices=("overlap", "btrack")),
            "min_overlap_link": env_float("TRACK_MIN_OVERLAP_LINK", 0.2),
            "min_daughter_frac": env_float("TRACK_MIN_DAUGHTER_FRAC", 0.30),
            "max_division_growth": env_float("TRACK_MAX_DIVISION_GROWTH", 1.4),
            "division_min_persist": env_int("TRACK_DIVISION_MIN_PERSIST", 4),
            "merge_frac": env_float("TRACK_MERGE_FRAC", 0.5),
            "split_suspect": env_bool("TRACK_SPLIT_SUSPECT", True),
            "min_overlap": env_float("TRACK_MIN_OVERLAP", 0.2),
            "max_step_um": env_float("TRACK_MAX_STEP_UM", 3.0),
            "max_area_ratio": env_float("TRACK_MAX_AREA_RATIO", 2.5),
            "max_gap_frames": env_int("TRACK_MAX_GAP_FRAMES", 2),
            "qc_frames": env_int("QC_FRAMES", 4),
        },
        "inspect": {
            "label_every_n_frames": env_int("LABEL_EVERY_N_FRAMES", 1),
            "label_font_size": env_int("LABEL_FONT_SIZE", 10),
            "label_min_track_length": env_int("LABEL_MIN_TRACK_LENGTH", 1),
            "show_outlines": env_bool("SHOW_OUTLINES", True),
            "overview_pdf": env_bool("OVERVIEW_PDF", True),
        },
        "cells": {
            # Text, not numbers: a track id (41) or a cell name (41-2) are
            # both accepted, and step 5 resolves names via lineage.csv.
            "track_ids": env_str_list("CELL_TRACK_IDS", []),
            "pad_px": env_int("CELL_PAD_PX", 10),
            "background": env_str(
                "CELL_BACKGROUND", "black", choices=("black", "white")
            ),
            "mask_outside": env_bool("CELL_MASK_OUTSIDE", True),
            "save_unmasked": env_bool("CELL_SAVE_UNMASKED", True),
            "align_major_axis": env_bool("CELL_ALIGN_MAJOR_AXIS", False),
            "follow_daughters": env_bool("CELL_FOLLOW_DAUGHTERS", False),
            "keep_gaps": env_bool("CELL_KEEP_GAPS", True),
        },
        "gfp": {
            # A pixel counts as structure when it is this much brighter than
            # the cell's OWN diffuse pool, as a fraction. Nothing here is a
            # percentile stretch, so a cell with an even glow yields nothing.
            "min_contrast": env_float("GFP_STRUCT_MIN_CONTRAST", 0.45),
            "noise_k": env_float("GFP_STRUCT_NOISE_K", 3.0),
            "min_area_px": env_int("GFP_STRUCT_MIN_AREA_PX", 4),
            "erode_px": env_int("GFP_ERODE_CELL_PX", 0),
            "close_gaps_px": env_int("GFP_CLOSE_GAPS_PX", 1),
            "use_ridge": env_bool("GFP_USE_RIDGE", True),
            "ridge_k": env_float("GFP_RIDGE_K", 4.0),
            "ridge_sigmas": env_float_list("GFP_RIDGE_SIGMAS", [1.0, 2.0, 3.0]),
            # Filament geometry, in microns — not as a fraction of the cell,
            # which would tighten silently as the cells elongate.
            "fil_min_length_um": env_float("GFP_FIL_MIN_LENGTH_UM", 1.2),
            "fil_max_width_um": env_float("GFP_FIL_MAX_WIDTH_UM", 0.45),
            "fil_min_aspect": env_float("GFP_FIL_MIN_ASPECT", 3.0),
            "fil_rule": env_str("GFP_FIL_RULE", "shape", choices=("shape", "microns")),
            "fil_max_circularity": env_float("GFP_FIL_MAX_CIRCULARITY", 0.60),
            "fil_min_elongation": env_float("GFP_FIL_MIN_ELONGATION", 1.8),
            "punct_unit_area_px": env_int("GFP_PUNCTA_UNIT_AREA_PX", 7),
            "punct_max_area_px": env_int("GFP_PUNCTA_MAX_AREA_PX", 70),
            "calibration_cells": env_int("GFP_CALIBRATION_CELLS", 6),
            "min_signal_over_bg": env_float("GFP_MIN_SIGNAL_OVER_BG", 3.0),
            "min_struct_area_frac": env_float("GFP_MIN_STRUCT_AREA_FRAC", 0.0),
            "punct_on_filament_ratio": env_float("GFP_PUNCTA_ON_FILAMENT_RATIO", 1.6),
            "punct_min_prominence": env_float("GFP_PUNCTA_MIN_PROMINENCE", 0.25),
            "polar_threshold": env_float("GFP_POLAR_THRESHOLD", 0.6),
            "state_min_duration": env_int("GFP_STATE_MIN_DURATION", 2),
        },
    }


# ── paths ───────────────────────────────────────────────────────────────────
STEP_DIRS = {
    "01_load": "01_load",
    "02_segment": "02_segment",
    "03_track": "03_track",
    "04_inspect": "04_inspect",
    "05_cells": "05_cells",
    "06_gfp": "06_gfp",
}


# ── lineages: names and colours ─────────────────────────────────────────────
# Nine colours chosen to stay apart from each other on a dark background.
LINEAGE_COLOURS = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def build_cell_names(records, continuation=None):
    """
    Give every track a name that shows where it came from.

    A founder keeps its track id: 12. Its daughters become 12-1 and 12-2, and
    their daughters 12-1-1, 12-1-2 and so on — so a name states the whole
    descent at a glance, and everything from one founder shares a colour.

    records: dicts with track_id, parent_id (blank for founders), start_frame.
    continuation: {track_id: original_id} for tracks that are the SAME cell
    continuing after a distrusted link was cut. Those get a name like 34c1
    and keep the generation of the original, because a cut is not a division —
    naming them 34-24 made a cut track look like a cell with 24 daughters.

    Returns {track_id: name}, {track_id: colour hex}.
    """
    continuation = continuation or {}
    from collections import defaultdict

    parent, start = {}, {}
    for r in records:
        tid = int(r["track_id"])
        p = str(r.get("parent_id", "") or "").strip()
        parent[tid] = int(p) if p else None
        try:
            start[tid] = int(r.get("start_frame", 0))
        except (TypeError, ValueError):
            start[tid] = 0

    kids = defaultdict(list)
    for tid, p in parent.items():
        if p is not None and p in parent and tid not in continuation:
            kids[p].append(tid)

    roots = sorted(
        [t for t, p in parent.items() if p is None or p not in parent],
        key=lambda t: (start.get(t, 0), t),
    )

    names, colours = {}, {}
    for i, root in enumerate(roots):
        names[root] = str(root)
        # Cycling rather than random: neighbouring founders always differ, and
        # the same movie always gets the same colours.
        colours[root] = LINEAGE_COLOURS[i % len(LINEAGE_COLOURS)]
        queue = [root]
        while queue:
            cur = queue.pop()
            ordered = sorted(kids.get(cur, []), key=lambda t: (start.get(t, 0), t))
            for n, kid in enumerate(ordered, start=1):
                if kid in names:  # guard against loops
                    continue
                names[kid] = f"{names[cur]}-{n}"
                colours[kid] = colours[root]
                queue.append(kid)

    # continuations: same cell, same colour, same generation, marked with c
    cont_count = defaultdict(int)
    for tid in sorted(continuation, key=lambda t: start.get(t, 0)):
        orig = continuation[tid]
        cont_count[orig] += 1
        base = names.get(orig, str(orig))
        names[tid] = f"{base}c{cont_count[orig]}"
        colours[tid] = colours.get(orig, LINEAGE_COLOURS[0])

    for tid in parent:  # anything left over
        names.setdefault(tid, str(tid))
        colours.setdefault(tid, LINEAGE_COLOURS[0])
    return names, colours


def load_cell_names(lineage_csv):
    """
    Names and colours for every track, from lineage.csv.

    Uses the columns written by the tracking step when they are there, and
    works them out from the parent column when they are not — so an older
    lineage.csv does not force the tracking step to be run again.
    """
    import csv as _csv

    rows = []
    with open(lineage_csv) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        return {}, {}
    if "cell_name" in rows[0] and rows[0].get("cell_name"):
        names = {int(r["track_id"]): r["cell_name"] for r in rows}
        colours = {
            int(r["track_id"]): r.get("lineage_colour") or LINEAGE_COLOURS[0]
            for r in rows
        }
        return names, colours
    return build_cell_names(rows)


def position_dir(cfg, position: int) -> Path:
    root = Path(cfg["experiment"]["output_root"]).expanduser()
    return root / cfg["experiment"]["name"] / f"pos_{position:02d}"


def step_dir(cfg, position: int, step: str, create: bool = False) -> Path:
    d = position_dir(cfg, position) / STEP_DIRS[step]
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def require(path, what: str, fix: str = "") -> Path:
    p = Path(path)
    if not p.exists():
        die(
            f"a file this step needs does not exist: {p}  ({what})",
            fix or "Run the previous step first — set it to TRUE in config.sh.",
        )
    return p


# ── nd2 timing ──────────────────────────────────────────────────────────────
# Fields that contain the word "time" but are not the frame's timestamp.
# 'Exposure Time [ms]' is the trap: it matches a naive search for a time
# field, it is the same on every frame, and the result is a run that appears
# to span zero minutes.
_NOT_A_TIMESTAMP = ("exposure", "temperature", "elapsed since", "residual")

# Tried in order. The first that exists in the records wins.
_TIME_FIELDS = (
    ("time [s]", 1.0),
    ("time [ms]", 1e-3),
    ("timestamp [s]", 1.0),
    ("timestamp [ms]", 1e-3),
    ("time (s)", 1.0),
    ("time", 1.0),
)


def _find_key(keys, *musts, forbid=()):
    for k in keys:
        low = str(k).lower()
        if all(m in low for m in musts) and not any(f in low for f in forbid):
            return k
    return None


def _find_time_key(keys):
    """The field holding each frame's timestamp, and its scale to seconds."""
    lowered = {str(k).strip().lower(): k for k in keys}
    for name, scale in _TIME_FIELDS:
        if name in lowered:
            return lowered[name], scale
    # nothing matched exactly: fall back to a search, but never onto a field
    # that only happens to have "time" in its name
    k = _find_key(keys, "time", "ms", forbid=_NOT_A_TIMESTAMP)
    if k:
        return k, 1e-3
    k = _find_key(keys, "time", forbid=("ms",) + _NOT_A_TIMESTAMP)
    if k:
        return k, 1.0
    return None, 1.0


def frame_times_from_events(f, position=None):
    """
    Actual per-frame timestamps in seconds, from the nd2's event records.

    These are what the microscope really did, as opposed to the period that
    was programmed before the run. Recent versions of the nd2 package return
    events as dicts; older ones return objects, so both are handled. In a
    multi-position file the same timepoint appears once per position, so
    records are grouped by timepoint index before differencing — otherwise
    the interval measured would be the gap between positions.
    """
    try:
        evs = list(f.events() or [])
    except Exception:
        return None
    rows = [
        e if isinstance(e, dict) else (getattr(e, "__dict__", None) or {}) for e in evs
    ]
    rows = [r for r in rows if r]
    if not rows:
        return None

    keys = list(rows[0].keys())
    tkey, scale = _find_time_key(keys)
    if tkey is None:
        return None
    ikey = _find_key(keys, "t", "index") or _find_key(
        keys, "index", forbid=("p ", "z ")
    )
    pkey = _find_key(keys, "p", "index") or _find_key(keys, "position", "name")

    times = {}
    for r in rows:
        val = r.get(tkey)
        if val is None:
            continue
        if position is not None and pkey is not None:
            try:
                if int(r.get(pkey)) != int(position):
                    continue
            except (TypeError, ValueError):
                pass
        idx = r.get(ikey) if ikey else len(times)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = len(times)
        times.setdefault(idx, []).append(float(val) * scale)

    if len(times) < 2:
        return None
    out = np.array([float(np.mean(times[k])) for k in sorted(times)])

    # A timestamp field must actually advance. If it does not, the wrong
    # field was read — a constant one such as exposure time — and returning
    # it would give an interval of zero that then propagates into every time
    # axis downstream.
    span = float(out[-1] - out[0])
    if not np.isfinite(span) or span <= 0:
        return None
    if float(np.median(np.diff(out))) <= 0:
        return None
    return out


def programmed_interval_min(f):
    """The period typed into the acquisition software before the run."""
    try:
        for loop in f.experiment:
            params = getattr(loop, "parameters", None)
            if params is None:
                continue
            per = getattr(params, "periodMs", None)
            if per:
                return float(per) / 60000.0
            for ph in getattr(params, "periods", []) or []:
                if getattr(ph, "periodMs", None):
                    return float(ph.periodMs) / 60000.0
    except Exception:
        pass
    return None


# ── channels ────────────────────────────────────────────────────────────────
def resolve_channel(setting_name, value, channel_names, log=None):
    """
    Turn a channel setting into an index.

    Either a number (0, 1, 2) or a name from the file ("PHASE", "GFP") works.
    Names are far safer: this file stores them as GFP, RFP, PHASE, so the
    usual assumption that brightfield is channel 0 would silently analyse the
    GFP image as if it were phase contrast.
    """
    value = str(value).strip()
    lowered = [str(n).strip().lower() for n in (channel_names or [])]

    if value.lstrip("-").isdigit():
        idx = int(value)
        if channel_names and 0 <= idx < len(channel_names) and log:
            log.info(f"{setting_name} = {idx} -> '{channel_names[idx]}'")
        return idx

    key = value.lower()
    if key in lowered:
        idx = lowered.index(key)
        if log:
            log.info(f"{setting_name} = '{value}' -> channel {idx}")
        return idx
    for i, n in enumerate(lowered):  # partial match
        if key in n or n in key:
            if log:
                log.info(
                    f"{setting_name} = '{value}' matched channel {i} "
                    f"('{channel_names[i]}')"
                )
            return i
    die(
        f"{setting_name} is set to '{value}', which is not a channel in this " f"file.",
        f"Channels present: {', '.join(str(n) for n in (channel_names or []))}."
        f" Use one of those names, or its number counting from 0.",
    )


# ── z stacks ────────────────────────────────────────────────────────────────
def focus_score(plane, crop=512, cells_only=True):
    """
    How sharp one z plane is.

    Variance of the Laplacian, divided by the square of the mean so planes of
    different brightness stay comparable, on a central crop for speed.

    By default only the busiest tenth of the pixels are used — the cells and
    their edges. In a sparse field most of the frame is empty background, and
    including it swamps the focus signal with camera noise: on a synthetic
    stack with realistic noise, restricting to the cells roughly doubled the
    margin between the sharpest plane and the next.
    """
    from scipy.ndimage import laplace, sobel

    a = np.asarray(plane, dtype=float)
    if crop and min(a.shape) > crop:
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        h = crop // 2
        a = a[cy - h : cy + h, cx - h : cx + h]
    m = float(a.mean())
    if m <= 0:
        return 0.0
    lap = laplace(a)
    if cells_only:
        g = np.hypot(sobel(a, 0), sobel(a, 1))
        sel = g >= np.percentile(g, 90)
        if sel.sum() >= 64:
            return float(lap[sel].var()) / (m * m)
    return float(lap.var()) / (m * m)


def best_plane(stack_zyx, crop=512, cells_only=True):
    """Index of the sharpest plane, plus the score of every plane."""
    scores = [
        focus_score(stack_zyx[z], crop, cells_only) for z in range(stack_zyx.shape[0])
    ]
    return int(np.argmax(scores)), scores


def smooth_plane_choices(chosen, max_step=1, log=None):
    """
    Stop the chosen plane hopping about.

    Focus drifts slowly, so a jump of several planes between one frame and the
    next is the focus metric being fooled — by a cell crossing the crop, say —
    not the stage moving. Each frame is held within max_step of the last, after
    a median filter has removed one-frame spikes.
    """
    if len(chosen) < 3:
        return list(chosen)
    smoothed = list(chosen)
    for i in range(1, len(smoothed) - 1):
        smoothed[i] = int(np.median(chosen[i - 1 : i + 2]))

    out = [smoothed[0]]
    for z in smoothed[1:]:
        prev = out[-1]
        out.append(int(np.clip(z, prev - max_step, prev + max_step)))
    moved = sum(1 for a, b in zip(chosen, out) if a != b)
    if moved and log:
        log.info(
            f"focus: {moved} of {len(chosen)} frames had their plane "
            f"adjusted to keep the choice smooth"
        )
    return out


def consensus_plane(all_scores):
    """
    One plane for the whole movie, decided by every frame together.

    Each frame's scores are divided by that frame's own mean first. Without
    that the comparison is dominated by how the field changes over time — a
    fuller field scores higher at every plane — and that variation is much
    larger than the difference between planes.

    Returns (best plane, margin over the runner-up, mean relative score per
    plane, how often that plane wins in a single frame).
    """
    rel = []
    for sc in all_scores.values():
        a = np.asarray(sc, dtype=float)
        m = a.mean()
        if m > 0:
            rel.append(a / m)
    if not rel:
        return None, 0.0, None, 0
    rel = np.vstack(rel)
    mean_rel = rel.mean(axis=0)
    best = int(np.argmax(mean_rel))
    order = np.sort(mean_rel)[::-1]
    margin = float((order[0] - order[1]) / order[0]) if len(order) > 1 else 0.0
    wins = int(np.sum(np.argmax(rel, axis=1) == best))
    return best, margin, mean_rel, wins


def read_focus_planes(path, n_frames, log):
    """
    The plane to use per frame, from a focus file written by step 00.

    Kept as a small editable table rather than baked into the pixels, so a
    frame that chose badly can be corrected by hand and the rest of the
    pipeline re-run without touching the nd2 again.
    """
    import csv as _csv

    path = Path(path)
    if not path.exists():
        return None
    chosen = {}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            try:
                chosen[int(r["frame"])] = int(float(r["z_plane"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not chosen:
        return None
    missing = [t for t in range(n_frames) if t not in chosen]
    if missing:
        log.warning(
            f"{path.name} has no plane for {len(missing)} frame(s), "
            f"e.g. {missing[:5]} — the nearest earlier frame is used"
        )
    out, last = [], chosen[min(chosen)]
    for t in range(n_frames):
        last = chosen.get(t, last)
        out.append(last)
    return out


def reduce_z(stack_zyx, method, plane):
    """Collapse a z stack to one image."""
    if method == "max":
        return stack_zyx.max(axis=0)
    if method == "mean":
        return stack_zyx.mean(axis=0).astype(stack_zyx.dtype)
    return stack_zyx[int(np.clip(plane, 0, stack_zyx.shape[0] - 1))]


def require_settings(section, keys, section_name, log=None):
    """
    Check that the config section really carries the settings this step uses.

    A version number alone is not enough: it only catches a mismatch if
    whoever changed common.py remembered to bump it. Naming the settings
    turns a bare KeyError deep inside a step into something that says which
    file is out of date and what is missing from it.
    """
    missing = [k for k in keys if k not in section]
    if missing:
        die(
            f"the settings for '{section_name}' are missing "
            f"{', '.join(repr(m) for m in missing)}.",
            "scripts/common.py is older than the step trying to use it. "
            "Copy the whole pipeline folder across, not single files.",
        )
    return section


def resolve_positions(cfg, spec, log):
    """
    Turn "0", "0 1 2" or "all" into a list of positions that actually exist.

    'all' means every position folder already on disk, so a tool run after the
    pipeline covers everything without the positions having to be listed
    again — the tools used to silently analyse only field 0.
    """
    spec = str(spec).strip()
    root = (
        Path(cfg["experiment"]["output_root"]).expanduser() / cfg["experiment"]["name"]
    )
    if spec.lower() == "all":
        found = sorted(
            int(d.name.split("_")[1])
            for d in root.glob("pos_*")
            if d.is_dir() and d.name.split("_")[-1].isdigit()
        )
        if not found:
            die(
                f"no processed positions found under {root}",
                "Run the pipeline first, or name a position with -p 0.",
            )
        log.info(f"positions: {found}")
        return found
    try:
        return [int(x) for x in spec.replace(",", " ").split()]
    except ValueError:
        die(
            f"could not read the position(s) '{spec}'.",
            'Use a number, a list like "0 1 2", or "all".',
        )


def load_masks(cfg, pos, log, allow_untracked=True):
    """
    The cell masks to analyse: tracked if available, plain segmentation if not.

    Counting how many cells carry a filament does NOT need tracking. Tracking
    answers "is this the same cell as before", which matters for single-cell
    movies and for lineage, but not for a proportion measured per frame. So
    when tracking has failed — dense microcolonies are the usual reason — the
    phenotype analysis can still run on the segmentation.

    The cost is that ids are not comparable between frames, so each frame is
    an independent sample of cells and a cell seen in 40 frames is counted 40
    times. Handle that by analysing ONE timepoint per field, which is what
    --frames is for.
    """
    import numpy as np

    track_file = step_dir(cfg, pos, "03_track") / "tracked_masks.npz"
    if track_file.exists():
        return np.load(track_file)["masks"], True, track_file

    seg_file = step_dir(cfg, pos, "02_segment") / "masks.npz"
    if seg_file.exists() and allow_untracked:
        log.warning(
            "no tracked masks — using the raw segmentation instead. "
            "Cell ids are then per frame only, so analyse a single "
            "timepoint per field rather than pooling frames."
        )
        return np.load(seg_file)["masks"], False, seg_file

    die(
        "no cell masks were found for this position.",
        "Run the segmentation step (SEGMENT_CELLS=TRUE), and the tracking "
        "step too if you want cells followed between frames.",
    )


def read_units(load_dir, log, cfg=None):
    """
    Pixel size and frame interval, or a clear failure.

    These used to fall back to 1.0 min per frame when acquisition.json was
    missing or incomplete. Nothing complained, and every figure came out
    labelled as though one frame were one minute — wrong axis labels on a
    plot that otherwise looks perfectly fine, which is the worst way for a
    number to be wrong. So there is no silent default any more.
    """
    import json as _json

    acq_file = Path(load_dir) / "acquisition.json"
    acq = {}
    if acq_file.exists():
        try:
            acq = _json.load(open(acq_file))
        except Exception:
            log.warning(f"could not read {acq_file}")

    px = acq.get("pixel_size_um")
    dt = acq.get("frame_interval_min")

    if cfg is not None:
        if cfg["load"].get("pixel_size_um_override"):
            px = float(cfg["load"]["pixel_size_um_override"])
        if cfg["load"].get("frame_interval_min_override"):
            dt = float(cfg["load"]["frame_interval_min_override"])

    if not px:
        die(
            f"the pixel size is not recorded in {acq_file}",
            "Re-run the load step (LOAD_ND2=TRUE), or set PIXEL_SIZE_UM in "
            "config.sh.",
        )
    if not dt:
        die(
            f"the frame interval is not recorded in {acq_file}",
            "Every time axis depends on it, so it is not guessed. Re-run the "
            "load step (LOAD_ND2=TRUE), or set FRAME_INTERVAL_MIN in "
            "config.sh.\n"
            "            To see what the file says:  python "
            "tools/check_nd2_timing.py <file.nd2>",
        )

    px, dt = float(px), float(dt)
    log.info(
        f"{px:.4f} um/px, {dt:g} min per frame "
        f"({acq.get('frame_interval_source', 'from acquisition.json')})"
    )
    if abs(dt - 1.0) < 1e-9:
        log.warning(
            "the frame interval is exactly 1.0 min — check that this "
            "is real and not a leftover default."
        )
    return px, dt


def resolve_timepoints(spec, n_frames, dt_min, log):
    """
    Turn TIMEPOINTS_MIN into minutes, understanding 'last'.

    'last' becomes the final frame of this movie, so a config can ask for the
    end without the number having to be edited per experiment — movies of
    different lengths then get the right final timepoint automatically.
    """
    end_min = (n_frames - 1) * dt_min
    out = []
    for part in str(spec).replace(",", " ").split():
        low = part.strip().lower()
        if low in ("last", "end", "final"):
            out.append(round(end_min, 4))
            continue
        try:
            out.append(float(part))
        except ValueError:
            die(
                f"TIMEPOINTS_MIN contains '{part}', which is neither a number "
                f"nor 'last'.",
                'For example  TIMEPOINTS_MIN="0 30 60 last"',
            )
    keep = sorted({t for t in out if t <= end_min + 1e-9})
    dropped = [t for t in out if t > end_min + 1e-9]
    if dropped:
        log.info(
            f"timepoints past the end of this movie "
            f"({end_min:g} min) were dropped: "
            + ", ".join(f"{t:g}" for t in sorted(set(dropped)))
        )
    return keep


def imread(path):
    """
    Read a TIFF, working around old tifffile against numpy 2.

    tifffile before ~2024.8 calls ndarray.newbyteorder(), which numpy 2.0
    removed. It only fires on files whose byte order differs from the
    machine's, so it takes out one position in a run and leaves the rest —
    which looks like a corrupt file rather than a library mismatch.
    """
    import tifffile

    try:
        return tifffile.imread(str(path))
    except AttributeError as exc:
        if "newbyteorder" not in str(exc):
            raise
        with tifffile.TiffFile(str(path)) as tf:
            pages = [p.asarray() for p in tf.pages]
        arr = np.squeeze(np.stack(pages)) if len(pages) > 1 else pages[0]
        if arr.dtype.byteorder not in ("=", "|"):
            arr = arr.view(arr.dtype.newbyteorder()).astype(arr.dtype.name)
        return arr


# ── provenance ──────────────────────────────────────────────────────────────
def write_meta(
    outdir: Path, step: str, params: dict, inputs: dict, results: dict
) -> Path:
    """Record what was run, with which settings, and what came out."""
    meta = {
        "step": step,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "settings_used": params,
        "inputs": {k: str(v) for k, v in inputs.items()},
        "results": results,
        "package_versions": _versions(),
    }
    path = Path(outdir) / "meta.json"
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path


def _versions():
    out = {}
    for mod in (
        "numpy",
        "skimage",
        "tifffile",
        "nd2",
        "btrack",
        "cellpose_omni",
        "omnipose",
    ):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            pass
    return out


# ── logging ─────────────────────────────────────────────────────────────────
def get_logger(name: str, quiet: bool = False) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.WARNING if quiet else logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("  %(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    return log


class Timer:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.log.info(f"{self.label} took {time.time() - self.t0:.1f}s")


# ── CLI ─────────────────────────────────────────────────────────────────────
def base_parser(description: str) -> argparse.ArgumentParser:
    """The only argument a step takes is which field of view to process."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--position", type=int, default=0, help="0-based field-of-view index"
    )
    p.add_argument("--quiet", action="store_true")
    return p


def run_safely(main_fn, step_name: str):
    """
    Run a step so that ANY failure ends in the same readable message.

    Errors we anticipated already call die(). This catches the rest — a
    corrupt file, a library complaining, a full disk — prints the technical
    detail for whoever can read it, and then says plainly what happened.
    """
    import traceback

    try:
        return main_fn()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        die(
            f"the '{step_name}' step was cancelled from the keyboard.",
            "Nothing was saved for this position. Run again when ready.",
        )
    except MemoryError:
        die(
            f"the '{step_name}' step ran out of memory.",
            "Reduce CROP_HALF_SIZE in config.sh, or process fewer positions "
            "at a time.",
        )
    except Exception as exc:
        print("")
        print("  --- technical details (for troubleshooting) ---")
        traceback.print_exc()
        die(
            f"the '{step_name}' step stopped with an unexpected error: "
            f"{type(exc).__name__}: {exc}",
            "The technical details above are the useful part — send them on "
            "if you need help. Check first that the input files are complete "
            "and the settings in config.sh are correct.",
        )

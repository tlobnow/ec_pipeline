#!/usr/bin/env python3
"""
Step 05 — follow one cell, together with its history, as its own small movie.

A track id only lasts from one division to the next, so a cell picked at
150 min has no past of its own. This step walks back up the lineage and shows
whichever ancestor was present at each earlier frame, so a single cell can be
watched from the start of the movie through every division that led to it.
Optionally it also walks forward into the daughters.

Outputs (in <pos>/05_cells/):
    track_0040.tif           masked, cell centred, channels bf/gfp/rfp
    track_0040_unmasked.tif  same crop with the surroundings kept
    track_0040_frames.csv    which movie frame, which ancestor, present or not
    track_0040_montage.png   fixed timepoints, one column each, with a merge
    meta.json

The box size is chosen once, from the largest the cell ever gets, then held
fixed — so the movie does not jump around as the cell grows.

Note: the masked stack is for looking at, not for measuring. Blanking the
surroundings removes the local background that intensity measurements need.
Quantification should use the measurement step on the full image.
"""

from __future__ import annotations

import csv as _csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    Timer,
    base_parser,
    die,
    env_bool,
    env_int_list,
    env_str,
    get_logger,
    load_cell_names,
    load_config,
    read_units,
    require,
    resolve_timepoints,
    run_safely,
    step_dir,
    write_meta,
)


def resolve_targets(wanted, names, present_ids, log):
    """
    Turn what was typed in CELL_TRACK_IDS into track ids.

    Either form works: the raw track id (41) or the lineage name (41-2).
    Names are the ones drawn on the overview PDF and the Fiji image, so they
    are usually what you have in front of you.
    """
    by_name = {v: k for k, v in names.items()}
    out, unknown, dropped = [], [], []
    for item in wanted:
        item = str(item).strip()
        if not item:
            continue
        if item in by_name:
            tid = by_name[item]
        elif item.lstrip("-").isdigit():
            tid = int(item)
        else:
            unknown.append(item)
            continue
        if tid not in present_ids:
            dropped.append(item)
        elif tid not in out:
            out.append(tid)
            if names.get(tid, str(tid)) != item:
                log.info(f"'{item}' is track {tid} " f"(cell {names.get(tid, tid)})")

    if unknown:
        sample = sorted(by_name)[:15]
        die(
            f"CELL_TRACK_IDS contains {unknown}, which is neither a track id "
            f"nor a cell name.",
            "Use the number or the name shown on the overview PDF, e.g. "
            'CELL_TRACK_IDS="41" or CELL_TRACK_IDS="41-2".\n'
            "            Names in this field of view include: "
            + ", ".join(sample)
            + (" ..." if len(by_name) > 15 else ""),
        )
    if dropped:
        die(
            f"{dropped} exist in lineage.csv but not in the mask stack.",
            "They were removed by MIN_TRACK_LENGTH because they are too "
            "short-lived. Lower it in config.sh and re-run the tracking step.",
        )
    return out


CHANNELS = ("bf", "gfp", "rfp")


# ── lineage ─────────────────────────────────────────────────────────────────
def read_lineage(lineage_csv: Path):
    """parent of each track, and the children of each track."""
    parent, kids = {}, defaultdict(list)
    with open(lineage_csv) as fh:
        for row in _csv.DictReader(fh):
            tid = int(row["track_id"])
            p = (row.get("parent_id") or "").strip()
            if p:
                parent[tid] = int(p)
                kids[int(p)].append(tid)
    return parent, kids


def lineage_chain(lineage_csv, tid, want_ancestors, want_daughters, log):
    """
    Every track id that is the same cell at some point in time.

    Backwards this is a straight line — a cell has exactly one mother — so at
    any earlier frame precisely one of these ids is present. Forwards it
    branches, and after a division the box holds the whole family.
    """
    if not Path(lineage_csv).exists():
        log.warning("lineage.csv not found; following this track id only")
        return [tid], [], []

    parent, kids = read_lineage(lineage_csv)
    ids = {tid}

    up = []
    if want_ancestors:
        cur, seen = parent.get(tid), {tid}
        while cur is not None and cur not in seen:
            up.append(cur)
            seen.add(cur)
            ids.add(cur)
            cur = parent.get(cur)
        if up:
            log.info(
                f"ancestors of {tid}: "
                f"{' -> '.join(str(x) for x in reversed(up))} -> {tid}"
            )
        else:
            log.info(
                f"track {tid} has no recorded mother — it is a founder, "
                f"or its mother was dropped by MIN_TRACK_LENGTH"
            )

    down = []
    if want_daughters:
        queue = [tid]
        while queue:
            cur = queue.pop()
            for k in kids.get(cur, []):
                if k not in ids:
                    ids.add(k)
                    down.append(k)
                    queue.append(k)
        if down:
            log.info(f"daughters of {tid}: {sorted(down)}")

    return sorted(ids), up, down


def lineage_targets(lineage_csv, tracked, present_ids, wanted, min_frames, log):
    """
    Which founders to extract, one output each, when working per lineage.

    A founder is a track with no mother. Its output follows it and every
    descendant, so one file covers a whole family tree from the first frame
    it appears to the last frame any of its offspring is present.
    """
    if not Path(lineage_csv).exists():
        die(
            "extracting per lineage needs lineage.csv, which is missing.",
            "Set TRACK_CELLS=TRUE in config.sh and run again.",
        )

    parent, kids = read_lineage(lineage_csv)
    founders = [
        t
        for t in present_ids
        if parent.get(t) is None or parent.get(t) not in present_ids
    ]

    if wanted:
        unknown = [t for t in wanted if t not in founders]
        if unknown:
            die(
                f"CELL_LINEAGE_IDS contains {unknown}, which are not founders.",
                "A founder is a cell with no mother — those are the ids "
                "without a '-' in their name. Founders here: "
                + ", ".join(str(t) for t in founders[:20])
                + (" ..." if len(founders) > 20 else ""),
            )
        return wanted

    # How long each family lasts in total, used to drop the debris.
    lifespan = {}
    for t in range(tracked.shape[0]):
        for tid in np.unique(tracked[t]):
            if tid:
                lifespan[int(tid)] = lifespan.get(int(tid), 0) + 1

    def family(root):
        out, queue = {root}, [root]
        while queue:
            for k in kids.get(queue.pop(), []):
                if k not in out:
                    out.add(k)
                    queue.append(k)
        return out

    keep = [
        f for f in founders if sum(lifespan.get(k, 0) for k in family(f)) >= min_frames
    ]
    log.info(
        f"{len(founders)} founders in this field of view, "
        f"{len(keep)} last at least {min_frames} frames in total"
    )
    if not keep:
        die(
            f"no lineage lasts {min_frames} frames or more, so nothing "
            f"would be written.",
            "Lower CELL_LINEAGE_MIN_FRAMES in config.sh.",
        )
    if len(keep) > 30:
        log.warning(
            f"{len(keep)} lineages will be written — that is a lot of "
            f"files. Raise CELL_LINEAGE_MIN_FRAMES, or name the ones "
            f"you want in CELL_LINEAGE_IDS."
        )
    return keep


# ── geometry ────────────────────────────────────────────────────────────────
def cell_frames(tracked, ids):
    """Frames where any of these ids is present, with position and extent."""
    info = {}
    for t in range(tracked.shape[0]):
        m = np.isin(tracked[t], ids)
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        here = [int(v) for v in np.unique(tracked[t][m])]
        info[t] = {
            "cy": float(ys.mean()),
            "cx": float(xs.mean()),
            "h": int(ys.max() - ys.min() + 1),
            "w": int(xs.max() - xs.min() + 1),
            "area": int(m.sum()),
            "ids": here,
        }
    return info


def box_size(info, pad, align, log):
    """One box for the whole movie, big enough for the cell at its largest."""
    if align:
        # after rotation the long axis lies across the box diagonal
        need = max(int(np.hypot(v["h"], v["w"])) for v in info.values())
    else:
        need = max(max(v["h"], v["w"]) for v in info.values())
    size = int(need + 2 * pad)
    size += size % 2  # keep it even
    log.info(f"box {size}x{size} px (largest extent {need} px + {pad} px pad)")
    return size


def centre_for(t, info, frames_present):
    """
    Where to point the box on a frame where the cell is not there.

    Before it first appears, use its first known position; after it is gone,
    its last known one. The box then stays where the cell was instead of
    jumping to a corner.
    """
    if t in info:
        return info[t]["cy"], info[t]["cx"]
    earlier = [f for f in frames_present if f < t]
    nearest = max(earlier) if earlier else min(frames_present)
    return info[nearest]["cy"], info[nearest]["cx"]


def crop_centred(plane, cy, cx, size, fill):
    """Crop a fixed-size box centred on (cy, cx), padding past the edges."""
    half = size // 2
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    out = np.full((size, size), fill, dtype=plane.dtype)
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(plane.shape[0], y0 + size), min(plane.shape[1], x0 + size)
    if ys1 > ys0 and xs1 > xs0:
        out[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0] = plane[ys0:ys1, xs0:xs1]
    return out


def rotate_to_horizontal(img, mask, angle_rad):
    """Turn the cell so its long axis runs left-right."""
    from scipy.ndimage import rotate

    deg = 90.0 - np.degrees(angle_rad)
    img_r = rotate(img, deg, reshape=False, order=1, mode="constant", cval=0)
    mask_r = (
        rotate(
            mask.astype(np.uint8), deg, reshape=False, order=0, mode="constant", cval=0
        )
        > 0
    )
    return img_r, mask_r


# ── montage ─────────────────────────────────────────────────────────────────
def channel_cmaps():
    """Black-based colour maps, so the panels look like the Fiji composite."""
    from matplotlib.colors import LinearSegmentedColormap as LSC

    return {
        "bf": LSC.from_list("bf", ["black", "white"]),
        "gfp": LSC.from_list("gfp", ["black", "#00ff44"]),
        "rfp": LSC.from_list("rfp", ["black", "#ff2222"]),
    }


def scale_channel(stack, ci, fill):
    """Display range from the cell's own pixels, fixed across all timepoints."""
    vals = stack[:, ci]
    inside = vals[vals != fill]
    if inside.size < 10:
        return 0.0, 1.0
    lo, hi = np.percentile(inside, [1, 99.5])
    if hi <= lo:
        lo, hi = float(inside.min()), float(max(inside.max(), inside.min() + 1))
    return float(lo), float(hi)


def montage(
    stack,
    raw,
    span,
    dt_min,
    title,
    requested_min,
    info,
    names,
    fill,
    show_merge,
    path,
    log,
):
    """One column per requested timepoint, whether or not the cell is there."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index_of = {t: i for i, t in enumerate(span)}
    cols, skipped = [], []
    for minute in requested_min:
        frame = int(round(minute / dt_min))
        if frame in index_of:
            cols.append((minute, frame, index_of[frame]))
        else:
            skipped.append(minute)
    if skipped:
        log.info(
            f"montage: {', '.join(f'{m:g}' for m in skipped)} min not "
            f"covered by this movie, those columns are left out"
        )
    if not cols:
        log.warning("montage: none of the requested timepoints exist here")
        return

    cmaps = channel_cmaps()
    ranges = {c: scale_channel(stack, i, fill) for i, c in enumerate(CHANNELS)}

    def norm(i, ci, src=None, blank_bg=True):
        src = stack if src is None else src
        lo, hi = ranges[CHANNELS[ci]]
        plane = src[i, ci].astype(float)
        d = np.clip((plane - lo) / (hi - lo), 0, 1)
        if blank_bg:
            d[stack[i, ci] == fill] = 0.0  # blank stays black
        return d

    def merged(i, src=None, blank_bg=True):
        bf, gfp, rfp = (
            norm(i, 0, src, blank_bg),
            norm(i, 1, src, blank_bg),
            norm(i, 2, src, blank_bg),
        )
        rgb = np.zeros(bf.shape + (3,))
        rgb[..., 0] = np.clip(rfp + 0.35 * bf, 0, 1)
        rgb[..., 1] = np.clip(gfp + 0.35 * bf, 0, 1)
        rgb[..., 2] = np.clip(0.35 * bf, 0, 1)
        return rgb

    rows = list(CHANNELS)
    if show_merge:
        rows += ["merge"]
        if raw is not None:
            rows += ["context"]  # the same merge without the mask
    fig, axes = plt.subplots(
        len(rows), len(cols), figsize=(2.3 * len(cols), 2.5 * len(rows)), squeeze=False
    )

    for col, (minute, frame, i) in enumerate(cols):
        present = frame in info
        for r, name in enumerate(rows):
            ax = axes[r][col]
            if name == "merge":
                ax.imshow(merged(i))
            elif name == "context":
                ax.imshow(merged(i, raw, blank_bg=False))
            else:
                ax.imshow(norm(i, rows.index(name)), cmap=cmaps[name], vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                shown = (
                    "/".join(names.get(x, str(x)) for x in info[frame]["ids"])
                    if present
                    else "absent"
                )
                ax.set_title(
                    f"{minute:g} min\nframe {frame} · {shown}",
                    fontsize=8,
                    color="black" if present else "0.55",
                )
            if col == 0:
                ax.set_ylabel(
                    {
                        "bf": "BF",
                        "gfp": "GFP",
                        "rfp": "RFP/PI",
                        "merge": "merge",
                        "context": "merge\n+ context",
                    }[name],
                    fontsize=9,
                )

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"montage -> {Path(path).name} ({len(cols)} timepoints)")


# ── main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = base_parser(__doc__.split("\n")[1]).parse_args(argv)
    cfg = load_config()
    log = get_logger("05_cells", args.quiet)

    cc = cfg["cells"]
    pos = args.position

    # settings introduced by this step, read with the same helpers
    follow_ancestors = env_bool("CELL_FOLLOW_ANCESTORS", True)
    span_mode = env_str("CELL_SPAN", "movie", choices=("movie", "lineage"))
    show_merge = env_bool("CELL_MONTAGE_SHOW_MERGE", True)
    requested_min = None  # needs the movie length; set below
    extract = env_str("CELL_EXTRACT", "ids", choices=("ids", "lineages"))
    lineage_min_frames = int(os.environ.get("CELL_LINEAGE_MIN_FRAMES") or 0)

    load_dir = step_dir(cfg, pos, "01_load")
    track_dir = step_dir(cfg, pos, "03_track")
    mask_file = require(
        track_dir / "tracked_masks.npz",
        "the tracked masks",
        "Set TRACK_CELLS=TRUE in config.sh and run again.",
    )
    tif_paths = {
        c: require(
            load_dir / f"{c}.tif",
            f"the {c} channel",
            "Set LOAD_ND2=TRUE in config.sh and run again.",
        )
        for c in CHANNELS
    }
    out = step_dir(cfg, pos, "05_cells", create=True)

    import tifffile
    from skimage import measure

    tracked = np.load(mask_file)["masks"]
    T = tracked.shape[0]

    px_um, dt_min = read_units(load_dir, log, cfg)
    log.info(f"{T} frames at {dt_min} min = {(T - 1) * dt_min:.0f} min total")
    requested_min = resolve_timepoints(
        os.environ.get("TIMEPOINTS_MIN", "0 30 45 60 90 120 150"), T, dt_min, log
    )

    lineage_csv = track_dir / "lineage.csv"
    names, _ = load_cell_names(lineage_csv) if lineage_csv.exists() else ({}, {})

    def label_of(tid):
        return names.get(int(tid), str(int(tid)))

    # ── which cells ────────────────────────────────────────────────────────
    present_ids = sorted({int(v) for v in np.unique(tracked) if v})

    if extract == "lineages":
        # One output per founder: the founder plus everything it divided into.
        wanted = lineage_targets(
            lineage_csv,
            tracked,
            present_ids,
            env_int_list("CELL_LINEAGE_IDS", None),
            lineage_min_frames,
            log,
        )
        follow_ancestors, follow_daughters = True, True
    else:
        wanted = resolve_targets(cc["track_ids"], names, present_ids, log)
        follow_daughters = bool(cc["follow_daughters"])

    if not wanted:
        lengths = {}
        for t in range(T):
            for tid in np.unique(tracked[t]):
                if tid:
                    lengths[int(tid)] = lengths.get(int(tid), 0) + 1
        longest = sorted(lengths.items(), key=lambda kv: -kv[1])[:10]
        die(
            "CELL_TRACK_IDS is empty, so there is no cell to extract.",
            "Open 04_inspect/open_in_fiji.ijm to read an id off the image, "
            'then set it in config.sh, e.g. CELL_TRACK_IDS="40" or '
            'CELL_TRACK_IDS="40-2".\n'
            "            The longest-lived tracks here are: "
            + ", ".join(f"{tid} ({n} frames)" for tid, n in longest),
        )

    missing = [i for i in wanted if i not in present_ids]
    if missing:
        die(
            f"track id(s) {missing} do not exist in this field of view.",
            f"Ids present here run from {min(present_ids)} to "
            f"{max(present_ids)}. Check CELL_TRACK_IDS in config.sh, and that "
            f"POSITIONS points at the field of view you looked at in Fiji.",
        )

    with Timer(log, "loading channels"):
        images = {c: tifffile.imread(p) for c, p in tif_paths.items()}

    bg_white = str(cc["background"]).lower() == "white"
    fill = 65535 if bg_white else 0
    pad = int(cc["pad_px"])
    align = bool(cc["align_major_axis"])
    keep_gaps = bool(cc["keep_gaps"])
    mask_outside = bool(cc["mask_outside"])
    save_unmasked = bool(cc["save_unmasked"])

    results = {}
    for tid in wanted:
        ids, up, down = lineage_chain(
            lineage_csv, tid, follow_ancestors, follow_daughters, log
        )

        info = cell_frames(tracked, ids)
        if not info:
            die(
                f"track {tid} has no pixels in the mask stack.",
                "It was probably dropped by MIN_TRACK_LENGTH. Lower that "
                "setting in config.sh and re-run the tracking step.",
            )

        frames = sorted(info)
        f0, f1 = frames[0], frames[-1]

        # ── which frames end up in the stack ───────────────────────────────
        if span_mode == "movie":
            span = list(range(T))
        elif keep_gaps:
            span = list(range(f0, f1 + 1))
        else:
            span = frames
        blanks = [f for f in span if f not in info]

        size = box_size(info, pad, align, log)
        log.info(
            f"track {tid}: lineage present frames {f0}-{f1} "
            f"({len(frames)} frames), writing {len(span)} frames "
            f"({len(blanks)} blank)"
        )
        if span_mode == "movie" and (f0 > 0 or f1 < T - 1):
            log.info(
                f"  frames outside {f0}-{f1} are blank, held at the "
                f"cell's nearest known position"
            )

        stack = np.full((len(span), len(CHANNELS), size, size), fill, dtype=np.uint16)
        # Always built: the montage shows a merge with the surroundings kept,
        # which is what tells you whether the outline is right.
        stack_raw = np.zeros_like(stack)

        rows = []
        for i, t in enumerate(span):
            cy, cx = centre_for(t, info, frames)
            here = np.isin(tracked[t], ids)
            mask_crop = crop_centred(here.astype(np.uint8), cy, cx, size, 0) > 0

            angle = None
            if align and mask_crop.any():
                sub = measure.regionprops(mask_crop.astype(np.uint8))
                angle = sub[0].orientation if sub else 0.0

            for ci, ch in enumerate(CHANNELS):
                crop = crop_centred(images[ch][t], cy, cx, size, 0)
                mk = mask_crop
                if align and angle is not None:
                    crop, mk = rotate_to_horizontal(crop, mask_crop, angle)
                stack_raw[i, ci] = crop
                stack[i, ci] = np.where(mk, crop, fill) if mask_outside else crop

            v = info.get(t)
            rows.append(
                (
                    t,
                    f"{cy:.1f}",
                    f"{cx:.1f}",
                    v["area"] if v else 0,
                    "/".join(str(x) for x in v["ids"]) if v else "",
                    1 if v else 0,
                )
            )

        meta_ij = {
            "axes": "TCYX",
            "unit": "um",
            "finterval": dt_min,
            "tunit": "min",
            "mode": "composite",
        }
        res = (1.0 / px_um, 1.0 / px_um)
        prefix = "lineage" if extract == "lineages" else "track"
        suffix = "" if label_of(tid) == str(tid) else f"_{label_of(tid)}"
        base = out / f"{prefix}_{tid:04d}{suffix}"
        tifffile.imwrite(
            f"{base}.tif", stack, imagej=True, resolution=res, metadata=meta_ij
        )
        if save_unmasked:
            tifffile.imwrite(
                f"{base}_unmasked.tif",
                stack_raw,
                imagej=True,
                resolution=res,
                metadata=meta_ij,
            )

        with open(f"{base}_frames.csv", "w") as fh:
            fh.write(
                "frame,time_min,centroid_y,centroid_x,area_px," "visible_id,present\n"
            )
            for t, cy, cx, area, vis, pres in rows:
                fh.write(f"{t},{t * dt_min:.2f},{cy},{cx},{area},{vis},{pres}\n")

        if extract == "lineages":
            title = (
                f"lineage {label_of(tid)} — founder and all its "
                f"descendants ({len(ids)} track ids)"
            )
        else:
            title = (
                f"cell {label_of(tid)} (track {tid}) — followed through " f"its lineage"
            )
        montage(
            stack,
            stack_raw,
            span,
            dt_min,
            title,
            requested_min,
            info,
            names,
            fill,
            show_merge,
            f"{base}_montage.png",
            log,
        )

        results[str(tid)] = {
            "cell_name": label_of(tid),
            "ids_followed": ids,
            "ancestors": list(reversed(up)),
            "daughters": sorted(down),
            "first_frame_present": f0,
            "last_frame_present": f1,
            "span_mode": span_mode,
            "n_frames_written": len(span),
            "n_blank_frames": len(blanks),
            "box_px": size,
            "box_um": round(size * px_um, 2),
        }
        log.info(f"track {tid} -> {base}.tif")

    write_meta(
        out,
        "05_cells",
        dict(
            cc,
            position=pos,
            follow_ancestors=follow_ancestors,
            span=span_mode,
            montage_timepoints_min=requested_min,
            montage_show_merge=show_merge,
        ),
        {"tracked_masks": mask_file},
        results,
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "single cell"))

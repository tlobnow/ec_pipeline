#!/usr/bin/env python
"""
Export one small image per cell, ready to sort into example folders by hand.

    python tools/export_cells.py -c configs/my.sh --frames "0 15 30"
    python tools/export_cells.py -c configs/my.sh --frames "0 15 30" --max 400

Each cell is cut out on its own, with everything outside its outline set to
black, so what you see is only that cell. Neighbours are gone, which is the
whole point: on the contact sheet a neighbour's focus sitting at the pole is
easy to mistake for this cell's.

What you get, under <pos>/07_training/:

    _unsorted/          every cell as a PNG, ready to look through
    none/               <- empty, drop examples in here
    diffuse/            <- and here
    punctate/
    filamentous/
    mixed/
    features.csv        the measurements for every cell, keyed by file name
    raw/                the same crops as 16-bit TIFF, if you want the numbers

Time stacks (--stacks)
----------------------
    python tools/export_cells.py -c configs/my.sh --stacks

writes one TIF per cell instead, following it through every frame it exists
in, for pixel classification in convpaint, napari, Ilastik and so on:

    cell_00042_7-1.tif    (T, C, Y, X) — GFP masked, plus a boundary channel

  * the values are the RAW counts, not display-scaled
  * the boundary is a SEPARATE channel, never drawn into the image, so a
    pixel classifier cannot key on a line that was added by this tool
  * the box is sized once from the cell at its largest and then held fixed,
    so the cell stays centred as it grows instead of drifting
  * --stack-channels "gfp,bf" adds more channels; --all-frames covers the
    whole movie with blanks where the cell is absent

Correcting instead of sorting
-----------------------------
    python tools/export_cells.py -c cfg.sh -p all --frames "0 20 40" \
           --presort --png-contrast per-cell --with-tif --max 0

    --presort         each cell starts in the folder the pipeline guessed
    --png-contrast    per-cell stretches every crop, so faint structure shows
    --with-tif        a calibrated 16-bit TIF beside each PNG, for measuring
    --max 0           no cap; export every segmented cell
    -p all            every field of view

Move only what is in the wrong folder. Then:

    python tools/export_cells.py -c cfg.sh --collect filamentous

writes sorted_labels.csv (the final call for every cell, beside the
pipeline's guess, and whether you changed it) and gathers that class's TIFs
into to_measure_filamentous/. Open those in Fiji, draw along the filament and
press M: the length is in microns, because the pixel size is in each file.

Sort by dragging files from _unsorted/ into the five folders. Anything left
in _unsorted/ is simply ignored. Then:

    python tools/tune_gfp.py -c configs/my.sh --train --labels-from-folders

which reads the folder each file ended up in as its label.

The PNGs are contrast-scaled on ONE shared range so a dim cell looks dim.
Sorting on individually stretched images would teach the model the opposite
of what you mean.
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_CANDIDATES = [HERE, HERE.parent / "scripts", HERE.parent]
_SCRIPTS = next((d for d in _CANDIDATES if (d / "06_gfp_structures.py").exists()), None)
if _SCRIPTS is None:
    sys.exit("Cannot find 06_gfp_structures.py next to this tool.")
sys.path.insert(0, str(_SCRIPTS))

import importlib.util

from common import (
    imread as safe_imread,
    die,
    get_logger,
    load_cell_names,
    load_config,
    load_masks,
    read_units,
    require,
    resolve_positions,
    resolve_timepoints,
    step_dir,
)

_spec = importlib.util.spec_from_file_location(
    "gfp6", _SCRIPTS / "06_gfp_structures.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def require_compatible(module, log):
    """
    Refuse to run against a stale copy of the detector.

    The tools and scripts are updated together. If only one is copied across,
    the mismatch surfaces either as a confusing TypeError or — far worse —
    as a run that quietly uses an old detector and produces results that
    cannot be compared with anything else.
    """
    from common import PIPELINE_VERSION

    have = getattr(module, "MODULE_VERSION", None)
    if have == PIPELINE_VERSION:
        return
    where = (
        Path(module.__file__).parent
        if getattr(module, "__file__", None)
        else "the scripts folder"
    )
    die(
        f"the detector in {where} is version {have or 'older than 0.8.0'}, "
        f"but this tool expects {PIPELINE_VERSION}.",
        "Copy the whole pipeline folder across, not single files — "
        "scripts/ and tools/ have to match.\n"
        "            The mismatched file is 06_gfp_structures.py; "
        "common.py is usually stale too.",
    )


CLASSES = ["none", "diffuse", "punctate", "filamentous", "mixed"]
BINARY_CLASSES = ["filamentous", "non_filamentous"]

# In binary mode a cell counts as filamentous if it has a filament at all,
# whether or not it also has foci — "mixed" is a filament-bearing cell.
BINARY_MAP = {
    "filamentous": "filamentous",
    "mixed": "filamentous",
    "none": "non_filamentous",
    "diffuse": "non_filamentous",
    "punctate": "non_filamentous",
}


def load_shell_config(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        die(f"config file not found: {path}", "Check the path given to -c.")
    out = subprocess.run(
        ["bash", "-c", f'set -a; . "{path}"; env'], capture_output=True, text=True
    )
    if out.returncode != 0:
        die(
            f"could not read the config file: {path}",
            "Look for a missing quote or a space around an = sign.",
        )
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v
    os.environ.setdefault("CONFIG_DIR", str(path.parent))


LUT_COLOURS = {
    "gfp": (0, 1, 0),
    "bf": (1, 1, 1),
    "rfp": (1, 0, 0),
    "boundary": (0.55, 0.55, 0.65),
}


def channel_lut(name):
    """
    The colour a channel is drawn in.

    Without this, viewers fall back on their default channel order — red,
    green, blue — so GFP comes out red and the boundary yellow, which is
    both confusing and easy to misread when several stacks are open.
    """
    r, g, b = LUT_COLOURS.get(name, (1, 1, 1))
    ramp = np.arange(256, dtype=np.uint8)
    return np.stack(
        [
            (ramp * r).astype(np.uint8),
            (ramp * g).astype(np.uint8),
            (ramp * b).astype(np.uint8),
        ]
    )


def context_stack(
    tracked, gfp, frames, tid_set, boxes, scale, px_um, dt_min, path, log
):
    """
    The whole field per frame, with the crop's box drawn on it.

    A per-cell stack looks convincing even when tracking has jumped to a
    neighbour: the crop is always centred on something cell-shaped. Showing
    where each frame's crop was taken from makes that visible — the box
    should creep along with one cell, and a jump between frames means the
    id moved to a different cell.

    Downscaled by `scale`, because a full 2048x2048 field over 60 frames is
    large and the only thing being judged here is position.
    """
    import tifffile
    from skimage.segmentation import find_boundaries

    Y, X = tracked.shape[1], tracked.shape[2]
    ys, xs = Y // scale, X // scale
    out = np.zeros((len(frames), 2, ys, xs), np.uint16)

    lo, hi = np.percentile(gfp[frames], [1, 99.5])
    hi = max(float(hi), float(lo) + 1)

    for i, t in enumerate(frames):
        small = gfp[t][::scale, ::scale].astype(float)
        out[i, 0] = np.clip((small - lo) / (hi - lo), 0, 1) * 65535

        mark = np.zeros((ys, xs), np.uint16)
        sel = np.isin(tracked[t], list(tid_set))[::scale, ::scale]
        if sel.any():
            mark[find_boundaries(sel, mode="outer")] = 40000
        box = boxes.get(t)
        if box is not None:
            y0, x0, y1, x1 = [v // scale for v in box]
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(ys - 1, y1), min(xs - 1, x1)
            mark[y0, x0 : x1 + 1] = 65535
            mark[y1, x0 : x1 + 1] = 65535
            mark[y0 : y1 + 1, x0] = 65535
            mark[y0 : y1 + 1, x1] = 65535
        out[i, 1] = mark

    tifffile.imwrite(
        path,
        out,
        imagej=True,
        resolution=(1.0 / (px_um * scale), 1.0 / (px_um * scale)),
        metadata={
            "axes": "TCYX",
            "unit": "um",
            "finterval": dt_min,
            "tunit": "min",
            "mode": "composite",
            "Labels": ["field (downscaled)", "crop box + cell"] * len(frames),
        },
    )
    log.info(
        f"    context -> {path.name}  (field at 1/{scale} scale, with "
        f"the crop box per frame)"
    )


def cell_stacks(
    gfp_all,
    tracked,
    channels,
    images,
    out_dir,
    names,
    px_um,
    dt_min,
    pad,
    max_cells,
    all_frames,
    log,
    context_scale=0,
    gfp_params=None,
):
    """
    One TIF per cell, following it through time.

    Written for pixel classification (convpaint and similar), which is why:

      * the values are the RAW 16-bit counts, not display-scaled. A classifier
        should see the real numbers.
      * the boundary is a SEPARATE channel, not drawn into the image. A line
        burned into the intensity data would become something the classifier
        learns from, which is not what you want it keying on.
      * the box is sized once from the cell at its largest and then held, so
        the cell does not drift or jump as it grows.
      * frames where the cell is missing are kept as blank, so the time axis
        still matches the original movie.
    """
    import tifffile
    from skimage import measure
    from skimage.segmentation import find_boundaries

    T = tracked.shape[0]
    Y, X = tracked.shape[1], tracked.shape[2]

    # where each cell is, in every frame
    where = defaultdict(dict)
    for t in range(T):
        if not tracked[t].any():
            continue
        for reg in measure.regionprops(tracked[t]):
            y0, x0, y1, x1 = reg.bbox
            where[int(reg.label)][t] = (reg.centroid, y1 - y0, x1 - x0, int(reg.area))

    order = sorted(where, key=lambda k: -len(where[k]))
    if max_cells:
        order = order[:max_cells]
    log.info(
        f"{len(order)} cell(s) to write "
        f"(longest-lived first, limit {max_cells or 'none'})"
    )

    written = 0
    for tid in order:
        frames = sorted(where[tid])
        span = range(T) if all_frames else range(frames[0], frames[-1] + 1)
        span = list(span)

        need = max(max(h, w) for (_, h, w, _) in where[tid].values())
        size = int(need + 2 * pad)
        size += size % 2

        want_path = gfp_params is not None
        n_ch = len(channels) + 1 + (1 if want_path else 0)
        stack = np.zeros((len(span), n_ch, size, size), np.uint16)
        boxes = {}

        last_centre = where[tid][frames[0]][0]
        for i, t in enumerate(span):
            if t in where[tid]:
                last_centre = where[tid][t][0]
            cy, cx = last_centre
            half = size // 2
            y0, x0 = int(round(cy)) - half, int(round(cx)) - half
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(Y, y0 + size), min(X, x0 + size)
            if ye <= ys or xe <= xs:
                continue
            dy, dx = ys - y0, xs - x0
            h, w = ye - ys, xe - xs

            cm = np.zeros((size, size), bool)
            cm[dy : dy + h, dx : dx + w] = tracked[t][ys:ye, xs:xe] == tid
            if not cm.any():
                continue  # blank frame, cell gone

            for ci, ch in enumerate(channels):
                tile = np.zeros((size, size), np.uint16)
                tile[dy : dy + h, dx : dx + w] = images[ch][t][ys:ye, xs:xe]
                stack[i, ci] = np.where(cm, tile, 0)

            edge = find_boundaries(cm, mode="inner")
            stack[i, len(channels)][edge] = 65535
            boxes[t] = (y0, x0, y0 + size - 1, x0 + size - 1)

            if want_path:
                # the line whose length is reported for this frame, so growth
                # over time can be read off the images and not just the CSV
                g = G
                bgv = float(np.median(images["gfp"][t][tracked[t] == 0]))
                sgv = g.pixel_noise(images["gfp"][t], tracked[t] == 0)
                sub = np.zeros((size, size), images["gfp"].dtype)
                sub[dy : dy + h, dx : dx + w] = images["gfp"][t][ys:ye, xs:xe]
                _f, fm, _p, _r, _s = g.measure_cell(
                    sub, cm, bgv, sgv, px_um, gfp_params
                )
                pth, _L = g.filament_paths(fm, px_um)
                stack[i, n_ch - 1][pth] = 65535

        name = str(names.get(tid, tid)).replace("/", "-")
        path = out_dir / f"cell_{tid:05d}_{name}.tif"
        tifffile.imwrite(
            path,
            stack,
            imagej=True,
            resolution=(1.0 / px_um, 1.0 / px_um),
            metadata={
                "axes": "TCYX",
                "unit": "um",
                "finterval": dt_min,
                "tunit": "min",
                "mode": "composite",
                "LUTs": [channel_lut(c) for c in channels] + [channel_lut("boundary")],
                "Labels": (
                    channels + ["boundary"] + (["measured path"] if want_path else [])
                )
                * len(span),
            },
        )
        written += 1
        if written <= 3 or written % 25 == 0:
            log.info(
                f"  {path.name}  {stack.shape}  frames "
                f"{span[0]}-{span[-1]}  box {size}px"
            )
        if context_scale:
            context_stack(
                tracked,
                images["gfp"],
                span,
                {tid},
                boxes,
                int(context_scale),
                px_um,
                dt_min,
                out_dir / f"cell_{tid:05d}_{name}_context.tif",
                log,
            )

    log.info(f"{written} cell stack(s) written to {out_dir}")
    log.info(f"channels in each: {', '.join(channels)}, boundary")
    return written


def merge_labels(src_dir, dest_dir, rows, log):
    """
    Copy a finished round of sorting into the shared training set.

    A review round lands in its own folder, and leaving it there means the
    next training run does not see it. Copying the images and their
    measurements into the one training folder keeps a single place to train
    from, and means a cell judged twice is judged once — the newer call wins.
    """
    import csv as _csv
    import shutil

    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for r in rows:
        cls = r["final_call"]
        (dest_dir / cls).mkdir(exist_ok=True)
        src = src_dir / cls / r["file"]
        if not src.exists():
            continue
        # a re-judged cell may sit in the other folder from an earlier round
        for other in (dest_dir / c for c in CLASSES + BINARY_CLASSES):
            stale = other / r["file"]
            if other.name != cls and stale.exists():
                stale.unlink()
        shutil.copy2(src, dest_dir / cls / r["file"])
        moved += 1

    # features: keep one row per file, the newest winning
    src_feat = {r["file"]: r for r in _csv.DictReader(open(src_dir / "features.csv"))}
    dest_feat_path = dest_dir / "features.csv"
    keep = []
    if dest_feat_path.exists():
        keep = [
            r
            for r in _csv.DictReader(open(dest_feat_path))
            if r["file"] not in src_feat
        ]
    add = [src_feat[r["file"]] for r in rows if r["file"] in src_feat]
    if keep or add:
        cols = list((keep or add)[0].keys())
        with open(dest_feat_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(keep + add)
    return moved


def collect_sorted(cfg, out_dir, want, log, pad=4, merge_into=""):
    """
    Read back what the folders say, and cut the TIFs for the class you kept.

    The crops are made HERE rather than at export time, so only the cells that
    survived sorting cost a file. Everything needed to find each cell again is
    in features.csv — position, frame and track id — so the crop is taken from
    the original image, not from the PNG that was looked at.
    """
    import csv as _csv

    import tifffile
    from skimage.segmentation import find_boundaries

    feats = out_dir / "features.csv"
    if not feats.exists():
        die(
            f"no features.csv in {out_dir}",
            "Export the cells first, then sort them, then collect.",
        )
    guess = {r["file"]: r for r in _csv.DictReader(open(feats))}

    # whichever folders actually exist — five phenotypes, or the two of
    # binary mode
    present = [c for c in CLASSES + BINARY_CLASSES if (out_dir / c).is_dir()]
    rows, counts, changed = [], defaultdict(int), 0
    for cls in present:
        folder = out_dir / cls
        if not folder.is_dir():
            continue
        for f in sorted(folder.glob("*.png")):
            rec = guess.get(f.name)
            if rec is None:
                log.warning(
                    f"'{f.name}' is in {cls}/ but not in "
                    f"features.csv — keep the exported file names"
                )
                continue
            row = {
                k: rec.get(k, "")
                for k in (
                    "file",
                    "position",
                    "frame",
                    "frame_fiji",
                    "time_min",
                    "track_id",
                    "cell_name",
                    "pixel_size_um",
                    "crop_y0",
                    "crop_x0",
                    "pipeline_call",
                )
            }
            row["final_call"] = cls
            row["corrected"] = int(rec.get("pipeline_call") != cls)
            changed += row["corrected"]
            rows.append(row)
            counts[cls] += 1

    if not rows:
        die(
            f"nothing has been sorted into the phenotype folders in {out_dir}",
            "Move the images into the folders first, or export with --presort "
            "so they start in the folder the pipeline guessed.",
        )

    path = out_dir / "sorted_labels.csv"
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    log.info(
        f"{len(rows)} sorted cell(s): "
        + "  ".join(f"{k} {v}" for k, v in counts.items())
    )
    log.info(
        f"you corrected {changed} of them "
        f"({100 * changed / len(rows):.0f}% of the pipeline's guesses)"
    )
    log.info(f"final labels -> {path}")

    if merge_into:
        merge_into = Path(merge_into).expanduser()
        if merge_into.resolve() != out_dir.resolve():
            n = merge_labels(out_dir, merge_into, rows, log)
            log.info(f"{n} cell(s) added to the training set at {merge_into}")

    if not want:
        return
    if want not in present:
        die(
            f"--collect '{want}' is not one of the folders in {out_dir}.",
            "Folders here: " + ", ".join(present),
        )

    wanted = [r for r in rows if r["final_call"] == want]
    if not wanted:
        log.warning(f"no cells ended up in '{want}'")
        return

    dest = out_dir / f"to_measure_{want}"
    dest.mkdir(exist_ok=True)

    # group the work by field and frame so each image is opened once
    by_field = defaultdict(list)
    for r in wanted:
        by_field[int(r["position"])].append(r)

    n_tif = 0
    for pos, recs in sorted(by_field.items()):
        ld = step_dir(cfg, pos, "01_load")
        gfp = safe_imread(
            require(
                ld / "gfp.tif", f"pos {pos} GFP", "Run the pipeline for that field."
            )
        )
        masks, _, _ = load_masks(cfg, pos, log)
        px_um, _ = read_units(ld, log, cfg)
        Y, X = masks.shape[1], masks.shape[2]

        by_frame = defaultdict(list)
        for r in recs:
            by_frame[int(r["frame"])].append(r)

        # ---- the crops, one per kept cell --------------------------------
        for t, rs in sorted(by_frame.items()):
            lbl = masks[t]
            for r in rs:
                tid = int(r["track_id"])
                sel = lbl == tid
                if not sel.any():
                    log.warning(
                        f"cell {tid} is not in frame {t} of field "
                        f"{pos} any more — skipped"
                    )
                    continue
                ys, xs = np.nonzero(sel)
                y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
                y1, x1 = min(Y, ys.max() + 1 + pad), min(X, xs.max() + 1 + pad)
                cm = sel[y0:y1, x0:x1]
                crop = np.where(cm, gfp[t][y0:y1, x0:x1], 0).astype(np.uint16)
                # two channels: the signal, and the outline on its own. Kept
                # separate so nothing is drawn into the data being measured.
                edge = find_boundaries(cm, mode="inner")
                stack = np.zeros((2,) + crop.shape, np.uint16)
                stack[0] = crop
                stack[1][edge] = 65535
                name = r["file"].replace(".png", ".tif")
                tifffile.imwrite(
                    dest / name,
                    stack,
                    imagej=True,
                    resolution=(1.0 / px_um, 1.0 / px_um),
                    metadata={
                        "axes": "CYX",
                        "unit": "um",
                        "mode": "composite",
                        "Labels": ["gfp", "outline"],
                    },
                )
                n_tif += 1

        # ---- the whole field, with those cells marked --------------------
        frames = sorted(by_frame)
        # two channels over the chosen frames:
        #   0  the kept cells, pixel VALUE = track id, so hovering in Fiji
        #      shows the id with nothing drawn over the image
        #   1  every cell outlined, to see the kept ones in context
        marked = np.zeros((len(frames), 2, Y, X), np.uint16)
        for i, t in enumerate(frames):
            ids = [int(r["track_id"]) for r in by_frame[t]]
            sel = np.isin(masks[t], ids)
            marked[i, 0][sel] = masks[t][sel].astype(np.uint16)
            marked[i, 1][find_boundaries(masks[t], mode="inner")] = 65535
        mark_path = out_dir / f"marked_{want}_pos{pos:02d}.tif"
        tifffile.imwrite(
            mark_path,
            marked,
            imagej=True,
            resolution=(1.0 / px_um, 1.0 / px_um),
            metadata={
                "axes": "TCYX",
                "unit": "um",
                "mode": "composite",
                "Labels": ["kept cell ids", "all cell outlines"] * len(frames),
            },
        )
        log.info(
            f"  field {pos}: {len(recs)} cell(s), marked frames "
            f"{frames} -> {mark_path.name}"
        )

    log.info("")
    log.info(f"{n_tif} calibrated TIF(s) -> {dest}")
    log.info(
        "  open one in Fiji, draw along the filament with the line tool "
        "and press M — the length is in microns."
    )
    log.info("")
    log.info(
        f"marked_{want}_posNN.tif shows the whole field with only these "
        f"cells kept. Each cell's PIXEL VALUE is its track id, so "
        f"hovering over it in Fiji shows the id in the status bar."
    )
    log.info(
        "  Drop it on the GFP stack (Image > Color > Merge Channels) to "
        "see which cells were kept, in place."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument(
        "-p",
        "--position",
        default="0",
        help="which field(s) of view: a number, a list like "
        '"0 1 2", or "all" for every one that has been '
        "processed. Default 0.",
    )
    ap.add_argument(
        "--frames",
        default="0",
        help='which frames, e.g. "0 15 30", or "last" for the '
        "final frame of the movie",
    )
    ap.add_argument(
        "--max",
        type=int,
        default=300,
        help="cap the cells taken PER FIELD OF VIEW (default 300; "
        "0 = no limit). The cap used to be shared across "
        "fields, so a busy first field could use it all up "
        "and the others produced nothing.",
    )
    ap.add_argument(
        "--pad", type=int, default=4, help="black margin around the cell, in pixels"
    )
    ap.add_argument(
        "--to",
        default="",
        help="write into this folder instead of the field's own "
        "07_training. Point every position and every "
        "experiment at one folder to build a single "
        "training set.",
    )
    ap.add_argument(
        "--binary",
        action="store_true",
        help="two folders only: filamentous and non_filamentous. "
        "'mixed' counts as filamentous, since the cell does "
        "have a filament.",
    )
    ap.add_argument(
        "--dir",
        default="",
        help="which folder to collect from. Defaults to "
        "07_training; point it at 08_review after a review "
        "round.",
    )
    ap.add_argument(
        "--collect",
        nargs="?",
        const="",
        default=None,
        help="read back the corrected folders and write "
        "sorted_labels.csv. Give a phenotype "
        "(--collect filamentous) to also gather that "
        "class's TIFs into one folder for measuring.",
    )
    ap.add_argument(
        "--raw", action="store_true", help="also write 16-bit TIFFs of the same crops"
    )
    ap.add_argument(
        "--with-tif",
        action="store_true",
        help="write a calibrated 16-bit TIF beside each PNG. The "
        "PNG is stretched for looking at; the TIF holds the "
        "real counts and the pixel size, so a length "
        "measured on it in Fiji comes out in microns.",
    )
    ap.add_argument(
        "--presort",
        action="store_true",
        help="drop each PNG straight into the folder for the "
        "phenotype the pipeline guessed, so you correct "
        "mistakes instead of sorting from scratch",
    )
    ap.add_argument(
        "--stacks",
        action="store_true",
        help="write one TIF per cell following it through time, "
        "instead of a PNG per cell per frame",
    )
    ap.add_argument(
        "--stack-channels",
        default="gfp",
        help="which channels go into the stacks, e.g. "
        '"gfp,bf". The boundary is always added as a '
        "separate channel.",
    )
    ap.add_argument(
        "--stack-max",
        type=int,
        default=0,
        help="write at most this many cells (0 = all)",
    )
    ap.add_argument(
        "--context-scale",
        type=int,
        default=4,
        help="also write a downscaled full-field stack per cell "
        "with the crop box drawn on each frame, so a crop "
        "that jumps to a different cell is visible. "
        "0 turns it off.",
    )
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="cover the whole movie, not only the frames where " "this cell exists",
    )
    ap.add_argument(
        "--png-contrast",
        default="shared",
        choices=("shared", "per-cell"),
        help="shared (default) puts every cell on one brightness "
        "scale, so a dim cell looks dim. per-cell stretches "
        "each one, which makes faint structure much easier "
        "to SEE — use it for sorting, but remember it makes "
        "an empty cell look full.",
    )
    ap.add_argument(
        "--no-outline", action="store_true", help="do not draw the cell boundary"
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=3,
        help="enlarge each crop by this factor, so small cells "
        "are visible in a file browser (default 3)",
    )
    args = ap.parse_args()

    load_shell_config(args.config)
    log = get_logger("export", False)
    require_compatible(G, log)
    try:
        import imageio.v3 as iio

        writer = lambda path, arr: iio.imwrite(path, arr)
    except ImportError:
        try:
            from matplotlib import pyplot as plt

            writer = lambda path, arr: plt.imsave(path, arr)
        except ImportError:
            die(
                "neither imageio nor matplotlib is installed in this python: "
                + sys.executable,
                "conda activate phage_pipeline && pip install imageio",
            )

    import tifffile
    from skimage import measure
    from skimage.segmentation import find_boundaries

    cfg = load_config()
    p = dict(cfg["gfp"])
    positions = resolve_positions(cfg, args.position, log)

    if args.collect is not None:
        if args.dir:
            out = Path(args.dir).expanduser()
        else:
            out = step_dir(cfg, positions[0], "03_track").with_name("07_training")
        if not out.is_dir():
            die(
                f"folder not found: {out}",
                "Point --dir at the folder holding the sorted images, for "
                "example the 08_review folder written by classify_cells.py.",
            )
        log.info(f"collecting from {out}")
        collect_sorted(cfg, out, args.collect, log, args.pad, args.to)
        return 0

    pos = positions[0]
    load_dir = step_dir(cfg, pos, "01_load")
    track_dir = step_dir(cfg, pos, "03_track")
    gfp = safe_imread(
        require(
            load_dir / "gfp.tif",
            "the GFP channel",
            "Run the pipeline's load step first.",
        )
    )
    tracked, _, _ = load_masks(cfg, pos, log)

    px_um, dt_min = read_units(load_dir, log, cfg)

    lin = track_dir / "lineage.csv"
    names = load_cell_names(lin)[0] if lin.exists() else {}

    frames = []
    for f in args.frames.replace(",", " ").split():
        if f.strip().lower() in ("last", "end", "final"):
            frames.append(gfp.shape[0] - 1)
        else:
            frames.append(int(f))
    frames = sorted(set(frames))
    # One --frames setting is shared across experiments of different lengths,
    # so a frame past the end is dropped with a note rather than being fatal.
    bad = [f for f in frames if not 0 <= f < gfp.shape[0]]
    frames = [f for f in frames if 0 <= f < gfp.shape[0]]
    if bad:
        log.info(
            f"frame(s) {bad} are past the end of this movie "
            f"({gfp.shape[0]} frames), skipped"
        )
    if not frames:
        die(
            f"none of the requested frames exist in this movie, which has "
            f"{gfp.shape[0]} frames (0 to {gfp.shape[0] - 1}).",
            "Pick frames inside that range with --frames.",
        )

    classes = BINARY_CLASSES if args.binary else CLASSES
    if args.to:
        # one shared folder, so every position and every experiment can be
        # sorted together and train one model
        out = Path(args.to).expanduser()
    else:
        out = step_dir(cfg, pos, "03_track").with_name("07_training")
    unsorted = out / "_unsorted"
    unsorted.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (out / c).mkdir(exist_ok=True)
    if args.raw:
        (out / "raw").mkdir(exist_ok=True)

    if args.stacks:
        wanted = [
            c.strip()
            for c in args.stack_channels.replace(",", " ").split()
            if c.strip()
        ]
        bad = [c for c in wanted if c not in ("gfp", "bf", "rfp")]
        if bad:
            die(
                f"--stack-channels contains {bad}, which is not a channel.",
                "Use any of: gfp, bf, rfp — for example  " '--stack-channels "gfp,bf"',
            )
        images = {"gfp": gfp}
        for c in wanted:
            if c not in images:
                images[c] = safe_imread(
                    require(
                        load_dir / f"{c}.tif",
                        f"the {c} channel",
                        "Run the pipeline's load step first.",
                    )
                )
        for pos_i in positions:
            if pos_i != pos:
                ld = step_dir(cfg, pos_i, "01_load")
                gfp = safe_imread(ld / "gfp.tif")
                tracked, _, _ = load_masks(cfg, pos_i, log)
                l2 = step_dir(cfg, pos_i, "03_track") / "lineage.csv"
                names = load_cell_names(l2)[0] if l2.exists() else {}
                images = {"gfp": gfp}
                for c in wanted:
                    if c not in images:
                        images[c] = safe_imread(ld / f"{c}.tif")
            log.info(f"--- position {pos_i} ---")
            cell_stacks(
                gfp,
                tracked,
                wanted,
                images,
                unsorted,
                names,
                px_um,
                dt_min,
                args.pad,
                args.stack_max,
                args.all_frames,
                log,
                args.context_scale,
                dict(cfg["gfp"]),
            )
        log.info("")
        log.info(
            "Each TIF opens as a time series with the channels above. "
            "The values are the original counts, and the boundary is a "
            "separate channel so it is not part of the image data a "
            "pixel classifier would learn from."
        )
        return 0

    # One display range for every crop, taken from in-cell pixels.
    sample = np.concatenate(
        [gfp[f][tracked[f] > 0].ravel() for f in frames if (tracked[f] > 0).any()]
    )
    if sample.size == 0:
        die(
            "no cells were found in those frames.",
            "Check that the tracking step produced masks for them.",
        )
    lo, hi = np.percentile(sample, [1, 99.5])
    hi = max(hi, lo + 1)
    log.info(f"display range {lo:.0f}-{hi:.0f} counts, shared by every crop")

    expt_tag = cfg["experiment"]["name"].replace(".nd2", "").replace("/", "-")
    rows, n = [], 0
    jobs = [(pos, gfp, tracked, names)]
    for extra in positions[1:]:
        ld = step_dir(cfg, extra, "01_load")
        g2 = safe_imread(
            require(
                ld / "gfp.tif",
                f"pos {extra} GFP",
                "Run the pipeline for that position.",
            )
        )
        m2, _, _ = load_masks(cfg, extra, log)
        l2 = step_dir(cfg, extra, "03_track") / "lineage.csv"
        jobs.append((extra, g2, m2, load_cell_names(l2)[0] if l2.exists() else {}))
    log.info(f"exporting from {len(jobs)} field(s) of view")

    capped = []
    for pos, gfp, tracked, names in jobs:
        n_here = 0
        for t in frames:
            if t >= gfp.shape[0]:
                continue
            lbl = tracked[t]
            if not lbl.any():
                continue
            bg = float(np.median(gfp[t][lbl == 0]))
            sigma = G.pixel_noise(gfp[t], lbl == 0)
            for reg in measure.regionprops(lbl):
                if args.max and n_here >= args.max:
                    if pos not in capped:
                        capped.append(pos)
                    break
                y0, x0, y1, x1 = reg.bbox
                pad = args.pad
                Y, X = lbl.shape
                ys, xs = max(0, y0 - pad), max(0, x0 - pad)
                ye, xe = min(Y, y1 + pad), min(X, x1 + pad)

                cm = lbl[ys:ye, xs:xe] == reg.label
                crop = gfp[t][ys:ye, xs:xe]

                feat, fmask, pts, rel, smask = G.measure_cell(
                    crop, cm, bg, sigma, px_um, p
                )
                state = G.state_of(
                    feat, p["min_signal_over_bg"], p["min_struct_area_frac"]
                )

                # black everywhere except this cell
                if args.png_contrast == "per-cell":
                    v = crop[cm]
                    clo, chi = np.percentile(v, [2, 99.5]) if v.size > 10 else (lo, hi)
                    chi = max(float(chi), float(clo) + 1)
                    disp = np.clip((crop.astype(float) - clo) / (chi - clo), 0, 1)
                else:
                    disp = np.clip((crop.astype(float) - lo) / (hi - lo), 0, 1)
                disp[~cm] = 0.0
                rgb = np.zeros(disp.shape + (3,), dtype=np.uint8)
                rgb[..., 1] = (disp * 255).astype(np.uint8)

                if not args.no_outline:
                    # Drawn just OUTSIDE the mask, in the padding, so it frames
                    # the cell without covering any signal at the rim — where
                    # polar foci sit.
                    edge = find_boundaries(cm, mode="outer")
                    rgb[edge] = (110, 110, 130)

                if args.scale > 1:
                    rgb = np.repeat(
                        np.repeat(rgb, args.scale, axis=0), args.scale, axis=1
                    )

                name = names.get(int(reg.label), str(int(reg.label)))
                safe = str(name).replace("/", "-")
                # the experiment goes in the name so several runs can share
                # one training folder without colliding
                fname = (
                    f"{expt_tag}_pos{pos:02d}_f{t:03d}_"
                    f"id{int(reg.label):05d}_{safe}.png"
                )
                # With --presort the guess decides where the file lands, so the
                # job becomes correcting rather than sorting. The guess is kept
                # in features.csv either way, so how often you disagreed can be
                # measured afterwards.
                filed = BINARY_MAP[state] if args.binary else state
                dest = (out / filed) if args.presort else unsorted
                writer(dest / fname, rgb)

                if args.with_tif:
                    # Raw counts, masked to this cell, with the pixel size in
                    # the header. Fiji then reports a length in microns
                    # directly, and the file sits beside its PNG so the cell
                    # sorted by eye is the cell measured.
                    masked = np.where(cm, crop, 0).astype(np.uint16)
                    tifffile.imwrite(
                        dest / fname.replace(".png", ".tif"),
                        masked,
                        imagej=True,
                        resolution=(1.0 / px_um, 1.0 / px_um),
                        metadata={"unit": "um"},
                    )
                if args.raw:
                    masked = np.where(cm, crop, 0).astype(np.uint16)
                    tifffile.imwrite(
                        out / "raw" / fname.replace(".png", ".tif"), masked
                    )

                row = {
                    "file": fname,
                    "experiment": expt_tag,
                    "condition": cfg["experiment"].get("condition", ""),
                    "position": pos,
                    "frame": t,
                    "frame_fiji": t + 1,
                    "time_min": round(t * dt_min, 2),
                    "track_id": int(reg.label),
                    "cell_name": name,
                    "pixel_size_um": px_um,
                    "crop_y0": int(ys),
                    "crop_x0": int(xs),
                    "pipeline_call": filed if args.binary else state,
                }
                row.update({k: round(float(feat.get(k, 0)), 4) for k in G.FEATURES})
                rows.append(row)
                n += 1
                n_here += 1

    feat_path = out / "features.csv"
    existing = []
    if feat_path.exists():
        # a shared folder collects several runs, so add rather than replace,
        # keeping only the newest row for any file exported twice
        existing = [
            r
            for r in _csv.DictReader(open(feat_path))
            if r["file"] not in {x["file"] for x in rows}
        ]
    with open(feat_path, "w", newline="") as fh:
        cols = list(rows[0].keys())
        w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(existing + rows)
    if existing:
        log.info(
            f"features.csv now holds {len(existing) + len(rows)} cell(s) "
            f"from this and earlier runs"
        )

    n_polar = sum(1 for r in rows if float(r.get("punct_axial_pos_max", 0)) >= 0.6)
    if capped:
        log.warning(
            f"the per-field cap of {args.max} was reached in "
            f"field(s) {capped}, so not every cell was exported. "
            f"Raise it with --max, or use --max 0 for no limit."
        )
    if args.presort:
        guessed = defaultdict(int)
        for r in rows:
            guessed[r["pipeline_call"]] += 1
        log.info(f"{n} cells written, already sorted by the pipeline's guess:")
        for k in classes:
            if guessed[k]:
                log.info(f"    {k:12s} {guessed[k]}")
        log.info("")
        log.info(
            "Go through them and MOVE anything that is in the wrong "
            "folder. What you do not move counts as agreement, so look "
            "at all of them, not only the ones you doubt."
        )
    else:
        log.info(f"{n} cells written to {unsorted}")
    if n_polar:
        log.info(
            f"{n_polar} of them have a focus in the outer third of the "
            f"cell — polar localisation is recorded per cell in "
            f"features.csv as punct_axial_pos_max, not filtered out"
        )
    log.info("")
    log.info("Next: drag files from _unsorted/ into the folders beside it —")
    log.info(f"    {'  '.join(CLASSES)}")
    log.info("Anything left in _unsorted/ is ignored. Aim for at least ten of")
    log.info("each, including the awkward ones. Then:")
    log.info("")
    log.info(
        f"    python tools/tune_gfp.py -c {args.config} "
        f"--train --labels-from-folders"
    )
    log.info("")
    log.info(f"everything is in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

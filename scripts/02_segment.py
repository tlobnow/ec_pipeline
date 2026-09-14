#!/usr/bin/env python
"""
Step 02 — Omnipose segmentation of the brightfield stack.

Outputs (in <pos>/02_segment/):
    masks.npz                   uint16 label image per frame, key 'masks'
    cells_per_frame.csv         frame, frame_fiji, time_min, n_cells,
                                median_cell_area_px, area_fraction
    segmentation_outlines.tif   binary layer to drop on the GFP stack in Fiji
    qc_segmentation.png
    meta.json

    ../cell_counts_all_positions.csv   every field so far, grouped by field
                                       and ordered by frame

Preview mode: set segment.preview_frames in the config to a short list to tune
mask_threshold on a handful of frames in seconds. Preview results are written
to 02_segment_preview/ and are never picked up by step 03, so a tuning run can
never be mistaken for a real one.
"""

from __future__ import annotations

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
    get_logger,
    load_config,
    require,
    run_safely,
    step_dir,
    write_meta,
)


def seg_cfg_bool(name, default):
    v = os.environ.get(name, "").strip().upper()
    return default if v == "" else v in ("TRUE", "YES", "1")


def combine_counts(cfg, log):
    """
    One CSV across every field processed so far, grouped by field and ordered
    by frame — written next to the per-field files so a whole experiment can
    be read at a glance without opening five folders.
    """
    import csv as _csv

    root = (
        Path(cfg["experiment"]["output_root"]).expanduser() / cfg["experiment"]["name"]
    )
    rows = []
    for csv_path in sorted(root.glob("pos_*/02_segment/cells_per_frame.csv")):
        pos = csv_path.parent.parent.name
        for r in _csv.DictReader(open(csv_path)):
            rows.append({"position": pos, **r})
    if not rows:
        return
    rows.sort(key=lambda r: (r["position"], int(r["frame"])))
    out = root / "cell_counts_all_positions.csv"
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n_pos = len({r["position"] for r in rows})
    log.info(f"cell counts for {n_pos} field(s) -> {out}")


def qc_figure(bf, masks, frames, path, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10), squeeze=False)
    for j, t in enumerate(frames):
        axes[0, j].imshow(bf[t], cmap="gray")
        axes[0, j].set_title(f"BF frame {t}")
        m = masks[t]
        shown = np.where(m > 0, (m % 19) + 1, 0)  # recycle colours, keep 0 black
        axes[1, j].imshow(shown, cmap="tab20", interpolation="nearest", vmin=0, vmax=20)
        axes[1, j].set_title(
            f"masks: {int(m.max())} labels, " f"{100 * (m > 0).mean():.0f}% area"
        )
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
    fig.suptitle(
        "Segmentation QC — merged cells: lower mask_threshold; "
        "missed cells: raise it",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"QC figure -> {path}")


def main(argv=None):
    args = base_parser(__doc__.split("\n")[1]).parse_args(argv)
    cfg = load_config()
    log = get_logger("02_segment", args.quiet)

    sc = cfg["segment"]
    preview = sc.get("preview_frames")
    params = dict(sc, position=args.position)

    load_dir = step_dir(cfg, args.position, "01_load")
    src = require(
        load_dir / f"{sc['channel']}.tif",
        "the tif written by the load step",
        "Set LOAD_ND2=TRUE in config.sh and run again.",
    )

    out = step_dir(cfg, args.position, "02_segment", create=True)
    if preview:
        out = out.with_name("02_segment_preview")
        out.mkdir(parents=True, exist_ok=True)
        log.warning(
            f"PREVIEW MODE — only frames {preview} will be segmented. "
            f'Set PREVIEW_FRAMES="" in config.sh for a real run.'
        )

    import tifffile

    dt_min = None
    acq = load_dir / "acquisition.json"
    if acq.exists():
        try:
            import json

            dt_min = json.load(open(acq)).get("frame_interval_min")
            dt_min = float(dt_min) if dt_min else None
        except Exception:
            pass

    img = safe_imread(src)
    if img.ndim != 3:
        die(
            f"expected a time series (frames, height, width) in {src.name}, "
            f"but its shape is {img.shape}.",
            "Re-run the load step with LOAD_ND2=TRUE.",
        )
    T = img.shape[0]
    frames = list(preview) if preview else list(range(T))
    frames = [f for f in frames if 0 <= f < T]
    log.info(f"segmenting {len(frames)} of {T} frames from {src.name}")

    # GPU
    use_gpu = sc.get("use_gpu", "auto")
    if use_gpu == "auto":
        try:
            from omnipose.gpu import use_gpu as _ug

            use_gpu = bool(_ug())
        except Exception:
            use_gpu = False
    log.info(f"GPU: {use_gpu}")

    from cellpose_omni import models

    try:
        from omnipose.gpu import empty_cache

        empty_cache()
    except Exception:
        pass

    model = models.CellposeModel(
        gpu=use_gpu, model_type=sc["model"], nclasses=2, nchan=1
    )
    with Timer(log, "omnipose eval"):
        masks_list, _, _ = model.eval(
            [img[t] for t in frames],
            channels=None,
            rescale=None,
            mask_threshold=sc["mask_threshold"],
            flow_threshold=sc["flow_threshold"],
            num_workers=0,
            transparency=True,
            omni=True,
            verbose=0,
            affinity_seg=1 if sc.get("affinity_seg", True) else 0,
            batch_size=os.cpu_count() or 1,
        )

    # Keep a full-length array even in preview so frame indices stay meaningful.
    stack = np.zeros((T,) + img.shape[1:], dtype=np.uint16)
    for t, m in zip(frames, masks_list):
        mm = np.asarray(m)
        if mm.max() > np.iinfo(np.uint16).max:
            die(
                f"frame {t} produced more than 65535 cells, which is far more "
                f"than expected.",
                "MASK_THRESHOLD is probably far too low — raise it in config.sh.",
            )
        stack[t] = mm.astype(np.uint16)

    # Count real objects, not the largest label. stack[t].max() is only the
    # cell count while every label from 1..max is present; after any gap in
    # the labelling it silently overcounts.
    def frame_stats(t):
        lbl = stack[t]
        if not lbl.any():
            return 0, 0.0, 0.0
        _, areas = np.unique(lbl[lbl > 0], return_counts=True)
        return (int(areas.size), float(np.median(areas)), float((lbl > 0).mean()))

    stats_per_frame = {t: frame_stats(t) for t in frames}
    counts = np.array([stats_per_frame[t][0] for t in frames])
    log.info(
        f"cells: first={counts[0]}  last={counts[-1]}  "
        f"median={np.median(counts):.0f}"
    )
    if counts.min() == 0:
        log.warning(
            f"{int((counts == 0).sum())} frame(s) segmented to zero "
            f"cells — check mask_threshold"
        )

    np.savez_compressed(
        out / "masks.npz", masks=stack, segmented_frames=np.array(frames)
    )
    with open(out / "cells_per_frame.csv", "w") as fh:
        fh.write(
            "frame,frame_fiji,time_min,n_cells,median_cell_area_px," "area_fraction\n"
        )
        for t in frames:
            n, med, frac = stats_per_frame[t]
            tm = "" if dt_min is None else f"{t * dt_min:.2f}"
            fh.write(f"{t},{t + 1},{tm},{n},{med:.0f},{frac:.4f}\n")

    # A binary layer to drop on top of the GFP stack in Fiji, so the outlines
    # can be checked against the signal — two cells segmented as one, or one
    # filament split in two, are obvious there and invisible in a count.
    if seg_cfg_bool("SEGMENT_WRITE_OUTLINES", True):
        from skimage.segmentation import find_boundaries

        outl = np.zeros(stack.shape, np.uint8)
        for t in frames:
            if stack[t].any():
                outl[t][find_boundaries(stack[t], mode="inner")] = 255
        tifffile.imwrite(out / "segmentation_outlines.tif", outl)
        log.info(
            "segmentation_outlines.tif — binary, same size as the movie, "
            "for overlaying on the GFP stack"
        )

    if not preview:
        combine_counts(cfg, log)

    qc_frames = frames if preview else sorted({0, T // 3, 2 * T // 3, T - 1})
    qc_figure(img, stack, qc_frames, out / "qc_segmentation.png", log)

    write_meta(
        out,
        "02_segment",
        params,
        {"bf": src},
        {
            "n_frames_segmented": len(frames),
            "preview": bool(preview),
            "median_cells": float(np.median(counts)),
            "first_frame_cells": int(counts[0]),
            "last_frame_cells": int(counts[-1]),
        },
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "segment"))

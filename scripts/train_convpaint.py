#!/usr/bin/env python3
"""
Train one convpaint classifier across every annotated cell stack.

    # 1. look at what the normalisation does before committing to it
    python tools/train_convpaint.py --dir .../07_training/_unsorted --check-contrast

    # 2. annotate in napari, saving each mask beside its image
    # 3. train on everything at once
    python tools/train_convpaint.py --dir .../07_training/_unsorted --out model.pkl

    # 4. apply it to every stack
    python tools/train_convpaint.py --dir .../07_training/_unsorted \\
           --apply model.pkl --out-dir .../segmented

Annotations
-----------
For each cell_XXXXX.tif, save the labels next to it as one of

    cell_XXXXX_labels.tif      cell_XXXXX_annot.tif      cell_XXXXX_mask.tif

with 0 meaning "not annotated" and 1, 2, 3 ... meaning your classes. In napari
that is the labels layer: File > Save selected layer. Anything without a
matching labels file is skipped, so you can annotate a few and add more later.

Contrast
--------
Convpaint feeds the image to a pretrained network, which expects values in a
sensible range. A very bright cell would otherwise dominate. Three options:

    per-cell    each stack scaled to its own percentiles.  Best for learning
                SHAPE — every cell arrives at comparable brightness, so the
                classifier has to use structure. But it destroys absolute
                brightness, so "no GFP" and "dim, even GFP" become identical.
    global      one range across every stack.  Keeps cells comparable to each
                other, so faint and bright cells stay distinguishable.
    none        raw counts.

Default is global, because it is the one that does not quietly throw away a
distinction you may care about. Use --check-contrast to see the difference
before deciding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LABEL_SUFFIXES = ("_labels", "_annot", "_annotations", "_mask")


def stub_gui():
    """convpaint's package import pulls in Qt; not needed to train."""
    import types

    for name in ("napari_convpaint.convpaint_widget",):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.ConvpaintWidget = object
            sys.modules[name] = m


def fail(problem, fix=""):
    print("")
    print("===========================================================")
    print("  STOPPED")
    print("===========================================================")
    print(f"  Problem : {problem}")
    if fix:
        print(f"  Fix     : {fix}")
    print("")
    sys.exit(1)


def find_pairs(folder, log=print):
    """Match each cell stack with its annotation file, if there is one."""
    import tifffile

    folder = Path(folder)
    if not folder.is_dir():
        fail(
            f"folder not found: {folder}",
            "Point --dir at the _unsorted folder written by export_cells.py.",
        )

    stacks = [
        f
        for f in sorted(folder.glob("*.tif"))
        if not any(s in f.stem for s in LABEL_SUFFIXES)
    ]
    if not stacks:
        fail(
            f"no cell stacks found in {folder}",
            "Run  python tools/export_cells.py -c <config> --stacks  first.",
        )

    pairs, lonely = [], []
    for img in stacks:
        lab = None
        for suffix in LABEL_SUFFIXES:
            cand = img.with_name(img.stem + suffix + img.suffix)
            if cand.exists():
                lab = cand
                break
        (pairs.append((img, lab)) if lab else lonely.append(img))
    log(f"{len(stacks)} cell stack(s), {len(pairs)} with annotations")
    if lonely:
        log(f"  {len(lonely)} not yet annotated, skipped " f"(e.g. {lonely[0].name})")
    return pairs, lonely, stacks


def load_stack(path, channel=0):
    """The image channel of a stack, as (T, Y, X) or (Y, X)."""
    import tifffile

    a = tifffile.imread(path)
    if a.ndim == 4:  # (T, C, Y, X)
        return a[:, channel]
    if a.ndim == 3:  # (C, Y, X) or (T, Y, X)
        return a[channel] if a.shape[0] <= 4 else a
    return a


def scale_for_features(arr, mode, lo=None, hi=None):
    """
    Put an image into a range the pretrained network can work with.

    Returns float32 in 0-1 plus the range used, so the same scaling can be
    reapplied later. Reapplying the SAME range at prediction time matters:
    a classifier trained on one scaling and applied under another is being
    shown different data.
    """
    a = arr.astype(np.float32)
    if mode == "none":
        return a, (float(a.min()), float(a.max()))
    if lo is None or hi is None:
        vals = a[a > 0]  # background is exactly 0
        if vals.size < 10:
            vals = a.ravel()
        lo, hi = np.percentile(vals, [1, 99.5])
    hi = max(float(hi), float(lo) + 1)
    out = np.clip((a - lo) / (hi - lo), 0, 1)
    return out.astype(np.float32), (float(lo), float(hi))


def global_range(stacks, channel, log=print):
    """One display range across every cell, from in-cell pixels only."""
    vals = []
    for f in stacks:
        a = load_stack(f, channel)
        v = a[a > 0]
        if v.size:
            vals.append(
                np.random.default_rng(0).choice(
                    v, size=min(v.size, 20000), replace=False
                )
            )
    if not vals:
        fail("every stack is empty.", "Check that export_cells.py wrote real data.")
    allv = np.concatenate(vals)
    lo, hi = np.percentile(allv, [1, 99.5])
    log(f"global range {lo:.0f}-{hi:.0f} counts, from {len(stacks)} cells")
    return float(lo), float(hi)


def check_contrast(stacks, channel, out_path, log=print):
    """Show what each normalisation choice does, before committing to one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    picks = stacks[:8]
    lo_g, hi_g = global_range(stacks, channel, log)

    fig, axes = plt.subplots(
        3, len(picks), figsize=(2.1 * len(picks), 7), squeeze=False
    )
    for j, f in enumerate(picks):
        a = load_stack(f, channel)
        frame = a[len(a) // 2] if a.ndim == 3 else a
        for i, (mode, kw, label) in enumerate(
            [
                ("none", {}, "raw"),
                ("global", {"lo": lo_g, "hi": hi_g}, "global"),
                ("per-cell", {}, "per-cell"),
            ]
        ):
            img, _ = scale_for_features(
                frame, "none" if mode == "none" else "clip", **kw
            )
            if mode == "none":
                img = frame
            axes[i][j].imshow(img, cmap="Greens_r" if mode == "none" else "gray")
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])
            if j == 0:
                axes[i][j].set_ylabel(label, fontsize=10)
        axes[0][j].set_title(f.stem.replace("cell_", "")[:14], fontsize=7)

    fig.suptitle(
        "What each contrast choice does.\\n"
        "global keeps dim cells dim — per-cell makes every cell look "
        "equally bright, which hides 'no signal' vs 'dim signal'.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log(f"contrast comparison -> {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="folder of cell stacks from export_cells.py --stacks",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=0,
        help="which channel of each stack to use (default 0, GFP)",
    )
    ap.add_argument(
        "--contrast", default="global", choices=("global", "per-cell", "none")
    )
    ap.add_argument(
        "--check-contrast",
        action="store_true",
        help="write a comparison figure and stop",
    )
    ap.add_argument(
        "--out", default="convpaint_model.pkl", help="where to save the trained model"
    )
    ap.add_argument(
        "--apply", default="", help="apply a saved model instead of training"
    )
    ap.add_argument("--out-dir", default="", help="where predictions go, with --apply")
    ap.add_argument(
        "--fe",
        default="",
        help="feature extractor, e.g. vgg16. Leave empty for the " "convpaint default.",
    )
    ap.add_argument(
        "--use-rf",
        action="store_true",
        help="random forest instead of the default classifier",
    )
    args = ap.parse_args()

    log = print
    folder = Path(args.dir).expanduser()
    pairs, lonely, stacks = find_pairs(folder, log)

    if args.check_contrast:
        check_contrast(stacks, args.channel, folder / "contrast_check.png", log)
        log("")
        log(
            "Pick with --contrast: global keeps faint and bright cells "
            "distinguishable; per-cell forces the classifier onto shape."
        )
        return 0

    lo = hi = None
    if args.contrast == "global":
        lo, hi = global_range(stacks, args.channel, log)

    stub_gui()
    try:
        from napari_convpaint.convpaint_model import ConvpaintModel
    except ImportError:
        fail(
            "napari-convpaint is not installed in this python: " + sys.executable,
            "conda activate phage_pipeline && pip install napari-convpaint",
        )

    import tifffile

    # ── apply ──────────────────────────────────────────────────────────────
    if args.apply:
        model_path = Path(args.apply).expanduser()
        if not model_path.exists():
            fail(f"model not found: {model_path}", "Train one first, without --apply.")
        out_dir = Path(args.out_dir or folder / "segmented").expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        cp = ConvpaintModel(model_path=str(model_path))
        log(f"applying {model_path.name} to {len(stacks)} stack(s)")
        for f in stacks:
            a = load_stack(f, args.channel)
            img, _ = scale_for_features(
                a, "clip" if args.contrast != "none" else "none", lo, hi
            )
            seg = cp.segment(img)
            tifffile.imwrite(
                out_dir / f"{f.stem}_seg.tif", np.asarray(seg).astype(np.uint16)
            )
        log(f"predictions -> {out_dir}")
        return 0

    # ── train ──────────────────────────────────────────────────────────────
    if not pairs:
        fail(
            "none of the stacks have annotations yet.",
            "Open a stack in napari, add a Labels layer, paint a few pixels "
            "of each class, and save it beside the image as\n"
            "            cell_XXXXX_labels.tif",
        )

    images, annots, ids = [], [], []
    for img_path, lab_path in pairs:
        a = load_stack(img_path, args.channel)
        ann = tifffile.imread(lab_path)
        img, _ = scale_for_features(
            a, "clip" if args.contrast != "none" else "none", lo, hi
        )
        if ann.shape != img.shape:
            # a labels layer drawn on one frame of a stack
            if ann.ndim == img.ndim - 1 and ann.shape == img.shape[1:]:
                full = np.zeros(img.shape, ann.dtype)
                full[0] = ann
                ann = full
                log(f"  {lab_path.name}: single-frame labels, applied to " f"frame 0")
            else:
                log(
                    f"  SKIPPED {lab_path.name}: labels are {ann.shape} but "
                    f"the image is {img.shape}"
                )
                continue
        if not (ann > 0).any():
            log(f"  SKIPPED {lab_path.name}: nothing painted in it")
            continue
        images.append(img)
        annots.append(ann.astype(np.uint8))
        ids.append(img_path.stem)

    if not images:
        fail(
            "no usable annotations were found.",
            "Each labels file must have the same height and width as its "
            "image, with at least some pixels painted.",
        )

    classes = sorted({int(v) for ann in annots for v in np.unique(ann) if v > 0})
    per_class = {c: int(sum((ann == c).sum() for ann in annots)) for c in classes}
    log(f"training on {len(images)} annotated cell(s)")
    log(f"  classes found: " + "  ".join(f"{c}: {n} px" for c, n in per_class.items()))
    if len(classes) < 2:
        fail(
            f"only class {classes} was painted, so there is nothing to tell " f"apart.",
            "Paint at least two classes — for example 1 for background "
            "inside the cell and 2 for the structure.",
        )
    thin = [c for c, n in per_class.items() if n < 50]
    if thin:
        log(
            f"  WARNING: very few pixels for class(es) {thin}. Paint more of "
            f"them or that class will be learned badly."
        )

    kwargs = {}
    if args.fe:
        kwargs["fe_name"] = args.fe
    cp = ConvpaintModel(**kwargs) if kwargs else ConvpaintModel(alias="vgg")
    log(f"  feature extractor: {args.fe or 'vgg (convpaint default)'}")
    cp.train(images, annots, img_ids=ids, use_rf=args.use_rf, allow_writing_files=False)

    out = Path(args.out).expanduser()
    cp.save(str(out))
    log(f"model saved -> {out}")
    log("")
    log("apply it to every cell with:")
    log(f"    python tools/train_convpaint.py --dir {folder} " f"--apply {out}")
    log("")
    log(
        f"NOTE: it was trained on {args.contrast}-scaled images. Applying it "
        f"to differently scaled data will not work properly, so keep "
        f"--contrast the same."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

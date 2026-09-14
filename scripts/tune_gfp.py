#!/usr/bin/env python3
"""
Tune the GFP phenotype calls on a few frames, before running the whole movie.

    python tools/tune_gfp.py -c configs/my.sh --frames "0 30 60"
    python tools/tune_gfp.py -c configs/my.sh --frames 0 --sweep GFP_RIDGE_K=2,3,4,6
    python tools/tune_gfp.py -c configs/my.sh --frames 0 --score

Everything is written to <pos>/06_gfp_tuning/ and nothing there is read by
the pipeline, so this cannot disturb a finished analysis.

What comes out
--------------
contact_sheet.png    every cell in the chosen frames, grouped by the call it
                     got, with its measurements underneath. This is the one
                     to look at first: a phenotype that is being missed shows
                     up as a row of cells filed under the wrong heading.
masks.tif            binary layers to open in Fiji over the GFP movie:
                     structures / filaments / puncta / cell outlines
labels.csv           one row per cell, with a blank column to fill in by hand
sweep.png            (with --sweep) how the counts move as one setting varies

Suggested loop
--------------
1. run with no options and read contact_sheet.png
2. fill in the your_call column of labels.csv for a few dozen cells
3. run with --score to get precision and recall per phenotype
4. run with --sweep on whichever setting looks wrong, then repeat step 3
5. put the chosen values in the config and run the pipeline
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# The scripts may sit beside this file or one level up, depending on where the
# tool was copied to. Look in both rather than assuming.
_CANDIDATES = [HERE, HERE.parent / "scripts", HERE.parent]
_SCRIPTS = next((d for d in _CANDIDATES if (d / "06_gfp_structures.py").exists()), None)
if _SCRIPTS is None:
    sys.exit(
        "Cannot find 06_gfp_structures.py next to this tool.\n"
        "Keep tune_gfp.py in the pipeline folder, beside scripts/."
    )
sys.path.insert(0, str(_SCRIPTS))

from common import die, get_logger, load_cell_names, load_config, require, step_dir

_spec = importlib.util.spec_from_file_location(
    "gfp6", _SCRIPTS / "06_gfp_structures.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def check_plotting():
    """
    Fail now, not after the detection has run.

    Running the tool as ./tune_gfp.py uses whatever python is first on PATH,
    which is often not the analysis environment. That python can have numpy
    and skimage but no matplotlib, so the work completes and then the figure
    stage falls over.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        die(
            "matplotlib is not installed in the python being used: "
            f"{sys.executable}",
            "Activate the analysis environment first, then run the tool with "
            "that python:\n"
            "                conda activate phage_pipeline\n"
            "                python tools/tune_gfp.py -c <config> ...\n"
            "            Running ./tune_gfp.py directly uses the system "
            "python, which is usually not the one with the analysis packages. "
            "If this really is the right environment:  pip install matplotlib",
        )


ORDER = ["none", "diffuse", "punctate", "filamentous", "mixed"]


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


# ── detection over a few frames ─────────────────────────────────────────────
def run_frames(gfp, tracked, frames, px_um, dt_min, p, name_of):
    """Every cell in the chosen frames, with its crop kept for display."""
    from skimage import measure

    cells, masks = [], {
        "structures": np.zeros(gfp.shape, np.uint8),
        "filaments": np.zeros(gfp.shape, np.uint8),
        "puncta": np.zeros(gfp.shape, np.uint8),
    }
    for t in frames:
        lbl = tracked[t]
        if not lbl.any():
            continue
        bg = float(np.median(gfp[t][lbl == 0]))
        sigma = G.pixel_noise(gfp[t], lbl == 0)
        for reg in measure.regionprops(lbl):
            y0, x0, y1, x1 = reg.bbox
            cm = lbl[y0:y1, x0:x1] == reg.label
            crop = gfp[t][y0:y1, x0:x1]
            feat, fmask, pts, rel, smask = G.measure_cell(crop, cm, bg, sigma, px_um, p)
            state = G.state_of(feat, p["min_signal_over_bg"], p["min_struct_area_frac"])
            if smask is not None and smask.any():
                masks["structures"][t, y0:y1, x0:x1][smask] = 255
            if fmask.any():
                masks["filaments"][t, y0:y1, x0:x1][fmask] = 255
            for y, x in pts:
                masks["puncta"][t, y0 + y, x0 + x] = 255
            cells.append(
                {
                    "frame": t,
                    "track_id": int(reg.label),
                    "cell_name": name_of(reg.label),
                    "state": state,
                    "crop": np.asarray(crop),
                    "cell_mask": cm,
                    "fil": fmask,
                    "pts": pts,
                    "feat": feat,
                    "bbox": (y0, x0, y1, x1),
                }
            )
    return cells, masks


# ── the contact sheet ───────────────────────────────────────────────────────
def contact_sheet(cells, p, path, log, per_state=12):
    """
    Every cell filed under the call it got, with its boundary drawn.

    Two things this has to get right, both of which it got wrong before:

    * The boundary is drawn, and everything outside the cell is dimmed. A
      bounding box usually contains parts of the neighbours, and their foci
      are not this cell's foci.
    * Brightness is scaled from the cell's own pixels against a range shared
      across the whole sheet, so a dim cell looks dim. Stretching each crop to
      its own percentiles — which this did before — makes an empty cell look
      full of texture, which is exactly the mistake this pipeline exists to
      avoid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_state = defaultdict(list)
    for c in cells:
        by_state[c["state"]].append(c)
    for k in by_state:
        by_state[k].sort(key=lambda c: -c["feat"].get("fil_length_um", 0))

    present = [s for s in ORDER if by_state[s]]
    if not present:
        log.warning("no cells to show")
        return

    # One display range for the whole sheet, from in-cell pixels only.
    inside = np.concatenate(
        [c["crop"][c["cell_mask"]].ravel() for c in cells if c["cell_mask"].any()]
    )
    lo, hi = np.percentile(inside, [1, 99.5])
    hi = max(hi, lo + 1)
    log.info(f"display range {lo:.0f}-{hi:.0f} counts, shared by every panel")

    ncol = min(per_state, max(len(by_state[s]) for s in present))
    fig, axes = plt.subplots(
        len(present), ncol, figsize=(2.1 * ncol, 2.8 * len(present)), squeeze=False
    )

    for r, state in enumerate(present):
        group = by_state[state][:ncol]
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(group):
                ax.axis("off")
                continue
            cell = group[c]
            cm = cell["cell_mask"]
            disp = np.clip((cell["crop"].astype(float) - lo) / (hi - lo), 0, 1)

            rgb = np.zeros(disp.shape + (3,))
            rgb[..., 1] = disp
            rgb[~cm] *= 0.28  # neighbours dimmed, not hidden
            rgb[cell["fil"]] = (1.0, 0.2, 1.0)
            ax.imshow(rgb, interpolation="nearest")

            ax.contour(cm.astype(float), levels=[0.5], colors="white", linewidths=0.8)
            for y, x in cell["pts"]:
                ax.plot(x, y, "o", mfc="none", mec="cyan", ms=8, mew=1.3)

            f = cell["feat"]
            ax.set_xlabel(
                f"{cell['cell_name']} f{cell['frame']}\n"
                f"cover {100 * f.get('gfp_area_frac', 0):.0f}%  "
                f"struct {100 * f.get('struct_area_frac', 0):.0f}%\n"
                f"round {f.get('shape_circularity', 0):.2f}  "
                f"elong {f.get('shape_elongation', 0):.1f}  sig{f['signal_over_bg']:.0f}",
                fontsize=6,
            )
        axes[r][0].set_ylabel(
            f"{state}\n(n={len(by_state[state])})",
            fontsize=10,
            color=G.STATE_COLOURS.get(state, "k"),
        )

    fig.suptitle(
        "Every cell filed under the call it got.  White line = this "
        "cell's boundary; anything dimmed is a neighbour.\n"
        "Magenta = filament, cyan = punctum. Brightness is on one "
        "shared scale, so a dim cell looks dim.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(
        f"contact sheet -> {path.name}  "
        + "  ".join(f"{s}:{len(by_state[s])}" for s in present)
    )


# ── sweep ───────────────────────────────────────────────────────────────────
def parse_sweep(text):
    if "=" not in text:
        die(
            f"--sweep should look like NAME=v1,v2,v3 but was '{text}'.",
            "For example  --sweep GFP_RIDGE_K=2,3,4,6",
        )
    name, values = text.split("=", 1)
    try:
        return name.strip(), [float(v) for v in values.split(",") if v.strip()]
    except ValueError:
        die(
            f"the values in --sweep must be numbers, got '{values}'.",
            "For example  --sweep GFP_FIL_MAX_WIDTH_UM=0.3,0.45,0.6",
        )


SWEEP_KEYS = {
    "GFP_STRUCT_MIN_CONTRAST": "min_contrast",
    "GFP_STRUCT_NOISE_K": "noise_k",
    "GFP_RIDGE_K": "ridge_k",
    "GFP_FIL_MIN_LENGTH_UM": "fil_min_length_um",
    "GFP_FIL_MAX_WIDTH_UM": "fil_max_width_um",
    "GFP_FIL_MIN_ASPECT": "fil_min_aspect",
    "GFP_MIN_SIGNAL_OVER_BG": "min_signal_over_bg",
    "GFP_PUNCTA_ON_FILAMENT_RATIO": "punct_on_filament_ratio",
}


def sweep(
    gfp, tracked, frames, px_um, dt_min, base_p, name_of, setting, values, path, log
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = SWEEP_KEYS.get(setting)
    if key is None:
        die(
            f"'{setting}' cannot be swept.",
            "Sweepable settings: " + ", ".join(sorted(SWEEP_KEYS)),
        )

    counts = {s: [] for s in ORDER}
    for v in values:
        p = dict(base_p)
        p[key] = v
        cells, _ = run_frames(gfp, tracked, frames, px_um, dt_min, p, name_of)
        n = max(len(cells), 1)
        for s in ORDER:
            counts[s].append(100 * sum(1 for c in cells if c["state"] == s) / n)
        log.info(
            f"  {setting}={v:g}  "
            + "  ".join(f"{s} {counts[s][-1]:.0f}%" for s in ORDER)
        )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in ORDER:
        ax.plot(
            values, counts[s], marker="o", lw=2, color=G.STATE_COLOURS.get(s), label=s
        )
    ax.axvline(base_p[key], color="k", ls=":", lw=1, label=f"current ({base_p[key]:g})")
    ax.set(
        xlabel=setting,
        ylabel="% of cells",
        title=f"How the calls move with {setting}\n" f"({len(frames)} frame(s))",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"sweep -> {path.name}")
    log.info(
        "A setting worth using has a plateau: if every value changes the "
        "answer, the phenotypes are not separated by that setting."
    )


def labels_from_folders(train_dir, out, log):
    """
    Read the label of each cell from the folder it was dragged into.

    tools/export_cells.py writes one image per cell into _unsorted/ next to
    empty folders named after the phenotypes. Whatever ends up in a named
    folder carries that label; whatever is left in _unsorted/ is ignored, so
    a cell you were not sure about simply does not vote.
    """
    import csv as _csv

    feats = train_dir / "features.csv"
    if not feats.exists():
        die(
            f"no exported cells found at {train_dir}",
            'Run  python tools/export_cells.py -c <config> --frames "0 15 30" '
            "first, then sort the images into the folders.",
        )
    by_file = {r["file"]: r for r in _csv.DictReader(open(feats))}

    rows, counts = [], defaultdict(int)
    for cls in ORDER:
        folder = train_dir / cls
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.name.startswith("."):
                continue
            rec = by_file.get(f.name)
            if rec is None:
                # a renamed or converted file cannot be matched back
                log.warning(
                    f"'{f.name}' is in {cls}/ but is not in "
                    f"features.csv — keep the file names as exported"
                )
                continue
            rec = dict(rec)
            rec["your_call"] = cls
            rec["state"] = rec.get("pipeline_call", "")
            rows.append(rec)
            counts[cls] += 1

    if not rows:
        die(
            "no images have been sorted into the phenotype folders yet.",
            f"Drag files from {train_dir / '_unsorted'} into the folders "
            f"beside it, then run this again.",
        )

    left = (
        len(list((train_dir / "_unsorted").glob("*.png")))
        if (train_dir / "_unsorted").is_dir()
        else 0
    )
    log.info(
        "sorted so far: "
        + "  ".join(f"{k} {v}" for k, v in counts.items())
        + f"   ({left} still unsorted, ignored)"
    )

    path = out / "labels_from_folders.csv"
    out.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ── learn the rule from hand labels ─────────────────────────────────────────
MIN_PER_CLASS = 15

# Which config setting each feature corresponds to, where one exists. Used to
# turn what the tree learned into something you can actually set.
FEATURE_TO_CONFIG = {
    "signal_over_bg": "GFP_MIN_SIGNAL_OVER_BG",
    "shape_circularity": "GFP_FIL_MAX_CIRCULARITY",
    "shape_elongation": "GFP_FIL_MIN_ELONGATION",
    "fil_length_um": "GFP_FIL_MIN_LENGTH_UM",
    "fil_peak_ratio": "GFP_PUNCTA_ON_FILAMENT_RATIO",
    "punct_prominence": "GFP_PUNCTA_MIN_PROMINENCE",
    "struct_area_frac": "GFP_MIN_STRUCT_AREA_FRAC",
    "rel_p95": "GFP_STRUCT_MIN_CONTRAST",
}

TRAIN_FEATURES = [
    "signal_over_bg",
    "gfp_area_frac",
    "struct_area_frac",
    "shape_circularity",
    "shape_elongation",
    "shape_solidity",
    "fil_length_um",
    "fil_aspect",
    "fil_peak_ratio",
    "punct_prominence",
    "rel_p95",
    "ridge_area_frac",
]


def train(labels_csv, out, log, save_model=False):
    """
    Fit the classification to hand labels instead of setting thresholds.

    A decision tree is used on purpose: it can be read. The rules it learns
    are printed, so the result stays a set of understandable cutoffs rather
    than something that cannot be checked or reported.
    """
    import csv as _csv

    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        die(
            "scikit-learn is not installed in this python: " + sys.executable,
            "conda activate phage_pipeline && pip install scikit-learn",
        )

    rows = [
        r
        for r in _csv.DictReader(open(labels_csv))
        if (r.get("your_call") or "").strip() in ORDER
    ]
    if len(rows) < 20:
        die(
            f"only {len(rows)} hand labels found — too few to learn from.",
            "Fill in the your_call column for at least 20 cells, ideally "
            "several of each phenotype, then run --train again.",
        )

    counts = defaultdict(int)
    for r in rows:
        counts[r["your_call"]] += 1
    log.info("labels: " + "  ".join(f"{k} {v}" for k, v in counts.items()))
    thin = [k for k, v in counts.items() if v < MIN_PER_CLASS]
    if thin:
        log.warning(f"fewer than {MIN_PER_CLASS} examples of: " f"{', '.join(thin)}.")
        log.warning(
            "  A class with a handful of examples cannot be learned. "
            "Whatever the tree says about it is an accident of those "
            "few cells, so do not put those branches in the config."
        )

    feats = [f for f in TRAIN_FEATURES if f in rows[0]]
    X = np.array([[float(r.get(f) or 0) for f in feats] for r in rows])
    y = np.array([r["your_call"] for r in rows])

    tree = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=0
    )
    n_split = min(5, min(counts.values()))
    if n_split >= 2:
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
        pred = cross_val_predict(tree, X, y, cv=cv)
        acc = float((pred == y).mean())
        log.info(
            f"cross-validated accuracy {acc:.2f} "
            f"({n_split}-fold, held-out cells only)"
        )
        log.info("truth \\ predicted   " + "".join(f"{s[:5]:>8s}" for s in ORDER))
        for truth_lbl in ORDER:
            if not (y == truth_lbl).any():
                continue
            line = f"{truth_lbl:18s}"
            for called in ORDER:
                line += f"{int(((y == truth_lbl) & (pred == called)).sum()):8d}"
            log.info(line)

        log.info("")
        log.info("how well each phenotype is actually learned:")
        useless = []
        for cls in ORDER:
            n_cls = int((y == cls).sum())
            if n_cls == 0:
                continue
            rec = float((pred[y == cls] == cls).mean())
            flag = ""
            if rec < 0.5:
                flag = "  <-- NOT LEARNED"
                useless.append(cls)
            log.info(f"  {cls:13s} n={n_cls:4d}  recall {rec:.2f}{flag}")
        if useless:
            log.warning("")
            log.warning(f"the model cannot recognise: {', '.join(useless)}.")
            log.warning(
                "  Overall accuracy hides this — it is carried by the "
                "classes with many examples. Label more of the missing "
                "ones before using any rule the tree gives for them."
            )
    else:
        log.warning("not enough examples per class to cross-validate")

    tree.fit(X, y)
    if save_model:
        try:
            import joblib

            joblib.dump(
                {"model": tree, "features": feats, "classes": list(tree.classes_)},
                out / "gfp_model.joblib",
            )
            log.info(f"model saved -> {out / 'gfp_model.joblib'}")
            log.info("  to use it instead of thresholds, set in config.sh:")
            log.info('    GFP_CLASSIFIER="model"')
            log.info(f'    GFP_MODEL_PATH="{out / "gfp_model.joblib"}"')
            log.warning(
                "  a saved model only applies to data like the cells "
                "it was trained on — same strain, stain and settings. "
                "Thresholds travel better between experiments."
            )
        except ImportError:
            die(
                "joblib is not installed, so the model cannot be saved.",
                "conda activate phage_pipeline && pip install joblib",
            )
    rules = export_text(tree, feature_names=feats, max_depth=4)
    (out / "learned_rules.txt").write_text(rules)
    log.info("")
    log.info("the rule it learned (also in learned_rules.txt):")
    for line in rules.splitlines()[:25]:
        log.info("  " + line)

    # A split whose two branches give the same answer decides nothing; it is
    # noise the tree fitted, and worth flagging so it is not copied out.
    dead = _dead_splits(tree, feats)
    if dead:
        log.warning("")
        log.warning(
            "these splits lead to the same answer on both sides, so "
            "they decide nothing — ignore them:"
        )
        for d in dead:
            log.warning(f"    {d}")

    order = np.argsort(tree.feature_importances_)[::-1]
    log.info("")
    log.info("what it actually used:")
    for i in order[:6]:
        if tree.feature_importances_[i] > 0.01:
            cfg_name = FEATURE_TO_CONFIG.get(feats[i], "(no setting)")
            log.info(
                f"  {feats[i]:22s} {tree.feature_importances_[i]:.2f}" f"   {cfg_name}"
            )

    log.info("")
    log.info("suggested config values, from the top split on each feature:")
    any_map = False
    for i in order:
        if tree.feature_importances_[i] <= 0.01:
            continue
        cfg_name = FEATURE_TO_CONFIG.get(feats[i])
        if not cfg_name:
            continue
        thr = _first_threshold(tree, feats, feats[i])
        if thr is None:
            continue
        log.info(f"    {cfg_name}={thr:.4g}")
        any_map = True
    if not any_map:
        log.info("    (none of the features it used map to a setting)")
    log.info("")
    log.info(
        "Only copy across values for phenotypes the model actually "
        "learned. To use the model itself instead of thresholds, see "
        "--save-model."
    )


def _dead_splits(tree, feats):
    """Splits whose subtrees both predict the same class."""
    tt = tree.tree_
    out = []

    def leaf_classes(node):
        if tt.children_left[node] == -1:
            return {int(np.argmax(tt.value[node][0]))}
        return leaf_classes(tt.children_left[node]) | leaf_classes(
            tt.children_right[node]
        )

    def walk(node):
        if tt.children_left[node] == -1:
            return
        if len(leaf_classes(node)) == 1:
            out.append(f"{feats[tt.feature[node]]} <= " f"{tt.threshold[node]:.4g}")
            return
        walk(tt.children_left[node])
        walk(tt.children_right[node])

    walk(0)
    return out


def _first_threshold(tree, feats, name):
    """The threshold of the highest split that uses this feature."""
    tt = tree.tree_
    idx = feats.index(name)
    best, best_depth = None, 1e9

    def walk(node, depth):
        nonlocal best, best_depth
        if tt.children_left[node] == -1:
            return
        if tt.feature[node] == idx and depth < best_depth:
            best, best_depth = float(tt.threshold[node]), depth
        walk(tt.children_left[node], depth + 1)
        walk(tt.children_right[node], depth + 1)

    walk(0, 0)
    return best


# ── scoring ─────────────────────────────────────────────────────────────────
def score(labels_csv, log):
    import csv as _csv

    rows = [
        r
        for r in _csv.DictReader(open(labels_csv))
        if (r.get("your_call") or "").strip() in ORDER
    ]
    if not rows:
        die(
            "no hand labels found in labels.csv.",
            "Fill in the your_call column with one of: " + ", ".join(ORDER),
        )

    log.info(f"scored against {len(rows)} hand-labelled cells")
    header = "truth \\ called   " + "".join(f"{s[:5]:>8s}" for s in ORDER)
    log.info(header)
    for truth in ORDER:
        line = f"{truth:15s}"
        for called in ORDER:
            n = sum(1 for r in rows if r["your_call"] == truth and r["state"] == called)
            line += f"{n:8d}"
        if any(r["your_call"] == truth for r in rows):
            log.info(line)

    log.info("")
    for s in ORDER:
        tp = sum(1 for r in rows if r["state"] == s and r["your_call"] == s)
        fp = sum(1 for r in rows if r["state"] == s and r["your_call"] != s)
        fn = sum(1 for r in rows if r["state"] != s and r["your_call"] == s)
        if tp + fp + fn == 0:
            continue
        log.info(
            f"  {s:12s} precision {tp / max(tp + fp, 1):.2f}   "
            f"recall {tp / max(tp + fn, 1):.2f}   (n={tp + fn})"
        )


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-p", "--position", type=int, default=0)
    ap.add_argument(
        "--frames", default="0", help='which frames, e.g. "0 30 60". Default: frame 0.'
    )
    ap.add_argument(
        "--sweep", default="", help="NAME=v1,v2,v3 — try one setting at several values"
    )
    ap.add_argument(
        "--score", action="store_true", help="score the filled-in labels.csv and stop"
    )
    ap.add_argument(
        "--train", action="store_true", help="learn the rule from hand labels"
    )
    ap.add_argument(
        "--save-model",
        action="store_true",
        help="also save the trained model, to use instead of " "thresholds",
    )
    ap.add_argument(
        "--labels-from-folders",
        action="store_true",
        help="take labels from how files were sorted in "
        "07_training/, instead of labels.csv",
    )
    args = ap.parse_args()

    load_shell_config(args.config)
    log = get_logger("tune", False)
    check_plotting()
    cfg = load_config()
    p = dict(cfg["gfp"])

    out = step_dir(cfg, args.position, "06_gfp").with_name("06_gfp_tuning")
    out.mkdir(parents=True, exist_ok=True)

    if args.score or args.train:
        if args.labels_from_folders:
            train_dir = step_dir(cfg, args.position, "03_track").with_name(
                "07_training"
            )
            lab = labels_from_folders(train_dir, out, log)
        else:
            lab = require(
                out / "labels.csv",
                "the labels file",
                "Run this tool with no options first, then fill in "
                "the your_call column.",
            )
        (train(lab, out, log, args.save_model) if args.train else score(lab, log))
        return 0

    import tifffile

    load_dir = step_dir(cfg, args.position, "01_load")
    track_dir = step_dir(cfg, args.position, "03_track")
    gfp = tifffile.imread(
        require(
            load_dir / "gfp.tif",
            "the GFP channel",
            "Set LOAD_ND2=TRUE and run the pipeline.",
        )
    )
    tracked = np.load(
        require(
            track_dir / "tracked_masks.npz",
            "the tracked masks",
            "Set TRACK_CELLS=TRUE and run the pipeline.",
        )
    )["masks"]

    px_um, dt_min = 0.065, 1.0
    acq = load_dir / "acquisition.json"
    if acq.exists():
        d = json.load(open(acq))
        px_um = float(d.get("pixel_size_um") or px_um)
        dt_min = float(d.get("frame_interval_min") or dt_min)

    lin = track_dir / "lineage.csv"
    names = load_cell_names(lin)[0] if lin.exists() else {}
    name_of = lambda t: names.get(int(t), str(int(t)))

    frames = [int(f) for f in args.frames.replace(",", " ").split()]
    bad = [f for f in frames if not 0 <= f < gfp.shape[0]]
    if bad:
        die(
            f"frame(s) {bad} are outside this movie, which has "
            f"{gfp.shape[0]} frames (0 to {gfp.shape[0] - 1}).",
            "Pick frames inside that range with --frames.",
        )
    log.info(f"frames {frames}, {px_um:.4f} um/px")
    log.info(
        f"a filament must be >= {p['fil_min_length_um']} um "
        f"({p['fil_min_length_um'] / px_um:.0f} px) and <= "
        f"{p['fil_max_width_um']} um ({p['fil_max_width_um'] / px_um:.0f} px) thick"
    )

    if args.sweep:
        setting, values = parse_sweep(args.sweep)
        sweep(
            gfp,
            tracked,
            frames,
            px_um,
            dt_min,
            p,
            name_of,
            setting,
            values,
            out / "sweep.png",
            log,
        )
        return 0

    cells, masks = run_frames(gfp, tracked, frames, px_um, dt_min, p, name_of)
    if not cells:
        die(
            "no cells were found in those frames.",
            "Check that the tracking step produced masks for them.",
        )

    n = len(cells)
    frac = {s: sum(1 for c in cells if c["state"] == s) / n for s in ORDER}
    log.info(
        f"{n} cell-frames:  " + "  ".join(f"{s} {100 * frac[s]:.0f}%" for s in ORDER)
    )

    punct_pos = frac["punctate"] + frac["mixed"]
    if punct_pos > 0.8:
        log.warning(
            f"{100 * punct_pos:.0f}% of cells are being given at least one "
            f"punctum. Unless nearly every cell really does have one, the "
            f"punctum detector is firing on ordinary brightness variation."
        )
        log.warning("  Try, in this order:")
        log.warning("    --sweep GFP_STRUCT_MIN_CONTRAST=0.45,0.8,1.2,1.8")
        log.warning(
            "    --sweep GFP_RIDGE_K=4,6,8   (the ridge filter finds "
            "faint lines, and its leftovers can become puncta)"
        )
        log.warning(
            "  and check mask_structures.tif in Fiji: if it is "
            "speckled inside otherwise even cells, the threshold is "
            "too low."
        )
    if frac["none"] == 0 and frac["diffuse"] < 0.05:
        log.warning(
            "almost nothing is called none or diffuse — if some of "
            "these cells are untagged or unassembled, they are being "
            "given structure they do not have."
        )

    contact_sheet(cells, p, out / "contact_sheet.png", log)

    # binary layers, as one file per layer so they stack in Fiji
    for name, arr in masks.items():
        tifffile.imwrite(out / f"mask_{name}.tif", arr[frames])
    log.info(
        f"binary masks -> mask_structures.tif, mask_filaments.tif, "
        f"mask_puncta.tif  (only the chosen frames)"
    )

    import csv as _csv

    with open(out / "labels.csv", "w", newline="") as fh:
        cols = [
            "frame",
            "track_id",
            "cell_name",
            "state",
            "your_call",
            "signal_over_bg",
            "n_puncta",
            "n_filaments",
            "fil_length_um",
            "fil_width_um",
            "fil_aspect",
            "fil_peak_ratio",
            "struct_area_frac",
        ]
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in cells:
            row = {
                k: round(float(c["feat"].get(k, 0)), 4) for k in cols if k in c["feat"]
            }
            row.update(
                {
                    "frame": c["frame"],
                    "track_id": c["track_id"],
                    "cell_name": c["cell_name"],
                    "state": c["state"],
                    "your_call": "",
                }
            )
            w.writerow(row)
    log.info(
        f"labels.csv written — fill in your_call with one of: "
        f"{', '.join(ORDER)}, then run again with --score"
    )
    log.info(f"everything is in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

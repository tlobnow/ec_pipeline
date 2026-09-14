#!/usr/bin/env python
"""
Learn the phenotype calls from cells you have corrected, then apply them.

    # 1. train on the cells you sorted (uses sorted_labels.csv)
    python tools/classify_cells.py -c cfg.sh --train -o model.joblib

    # 2. apply to every cell in a dataset — no sorting needed
    python tools/classify_cells.py -c other.sh --apply model.joblib \\
           -p all --frames "0 20 40 60"

    # 3. correct only the cells it was unsure about
    python tools/classify_cells.py -c other.sh --review 100

Why this order
--------------
Sorting thousands of cells by hand does not scale, and most of them are easy —
the model gets those right and learns nothing from them being confirmed. What
is worth your time is the handful it finds genuinely ambiguous.

Step 3 exports the LEAST CONFIDENT cells, already filed under the model's
guess, so an hour spent correcting goes entirely on the cells that are still
in doubt. Feeding those back into step 1 improves the model far faster than
the same hour spent on random cells, because a cell the model already gets
right adds nothing.

Everything works from the measurements the pipeline already computed, so
applying a model to a new dataset needs no annotation at all.
"""

from __future__ import annotations

import argparse
import csv as _csv
import importlib.util
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

from common import (
    die,
    get_logger,
    imread as safe_imread,
    load_cell_names,
    load_config,
    load_masks,
    read_units,
    require,
    resolve_positions,
    step_dir,
)

_spec = importlib.util.spec_from_file_location(
    "gfp6", _SCRIPTS / "06_gfp_structures.py"
)
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

CLASSES = ["none", "diffuse", "punctate", "filamentous", "mixed", "non_filamentous"]

# A cell with a filament is filament-bearing whether or not it also has foci.
BINARY_MAP = {
    "filamentous": "filamentous",
    "mixed": "filamentous",
    "none": "non_filamentous",
    "diffuse": "non_filamentous",
    "punctate": "non_filamentous",
    "non_filamentous": "non_filamentous",
}

# Absolute brightness does not survive a change of exposure, gain or strain,
# so a model meant for every experiment should not lean on it. Shapes and
# ratios do transfer.
PORTABLE_FEATURES = [
    "gfp_area_frac",
    "struct_area_frac",
    "shape_circularity",
    "shape_elongation",
    "shape_solidity",
    "fil_aspect",
    "fil_peak_ratio",
    "punct_prominence",
    "punct_axial_pos_max",
    "rel_p95",
    "ridge_area_frac",
    "n_puncta",
    "n_filaments",
    "fil_length_um",
    "fil_width_um",
]

FEATURES = [
    "signal_over_bg",
    "gfp_area_frac",
    "struct_area_frac",
    "shape_circularity",
    "shape_elongation",
    "shape_solidity",
    "fil_length_um",
    "fil_aspect",
    "fil_width_um",
    "fil_peak_ratio",
    "punct_prominence",
    "punct_axial_pos_max",
    "rel_p95",
    "ridge_area_frac",
    "n_puncta",
    "n_filaments",
]

MIN_PER_CLASS = 15


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


def training_dir(cfg, pos):
    return step_dir(cfg, pos, "03_track").with_name("07_training")


# ── gathering labels ────────────────────────────────────────────────────────
def label_folders(cfg, positions):
    """Every folder that might hold labels: the first sort and every review."""
    seen, out = set(), []
    for pos in positions:
        base = training_dir(cfg, pos).parent
        for d in sorted(base.glob("0[78]_*")):
            if d.is_dir() and (d / "sorted_labels.csv").exists() and d not in seen:
                seen.add(d)
                out.append(d)
    return out


def gather_labels(cfg, positions, log):
    """
    Every corrected cell, from all rounds of sorting.

    The first sort and each review round live in their own folder, and every
    round is used — a review adds to the training set rather than replacing
    it, which is the whole point of doing rounds. If the same cell was judged
    twice, the later round wins, since that is the more considered look at a
    cell the model found hard.
    """
    by_cell, order = {}, []
    for d in label_folders(cfg, positions):
        feat_path = d / "features.csv"
        if not feat_path.exists():
            log.warning(f"{d.name} has labels but no features.csv, skipped")
            continue
        feats = {r["file"]: r for r in _csv.DictReader(open(feat_path))}
        n_new = n_upd = 0
        for r in _csv.DictReader(open(d / "sorted_labels.csv")):
            f = feats.get(r["file"])
            if f is None or r.get("final_call") not in CLASSES:
                continue
            key = (str(f.get("position")), str(f.get("frame")), str(f.get("track_id")))
            rec = {
                **f,
                "label": r["final_call"],
                "was_corrected": r.get("corrected", "0"),
                "round": d.name,
            }
            if key in by_cell:
                n_upd += 1
            else:
                n_new += 1
                order.append(key)
            by_cell[key] = rec
        log.info(
            f"  {d.name}: {n_new} new cell(s)"
            + (f", {n_upd} re-judged" if n_upd else "")
        )

    if not by_cell:
        die(
            "no corrected labels were found.",
            "Sort the exported cells, then write sorted_labels.csv with\n"
            "            python tools/export_cells.py -c <cfg> --collect\n"
            "            (add --dir <...>/08_review after a review round),\n"
            "            then train.",
        )
    rows = [by_cell[k] for k in order]
    rounds = defaultdict(int)
    for r in rows:
        rounds[r["round"]] += 1
    if len(rounds) > 1:
        log.info(
            "  training on " + " + ".join(f"{v} from {k}" for k, v in rounds.items())
        )
    return rows


def gather_from_dir(d, log):
    """Labels from one shared folder, whatever classes it holds."""
    lab, feat = d / "sorted_labels.csv", d / "features.csv"
    if not lab.exists() or not feat.exists():
        die(
            f"{d} has no sorted_labels.csv and features.csv",
            "Sort the images there, then run\n"
            "            python tools/export_cells.py -c <cfg> --collect "
            "--dir " + str(d),
        )
    feats = {r["file"]: r for r in _csv.DictReader(open(feat))}
    rows = []
    for r in _csv.DictReader(open(lab)):
        f = feats.get(r["file"])
        if f is None or not r.get("final_call"):
            continue
        rows.append(
            {
                **f,
                "label": r["final_call"],
                "round": d.name,
                "experiment": f.get("experiment", ""),
            }
        )
    if not rows:
        die(f"no usable labels in {d}", "Check sorted_labels.csv.")
    by_expt = defaultdict(int)
    for r in rows:
        by_expt[r.get("experiment") or "(unknown)"] += 1
    log.info(f"  {len(rows)} labelled cell(s) from {len(by_expt)} " f"experiment(s):")
    for k, v in sorted(by_expt.items()):
        log.info(f"    {k[:50]:50s} {v}")
    return rows


def matrix(rows, portable=False):
    pool = PORTABLE_FEATURES if portable else FEATURES
    feats = [f for f in pool if f in rows[0]]
    X = np.array([[float(r.get(f) or 0) for f in feats] for r in rows])
    return X, feats


# ── train ───────────────────────────────────────────────────────────────────
def leave_one_experiment_out(model, X, y, groups, log):
    """
    Train on all experiments but one, test on that one. Repeat.

    Splitting cells at random reports how well the model interpolates inside
    experiments it has already seen — cells from one field are correlated, so
    that number is optimistic. Holding out a whole experiment answers the
    question that matters for a shared model: does it work on the next
    dataset.
    """
    from sklearn.base import clone

    uniq = sorted(set(groups))
    if len(uniq) < 2:
        log.warning(
            "labels come from a single experiment, so how well this "
            "transfers to another cannot be measured. Label a few "
            "cells from a second experiment before trusting it "
            "elsewhere."
        )
        return None

    log.info("")
    log.info("held-out experiment test (train on the rest, test on this one):")
    accs = []
    for g in uniq:
        te = np.array([x == g for x in groups])
        if te.all() or len(set(y[~te])) < 2:
            continue
        m = clone(model).fit(X[~te], y[~te])
        pred = m.predict(X[te])
        acc = float(np.mean(pred == y[te]))
        accs.append(acc)
        parts = []
        for c in sorted(set(y[te])):
            sel = y[te] == c
            parts.append(f"{c} {np.mean(pred[sel] == c):.2f}")
        log.info(
            f"  {g[:44]:44s} n={int(te.sum()):4d}  acc {acc:.2f}   " + "  ".join(parts)
        )
    if accs:
        log.info(f"  mean across experiments: {np.mean(accs):.2f}")
        log.info(
            "  This is the number to quote for a model used on new "
            "data — the random-split accuracy above will be higher and "
            "means less."
        )
    return accs


def train(cfg, positions, out_path, log, labels_dir=None, portable=False):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
    except ImportError:
        die(
            "scikit-learn is not installed in this python: " + sys.executable,
            "conda activate phage_pipeline && pip install scikit-learn joblib",
        )
    try:
        import joblib
    except ImportError:
        die("joblib is not installed.", "pip install joblib")

    rows = (
        gather_from_dir(Path(labels_dir).expanduser(), log)
        if labels_dir
        else gather_labels(cfg, positions, log)
    )
    counts = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    log.info("labels: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    X, feats = matrix(rows, portable)
    if portable:
        log.info(
            f"portable mode: {len(feats)} feature(s), none of them "
            f"absolute brightness"
        )
    y = np.array([r["label"] for r in rows])
    if len(set(y)) < 2:
        die(
            f"every labelled cell is '{y[0]}', so there is nothing to learn.",
            "Sort cells of at least two phenotypes.",
        )

    thin = [k for k, v in counts.items() if v < MIN_PER_CLASS]
    if thin:
        log.warning(
            f"fewer than {MIN_PER_CLASS} examples of: "
            f"{', '.join(thin)} — those classes will be unreliable "
            f"whatever the overall accuracy says."
        )

    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
        n_jobs=-1,
    )
    n_split = min(5, min(counts[k] for k in counts))
    if n_split >= 2:
        cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
        pred = cross_val_predict(model, X, y, cv=cv)
        log.info(
            f"cross-validated accuracy {np.mean(pred == y):.2f} "
            f"({n_split}-fold, held-out cells only)"
        )
        log.info("")
        log.info("how well each phenotype is learned:")
        for c in sorted(counts):
            n = int((y == c).sum())
            if not n:
                continue
            rec = float((pred[y == c] == c).mean())
            flag = "   <-- NOT LEARNED" if rec < 0.5 else ""
            log.info(f"  {c:13s} n={n:4d}  recall {rec:.2f}{flag}")
        log.info("")
        log.info(
            "Overall accuracy is carried by whichever class has the most "
            "examples, so read the per-class numbers before trusting it."
        )
    else:
        log.warning("too few examples per class to cross-validate")

    groups = [r.get("experiment") or "(unknown)" for r in rows]
    leave_one_experiment_out(model, X, y, groups, log)

    model.fit(X, y)
    joblib.dump(
        {
            "model": model,
            "features": feats,
            "classes": list(model.classes_),
            "portable": portable,
            "n_train": len(rows),
            "experiments": sorted({r.get("experiment") or "" for r in rows}),
        },
        out_path,
    )
    log.info("")
    log.info(
        f"model saved -> {out_path}  ({len(rows)} cells, " f"{len(feats)} features)"
    )
    order = np.argsort(model.feature_importances_)[::-1]
    log.info(
        "features it relies on: "
        + ", ".join(
            f"{feats[i]} {model.feature_importances_[i]:.2f}" for i in order[:5]
        )
    )


# ── apply ───────────────────────────────────────────────────────────────────
def measure_all(cfg, positions, frames_spec, log):
    """Measure every cell in the chosen frames, without any labels."""
    from skimage import measure as _m

    p = dict(cfg["gfp"])
    rows = []
    for pos in positions:
        ld = step_dir(cfg, pos, "01_load")
        gfp = safe_imread(
            require(
                ld / "gfp.tif", f"pos {pos} GFP", "Run the pipeline for that field."
            )
        )
        masks, _, _ = load_masks(cfg, pos, log)
        px_um, dt_min = read_units(ld, log, cfg)
        lin = step_dir(cfg, pos, "03_track") / "lineage.csv"
        names = load_cell_names(lin)[0] if lin.exists() else {}

        T = gfp.shape[0]
        frames = []
        for f in str(frames_spec).replace(",", " ").split():
            frames.append(T - 1 if f.lower() in ("last", "end", "final") else int(f))
        frames = [f for f in sorted(set(frames)) if 0 <= f < T]

        for t in frames:
            lbl = masks[t]
            if not lbl.any():
                continue
            bg = float(np.median(gfp[t][lbl == 0]))
            sigma = G.pixel_noise(gfp[t], lbl == 0)
            for reg in _m.regionprops(lbl):
                y0, x0, y1, x1 = reg.bbox
                cm = lbl[y0:y1, x0:x1] == reg.label
                feat, _, _, _, _ = G.measure_cell(
                    gfp[t][y0:y1, x0:x1], cm, bg, sigma, px_um, p
                )
                rows.append(
                    {
                        "position": pos,
                        "frame": t,
                        "frame_fiji": t + 1,
                        "time_min": round(t * dt_min, 2),
                        "track_id": int(reg.label),
                        "cell_name": names.get(int(reg.label), str(reg.label)),
                        "rule_call": G.state_of(
                            feat, p["min_signal_over_bg"], p["min_struct_area_frac"]
                        ),
                        **{k: round(float(feat.get(k, 0)), 4) for k in G.FEATURES},
                    }
                )
        log.info(f"  field {pos}: {len(frames)} frame(s) measured")
    return rows


def apply_model(cfg, positions, model_path, frames_spec, out_csv, log):
    import joblib

    bundle = joblib.load(model_path)
    model, feats = bundle["model"], bundle["features"]
    log.info(f"model trained on {bundle.get('n_train', '?')} cell(s)")

    rows = measure_all(cfg, positions, frames_spec, log)
    if not rows:
        die("no cells were measured.", "Check the frames and the positions.")

    missing = [f for f in feats if f not in rows[0]]
    if missing:
        die(
            f"the model needs features this data does not have: {missing}",
            "The model was trained with a different pipeline version. "
            "Re-run the GFP step, or retrain.",
        )

    X = np.array([[float(r.get(f) or 0) for f in feats] for r in rows])
    proba = model.predict_proba(X)
    pred = model.classes_[np.argmax(proba, axis=1)]
    conf = proba.max(axis=1)

    # The rules speak in five phenotypes; a binary model speaks in two. Map
    # the rule call into the model's own vocabulary before comparing, or a
    # "mixed" cell — which DOES have a filament — never counts as agreement
    # and the agreement rate reads far too low.
    binary = set(model.classes_) <= {"filamentous", "non_filamentous"}
    to_model = (lambda s: BINARY_MAP.get(s, s)) if binary else (lambda s: s)
    for r, pc, cf in zip(rows, pred, conf):
        r["predicted"] = pc
        r["confidence"] = round(float(cf), 4)
        r["rule_call_comparable"] = to_model(r["rule_call"])
        r["agrees_with_rules"] = int(pc == r["rule_call_comparable"])

    with open(out_csv, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    counts = defaultdict(int)
    for r in rows:
        counts[r["predicted"]] += 1
    n = len(rows)
    log.info("")
    log.info(
        f"{n} cells: "
        + "  ".join(f"{c} {100 * counts[c] / n:.0f}%" for c in CLASSES if counts[c])
    )
    agree = sum(r["agrees_with_rules"] for r in rows)
    log.info(
        f"the model and the threshold rules agree on "
        f"{100 * agree / n:.0f}% of cells"
    )
    for thr in (0.9, 0.8, 0.7):
        k = int(np.sum(conf >= thr))
        log.info(
            f"  {k:6d} cells ({100 * k / n:4.0f}%) predicted with "
            f"confidence >= {thr}"
        )
    log.info("")
    log.info(f"predictions -> {out_csv}")
    log.info(f"Next: correct only the unsure ones with")
    log.info(f"    python tools/classify_cells.py -c <cfg> --review 100")


# ── review ──────────────────────────────────────────────────────────────────
def already_judged(cfg, log):
    """Every cell that has been given a label in any round, anywhere."""
    seen = set()
    roots = set()
    for pos in range(0, 64):
        d = step_dir(cfg, pos, "03_track").parent
        if d.exists():
            roots.add(d)
    extra = os.environ.get("FILAMENT_TRAINING_DIR", "").strip()
    dirs = [Path(extra)] if extra else []
    for r in roots:
        dirs.extend(sorted(r.glob("0[78]_*")))
    for d in dirs:
        f = Path(d) / "sorted_labels.csv"
        if not f.exists():
            continue
        for r in _csv.DictReader(open(f)):
            seen.add(
                (str(r.get("position")), str(r.get("frame")), str(r.get("track_id")))
            )
    return seen


def review(cfg, pred_csv, n_want, out_dir, log, pad=4):
    """
    Export the least confident cells, filed under the model's guess.

    Confirming a cell the model already gets right teaches it nothing. The
    cells worth an hour of your time are the ones it cannot decide, so those
    are the ones exported.
    """
    import tifffile

    try:
        import imageio.v3 as iio

        write = iio.imwrite
    except ImportError:
        from matplotlib import pyplot as plt

        write = plt.imsave

    rows = list(_csv.DictReader(open(pred_csv)))
    if not rows:
        die(f"no predictions in {pred_csv}", "Run --apply first.")

    # Cells already judged must not come back. Without this the same hard
    # cells are exported every round — they stay the least confident, so
    # each review re-asks questions that have already been answered and the
    # training set stops growing.
    judged = already_judged(cfg, log)
    fresh = [
        r
        for r in rows
        if (str(r["position"]), str(r["frame"]), str(r["track_id"])) not in judged
    ]
    if judged:
        log.info(
            f"{len(judged)} cell(s) already judged in an earlier round, " f"skipped"
        )
    if not fresh:
        die(
            "every cell has been judged already.",
            "Add frames with --frames, or another field, to find new ones.",
        )

    fresh.sort(key=lambda r: float(r["confidence"]))
    picked = fresh[:n_want]
    rows = fresh
    log.info(
        f"{len(rows)} cells; taking the {len(picked)} least confident "
        f"(confidence {float(picked[0]['confidence']):.2f} to "
        f"{float(picked[-1]['confidence']):.2f})"
    )

    # A fresh folder per round. Reusing one leaves the previous round's
    # images behind with no matching feature rows, which is what produced
    # all those "not in features.csv" warnings.
    if out_dir.exists() and any(out_dir.rglob("*.png")):
        n = 1
        while (out_dir.parent / f"{out_dir.name}_{n:02d}").exists():
            n += 1
        keep = out_dir.parent / f"{out_dir.name}_{n:02d}"
        out_dir.rename(keep)
        log.info(
            f"the previous round was moved to {keep.name} so this one " f"starts clean"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = (
        ["filamentous", "non_filamentous"]
        if {r["predicted"] for r in picked} <= {"filamentous", "non_filamentous"}
        else CLASSES
    )
    for c in classes:
        (out_dir / c).mkdir(exist_ok=True)

    by_pos = defaultdict(list)
    for r in picked:
        by_pos[int(r["position"])].append(r)

    written, feat_rows = 0, []
    for pos, recs in sorted(by_pos.items()):
        ld = step_dir(cfg, pos, "01_load")
        gfp = safe_imread(ld / "gfp.tif")
        masks, _, _ = load_masks(cfg, pos, log)
        Y, X = masks.shape[1], masks.shape[2]
        for r in recs:
            t, tid = int(r["frame"]), int(r["track_id"])
            sel = masks[t] == tid
            if not sel.any():
                continue
            ys, xs = np.nonzero(sel)
            y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
            y1, x1 = min(Y, ys.max() + 1 + pad), min(X, xs.max() + 1 + pad)
            cm = sel[y0:y1, x0:x1]
            crop = gfp[t][y0:y1, x0:x1].astype(float)
            v = crop[cm]
            lo, hi = (
                np.percentile(v, [2, 99.5]) if v.size > 10 else (crop.min(), crop.max())
            )
            hi = max(float(hi), float(lo) + 1)
            disp = np.clip((crop - lo) / (hi - lo), 0, 1)
            disp[~cm] = 0
            rgb = np.zeros(disp.shape + (3,), np.uint8)
            rgb[..., 1] = (disp * 255).astype(np.uint8)
            rgb = np.repeat(np.repeat(rgb, 3, 0), 3, 1)

            fname = (
                f"pos{pos:02d}_f{t:03d}_id{tid:05d}_"
                f"{r['cell_name']}.png".replace("/", "-")
            )
            write(out_dir / r["predicted"] / fname, rgb)
            feat_rows.append({**r, "file": fname, "pipeline_call": r["predicted"]})
            written += 1

    with open(out_dir / "features.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(feat_rows[0].keys()))
        w.writeheader()
        w.writerows(feat_rows)

    log.info(f"{written} cell(s) -> {out_dir}")
    log.info("They are filed under the model's guess. Move only what is wrong,")
    log.info("then run  --collect  in that folder and train again — the model")
    log.info("improves fastest from the cells it could not decide.")


def export_class(cfg, pred_csv, want, min_conf, max_per_field, out_dir, log, pad=4):
    """
    Cut measurable crops for every cell the model called `want`.

    Driven by predictions.csv, so it covers a whole dataset without anything
    having been sorted by hand. Two things come out per field:

      * one calibrated TIF per cell, in to_measure_<class>/ — raw counts,
        masked, pixel size in the header, so a line drawn in Fiji reads out
        in microns
      * marked_<class>_posNN.tif, the whole field with only those cells kept
        and each cell's PIXEL VALUE set to its track id, so hovering over a
        cell in Fiji shows its id

    File names carry the experiment, so crops from infected and uninfected
    runs can sit in one folder without colliding or being confused.
    """
    import tifffile

    rows = [r for r in _csv.DictReader(open(pred_csv)) if r.get("predicted") == want]
    if not rows:
        die(
            f"no cells were predicted '{want}' in {pred_csv}",
            "Check the class name, or run --apply first. Classes: "
            + ", ".join(CLASSES),
        )

    n_all = len(rows)
    rows = [r for r in rows if float(r["confidence"]) >= min_conf]
    if not rows:
        die(
            f"all {n_all} '{want}' cells are below --min-confidence " f"{min_conf}.",
            "Lower it, or review and retrain first.",
        )
    log.info(
        f"{n_all} cell(s) predicted '{want}', {len(rows)} of them at "
        f"confidence >= {min_conf}"
    )

    expt = cfg["experiment"]["name"].replace(".nd2", "").replace("/", "-")
    cond = cfg["experiment"].get("condition", "") or "unspecified"
    gp_full = dict(cfg["gfp"])
    from skimage.segmentation import find_boundaries

    by_pos = defaultdict(list)
    for r in rows:
        by_pos[int(r["position"])].append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"to_measure_{want}"
    dest.mkdir(exist_ok=True)

    listed, n_tif = [], 0
    for pos, recs in sorted(by_pos.items()):
        if max_per_field and len(recs) > max_per_field:
            # spread the sample over time rather than taking the first N,
            # which would all come from the earliest frame
            recs = sorted(
                recs, key=lambda r: (int(r["frame"]), -float(r["confidence"]))
            )
            step = len(recs) / max_per_field
            recs = [recs[int(i * step)] for i in range(max_per_field)]

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

        for t, rs in sorted(by_frame.items()):
            for r in rs:
                tid = int(r["track_id"])
                sel = masks[t] == tid
                if not sel.any():
                    continue
                ys, xs = np.nonzero(sel)
                y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
                y1, x1 = min(Y, ys.max() + 1 + pad), min(X, xs.max() + 1 + pad)
                cm = sel[y0:y1, x0:x1]
                raw = gfp[t][y0:y1, x0:x1]
                crop = np.where(cm, raw, 0).astype(np.uint16)

                # re-run the detector on this crop to recover the filament
                # mask, then the exact path whose length was reported
                feat, fmask, _pts, _rel, _s = G.measure_cell(
                    raw,
                    cm,
                    float(np.median(gfp[t][masks[t] == 0])),
                    G.pixel_noise(gfp[t], masks[t] == 0),
                    px_um,
                    gp_full,
                )
                path, lengths = G.filament_paths(fmask, px_um)

                edge = find_boundaries(cm, mode="inner")
                stack = np.zeros((4,) + crop.shape, np.uint16)
                stack[0] = crop
                stack[1][edge] = 65535  # cell boundary
                stack[2][fmask] = 65535  # what counted as filament
                stack[3][path] = 65535  # the line that was measured
                fname = f"{expt}_pos{pos:02d}_f{t:03d}_id{tid:05d}.tif"
                tifffile.imwrite(
                    dest / fname,
                    stack,
                    imagej=True,
                    resolution=(1.0 / px_um, 1.0 / px_um),
                    metadata={
                        "axes": "CYX",
                        "unit": "um",
                        "mode": "composite",
                        "Labels": [
                            "gfp",
                            "cell outline",
                            "filament mask",
                            "measured path",
                        ],
                    },
                )
                n_tif += 1
                r["_paths_um"] = ";".join(f"{x:.3f}" for x in lengths)
                listed.append(
                    {
                        "file": fname,
                        "experiment": expt,
                        "condition": cond,
                        "position": pos,
                        "frame": t,
                        "frame_fiji": t + 1,
                        "time_min": r.get("time_min", ""),
                        "track_id": tid,
                        "cell_name": r.get("cell_name", ""),
                        "predicted": want,
                        "confidence": r["confidence"],
                        "auto_length_um": r.get("fil_length_um", ""),
                        "all_path_lengths_um": r.get("_paths_um", ""),
                        "measured_length_um": "",
                    }
                )

        frames = sorted(by_frame)
        # channel 0: kept cells, pixel value = track id (hover shows the id)
        # channel 1: the measured filament paths, in place on the full field
        marked = np.zeros((len(frames), 2, Y, X), np.uint16)
        for i, t in enumerate(frames):
            ids = [int(r["track_id"]) for r in by_frame[t]]
            sel = np.isin(masks[t], ids)
            marked[i, 0][sel] = masks[t][sel].astype(np.uint16)
            bg_t = float(np.median(gfp[t][masks[t] == 0]))
            sg_t = G.pixel_noise(gfp[t], masks[t] == 0)
            for r in by_frame[t]:
                tid = int(r["track_id"])
                s2 = masks[t] == tid
                if not s2.any():
                    continue
                ys, xs = np.nonzero(s2)
                a0, b0 = ys.min(), xs.min()
                a1, b1 = ys.max() + 1, xs.max() + 1
                cm2 = s2[a0:a1, b0:b1]
                _f, fm2, _p, _r, _s = G.measure_cell(
                    gfp[t][a0:a1, b0:b1], cm2, bg_t, sg_t, px_um, gp_full
                )
                pth, _ = G.filament_paths(fm2, px_um)
                marked[i, 1][a0:a1, b0:b1][pth] = 65535
        mark = out_dir / f"marked_{want}_{expt}_pos{pos:02d}.tif"
        tifffile.imwrite(
            mark,
            marked,
            imagej=True,
            resolution=(1.0 / px_um, 1.0 / px_um),
            metadata={
                "axes": "TCYX",
                "unit": "um",
                "mode": "composite",
                "Labels": ["kept cell ids", "measured paths"] * len(frames),
            },
        )
        log.info(f"  field {pos}: {len(recs)} cell(s) over frames {frames}")

    index = out_dir / f"to_measure_{want}.csv"
    write_header = not index.exists()
    with open(index, "a", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(listed[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(listed)

    log.info("")
    log.info(f"{n_tif} calibrated TIF(s) -> {dest}")
    log.info(f"index (append-only, so both conditions land in one file) " f"-> {index}")
    log.info("")
    log.info(
        "  Each crop has four channels: the GFP, the cell outline, what "
        "counted as filament, and THE LINE THAT WAS MEASURED."
    )
    log.info(
        "  Turn the last one on to see where the algorithm went — through "
        "a branch, around a bend, or off along a neighbour — instead of "
        "only comparing its number with yours."
    )
    log.info("  To measure by hand anyway: draw along the filament, press M.")
    log.info("  Put it in the measured_length_um column of the index, next to")
    log.info("  auto_length_um, which is what the detector measured — the two")
    log.info("  together tell you whether the automatic lengths can be " "trusted.")
    log.info("")
    log.info(
        f"  marked_{want}_..._posNN.tif is the whole field with only "
        f"these cells kept; hovering over one in Fiji shows its track "
        f"id, because the pixel value IS the id."
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-p", "--position", default="all")
    ap.add_argument(
        "--frames", default="0", help='frames to classify, e.g. "0 20 40 last"'
    )
    ap.add_argument("--train", action="store_true")
    ap.add_argument(
        "--labels-dir",
        default="",
        help="a shared training folder to learn from, instead of "
        "each field's own 07_training",
    )
    ap.add_argument(
        "--portable",
        action="store_true",
        help="use only features that survive a change of "
        "exposure, gain or strain — for a model meant to be "
        "used on experiments it was not trained on",
    )
    ap.add_argument("--apply", default="")
    ap.add_argument(
        "--review",
        type=int,
        default=0,
        help="export this many of the least confident cells",
    )
    ap.add_argument(
        "--export-class",
        default="",
        help="cut measurable TIFs for every cell predicted as "
        "this phenotype, e.g. filamentous",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="only export cells the model is at least this sure " "about",
    )
    ap.add_argument(
        "--max-per-field",
        type=int,
        default=0,
        help="cap per field, sampled across time (0 = all)",
    )
    ap.add_argument(
        "--to",
        default="",
        help="where the crops go. Point both conditions at the "
        "same folder to measure them together.",
    )
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()

    load_shell_config(args.config)
    log = get_logger("classify", False)
    cfg = load_config()
    positions = resolve_positions(cfg, args.position, log)
    root = (
        Path(cfg["experiment"]["output_root"]).expanduser() / cfg["experiment"]["name"]
    )

    if args.train:
        train(
            cfg,
            positions,
            Path(args.out or root / "gfp_model.joblib"),
            log,
            args.labels_dir,
            args.portable,
        )
        return 0
    if args.apply:
        apply_model(
            cfg,
            positions,
            Path(args.apply).expanduser(),
            args.frames,
            Path(args.out or root / "predictions.csv"),
            log,
        )
        return 0
    if args.export_class:
        if args.export_class not in CLASSES:
            die(
                f"'{args.export_class}' is not a phenotype.",
                "Use one of: " + ", ".join(CLASSES),
            )
        export_class(
            cfg,
            Path(args.out or root / "predictions.csv"),
            args.export_class,
            args.min_confidence,
            args.max_per_field,
            Path(args.to).expanduser() if args.to else root / "measure",
            log,
        )
        return 0
    if args.review:
        review(
            cfg,
            Path(args.out or root / "predictions.csv"),
            args.review,
            training_dir(cfg, positions[0]).with_name("08_review"),
            log,
        )
        return 0

    die("nothing to do.", "Use --train, --apply <model>, or --review <n>.")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""
Split a large nd2 into one small folder per field of view.

    python tools/split_nd2.py -c configs/my_config.sh
    python tools/split_nd2.py -c configs/my_config.sh --positions "0 2"
    python tools/split_nd2.py -c configs/my_config.sh --inspect-focus

Why
---
A 4-position, 17-plane nd2 is 72 GB, and reading one position took 76 seconds
before any analysis started. Splitting once writes per-position TIFFs next to
the nd2, already reduced to a single focal plane, which is about 17x smaller
and opens instantly. Every later run reads those instead of the nd2.

The split folder is written beside the nd2 as <name>_split/pos_00/ and so on,
so it travels with the raw data.

Z stacks
--------
Each frame is reduced to one image per channel. How is set by Z_METHOD:

    focus  (default)  the sharpest plane, chosen per frame from the phase
                      channel and applied to every channel
    fixed             one plane for the whole movie, set by Z_PLANE
    max               brightest value across planes
    mean              average across planes

'focus' is the sensible default for a long timelapse: it tracks the slow
defocus that happens over hours without being fooled frame to frame, because
the choice is median-filtered and held within Z_MAX_STEP planes of the last.

--inspect-focus writes the focus diagnostics for the first few frames and
stops, without writing any image data. Worth doing once on a new dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from common import (
    Timer,
    best_plane,
    die,
    focus_score,
    frame_times_from_events,
    get_logger,
    programmed_interval_min,
    reduce_z,
    resolve_channel,
    smooth_plane_choices,
)

CHANNELS = ("bf", "gfp", "rfp")


# ── config ──────────────────────────────────────────────────────────────────
def load_shell_config(path):
    """Read config.sh by sourcing it, so there is one config format."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        die(f"config file not found: {path}", "Check the path you passed to -c.")
    out = subprocess.run(
        ["bash", "-c", f'set -a; . "{path}"; env'], capture_output=True, text=True
    )
    if out.returncode != 0:
        die(
            f"could not read the config file: {path}",
            "Look for a missing quote or a space around an = sign.",
        )
    env = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    os.environ.update(env)
    os.environ.setdefault("CONFIG_DIR", str(path.parent))
    return env


def split_root(cfg_env):
    explicit = cfg_env.get("SPLIT_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    nd2 = Path(cfg_env["ND2_PATH"]).expanduser()
    return nd2.parent / f"{nd2.stem}_split"


# ── nd2 helpers ─────────────────────────────────────────────────────────────
def axis_indexer(sizes, **fixed):
    """
    Build an index tuple for an array whose axes follow the file's own order.

    Guessing axis order from the shape breaks as soon as a file has a Z or P
    axis, which is exactly what went wrong before. f.sizes is ordered, so it
    can be used directly.
    """
    idx = []
    for axis in sizes:
        # A None value means "not applicable" (a single-position file has no
        # P index to fix). It must become a full slice: using None directly
        # inserts a new axis, which silently produced an empty array.
        if axis in fixed and fixed[axis] is not None:
            idx.append(fixed[axis])
        else:
            idx.append(slice(None))
    return tuple(idx)


def read_acquisition(f, log):
    acq = {
        "sizes": dict(f.sizes),
        "channel_names": [],
        "pixel_size_um": None,
        "frame_interval_min": None,
    }
    try:
        acq["channel_names"] = [c.channel.name for c in f.metadata.channels]
    except Exception:
        pass
    try:
        acq["pixel_size_um"] = float(f.voxel_size().x)
    except Exception:
        pass
    # Same rule as the load step: the timestamps the microscope recorded beat
    # the period that was programmed before the run.
    times = frame_times_from_events(f, None)
    if times is not None and len(times) > 1:
        acq["frame_interval_min"] = float(np.median(np.diff(times)) / 60.0)
        acq["frame_interval_source"] = "recorded timestamps"
    else:
        prog = programmed_interval_min(f)
        if prog:
            acq["frame_interval_min"] = prog
            acq["frame_interval_source"] = "programmed period"
    if not acq["frame_interval_min"]:
        override = os.environ.get("FRAME_INTERVAL_MIN", "").strip()
        if override:
            acq["frame_interval_min"] = float(override)
            acq["frame_interval_source"] = "config override"
        else:
            log.warning(
                "no frame interval found in the nd2 — set "
                "FRAME_INTERVAL_MIN in the config before analysing"
            )
    if not acq["pixel_size_um"]:
        override = os.environ.get("PIXEL_SIZE_UM", "").strip()
        if override:
            acq["pixel_size_um"] = float(override)
            acq["pixel_size_source"] = "config override"
    if acq.get("frame_interval_min"):
        log.info(
            f"  interval {acq['frame_interval_min']:.3f} min "
            f"({acq.get('frame_interval_source', '?')})"
        )
    return acq


# ── focus ───────────────────────────────────────────────────────────────────
def write_focus_plan(path, chosen, all_scores, z_method, n_z, log):
    """
    Save which plane each frame uses, as a file you can read and edit.

    The plan is kept separate from the images on purpose. Choosing planes
    means reading the whole stack, which is the slow part; writing the images
    is cheap. Saving the decision means it can be inspected, corrected by
    hand, and reused — rather than being baked invisibly into the TIFFs where
    the only way to change it is to redo everything.
    """
    import csv as _csv

    with open(path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(
            ["frame", "z_plane", "method", "n_z", "sharpness", "margin_over_next"]
        )
        for t_i, z in enumerate(chosen):
            sc = (all_scores or {}).get(t_i)
            if sc:
                s = sorted(sc, reverse=True)
                val = f"{sc[int(z)]:.6g}"
                marg = f"{(s[0] - s[1]) / s[0]:.4f}" if len(s) > 1 and s[0] else ""
            else:
                val, marg = "", ""
            w.writerow([t_i, int(z), z_method, n_z, val, marg])
    log.info(f"focus plan -> {path}")


def read_focus_plan(path, n_t, log):
    """Planes saved by an earlier run, if the file is there and usable."""
    import csv as _csv

    if not Path(path).exists():
        return None
    try:
        rows = list(_csv.DictReader(open(path)))
        plan = {int(r["frame"]): int(r["z_plane"]) for r in rows}
    except Exception:
        log.warning(f"could not read {path}, choosing planes again")
        return None
    missing = [t for t in range(n_t) if t not in plan]
    if missing:
        log.warning(f"{path} covers {len(plan)} of {n_t} frames, choosing " f"again")
        return None
    log.info(
        f"using the saved focus plan from {Path(path).name} "
        f"(planes {min(plan.values())}-{max(plan.values())})"
    )
    return [plan[t] for t in range(n_t)]


def consensus_plane(all_scores, log):
    """
    One plane for the whole movie, decided by every frame together.

    Each frame's scores are divided by that frame's own mean first. Without
    that, the comparison is dominated by how the field changes over time —
    more cells later means a higher score at every plane — and that variation
    is much larger than the difference between planes, so averaging raw
    scores would just measure growth.

    Normalising per frame removes the drift and leaves only the question that
    matters: relative to its own frame, which plane is consistently sharpest?
    A single frame cannot answer that when the planes differ by well under a
    percent; sixty frames can.
    """
    rel = []
    for sc in all_scores.values():
        a = np.asarray(sc, dtype=float)
        m = a.mean()
        if m > 0:
            rel.append(a / m)
    if not rel:
        return None, 0.0, None
    rel = np.vstack(rel)
    mean_rel = rel.mean(axis=0)
    best = int(np.argmax(mean_rel))
    order = np.sort(mean_rel)[::-1]
    margin = float((order[0] - order[1]) / order[0]) if len(order) > 1 else 0.0

    # how often that plane wins in an individual frame
    wins = int(np.sum(np.argmax(rel, axis=1) == best))
    log.info(
        f"consensus over {len(rel)} frame(s): plane {best} is sharpest "
        f"on average, winning in {wins} of {len(rel)} frames "
        f"({100 * wins / len(rel):.0f}%)"
    )
    log.info(
        "  mean relative sharpness per plane: "
        + "  ".join(f"z{i}={v:.4f}" for i, v in enumerate(mean_rel))
    )
    return best, margin, mean_rel


def choose_planes(
    arr,
    sizes,
    position,
    bf_index,
    n_z,
    n_t,
    method,
    fixed_plane,
    max_step,
    sample_every,
    log,
):
    """One z plane per frame, judged on the phase channel."""
    if method not in ("focus", "consensus"):
        return [int(fixed_plane)] * n_t, None

    frames = list(range(0, n_t, max(1, sample_every)))
    if frames[-1] != n_t - 1:
        frames.append(n_t - 1)

    measured, all_scores = {}, {}
    with Timer(log, f"focus search ({len(frames)} of {n_t} frames)"):
        for t in frames:
            idx = axis_indexer(sizes, T=t, P=position, C=bf_index)
            stack = np.asarray(arr[idx])
            z, scores = best_plane(stack)
            measured[t] = z
            all_scores[t] = scores

    # fill in the frames that were not measured
    chosen = []
    for t in range(n_t):
        if t in measured:
            chosen.append(measured[t])
        else:
            before = max(x for x in measured if x <= t)
            after = min((x for x in measured if x >= t), default=before)
            if after == before:
                chosen.append(measured[before])
            else:
                w = (t - before) / (after - before)
                chosen.append(
                    int(round(measured[before] * (1 - w) + measured[after] * w))
                )

    # How clearly the sharpest plane beat the runner-up, per frame. A small
    # margin means the metric is choosing between planes that look the same,
    # so the per-frame choice is noise rather than drift.
    margins = []
    for sc in all_scores.values():
        s = sorted(sc, reverse=True)
        if len(s) > 1 and s[0] > 0:
            margins.append((s[0] - s[1]) / s[0])
    med_margin = float(np.median(margins)) if margins else 0.0

    best, cons_margin, _ = consensus_plane(all_scores, log)

    if method == "consensus":
        if best is None:
            die(
                "the focus scores could not be read.",
                'Use Z_METHOD="fixed" with Z_PLANE set by hand.',
            )
        log.info(f"using plane {best} for every frame")
        return [int(best)] * n_t, all_scores

    chosen = smooth_plane_choices(chosen, max_step=max_step, log=log)
    lo, hi = min(chosen), max(chosen)
    log.info(
        f"focus: plane {lo} to {hi} of 0-{n_z - 1} over the movie"
        + ("" if lo != hi else "  (no drift)")
    )
    log.info(
        f"focus: per frame, the sharpest plane beat the next by "
        f"{100 * med_margin:.2f}% (median)"
    )

    if med_margin < 0.02:
        log.warning(
            f"the planes are nearly identical to this metric "
            f"({100 * med_margin:.2f}% margin per frame), so choosing one per "
            f"frame is following noise, not focus."
        )
        if best is not None:
            log.warning(f"  Use one plane for the whole movie instead:")
            log.warning(f'      Z_METHOD="consensus"      (picks it for you)')
            log.warning(
                f'      Z_METHOD="fixed"  Z_PLANE={best}   '
                f"(same answer, written down)"
            )
    if hi - lo >= n_z - 1 and n_z > 1:
        log.warning(
            f"the choice swept the whole stack, plane {lo} to {hi} of {n_z}. "
            f"Focus drifts slowly and smoothly, so using the full depth "
            f"usually means the planes cannot be separated rather than that "
            f"the stage moved that far."
        )
    if lo == 0 or hi == n_z - 1:
        log.warning(
            "the chosen plane reaches the edge of the stack — true "
            "focus may lie outside the range acquired."
        )
    return chosen, all_scores


def focus_figure(arr, sizes, position, bf_index, n_z, chosen, scores, path, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.3])

    ax = fig.add_subplot(gs[0, :2])
    if scores:
        for t, sc in list(scores.items())[:6]:
            ax.plot(sc, marker="o", ms=3, label=f"frame {t}")
        ax.legend(fontsize=7)
    ax.set(
        xlabel="z plane",
        ylabel="sharpness",
        title="sharpness across the stack\n(a single clear peak means the "
        "choice is well defined)",
    )

    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(chosen, color="crimson", lw=2)
    ax.set(
        xlabel="frame",
        ylabel="plane used",
        ylim=(-0.5, n_z - 0.5),
        title="plane used over the movie\n(a steady drift is normal, a "
        "sawtooth is not)",
    )

    # the planes themselves at the first frame
    idx = axis_indexer(sizes, T=0, P=position, C=bf_index)
    stack = np.asarray(arr[idx])
    picks = sorted({0, n_z // 4, n_z // 2, 3 * n_z // 4, n_z - 1, chosen[0]})
    for j, z in enumerate(picks[:4]):
        ax = fig.add_subplot(gs[1, j])
        crop = stack[z]
        c = min(crop.shape) // 2
        crop = crop[
            max(0, crop.shape[0] // 2 - c // 2) : crop.shape[0] // 2 + c // 2,
            max(0, crop.shape[1] // 2 - c // 2) : crop.shape[1] // 2 + c // 2,
        ]
        ax.imshow(crop, cmap="gray")
        ax.set_title(
            f"z = {z}" + ("  <- chosen" if z == chosen[0] else ""),
            fontsize=9,
            color="crimson" if z == chosen[0] else "black",
        )
        ax.axis("off")

    fig.suptitle("Focus check — frame 0, phase channel", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"focus QC -> {path}")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument(
        "--positions", default="", help='which FOVs, e.g. "0 2". Default: all of them.'
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="work out the best plane for every frame, save it as "
        "focus_plan.csv with a QC figure, and write no "
        "images. Run this first for a z stack.",
    )
    ap.add_argument(
        "--replan",
        action="store_true",
        help="ignore any saved focus_plan.csv and choose again",
    )
    ap.add_argument(
        "--inspect-focus",
        action="store_true",
        help="write the focus diagnostics only, then stop",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite positions that were already split",
    )
    args = ap.parse_args()

    env = load_shell_config(args.config)
    log = get_logger("split", False)

    # Checked before the focus search, not after it. Running
    # ./split_nd2.py uses the first python on PATH, which is often not the
    # analysis environment — that python can have nd2 and numpy but no
    # matplotlib, so the reading finishes and the figure stage then fails.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        die(
            "matplotlib is not installed in the python being used: "
            f"{sys.executable}",
            "Activate the environment and call its python explicitly:\n"
            "                conda activate phage_pipeline\n"
            "                python tools/split_nd2.py -c <config> ...\n"
            "            ./split_nd2.py uses whatever python is first on "
            "PATH.",
        )

    try:
        import nd2
        import tifffile
    except ImportError as exc:
        die(
            f"the python package '{exc.name}' is not installed.",
            "Activate the analysis environment first, or  pip install nd2 tifffile",
        )

    nd2_path = Path(env["ND2_PATH"]).expanduser()
    if not nd2_path.exists():
        die(
            f"the nd2 file does not exist: {nd2_path}",
            "Check ND2_PATH in the config file.",
        )

    out_root = split_root(env)
    has_z = str(env.get("HAS_Z_STACK", "")).strip().upper() in ("TRUE", "YES", "1")
    z_method = env.get("Z_METHOD", "focus").strip() or "focus"
    if z_method not in ("consensus", "focus", "fixed", "max", "mean"):
        die(
            f"Z_METHOD is '{z_method}', which is not one of the options.",
            "Use  consensus, focus, fixed, max  or  mean .",
        )
    z_plane = int(float(env.get("Z_PLANE") or 0))
    z_max_step = int(float(env.get("Z_MAX_STEP") or 1))
    z_sample = int(float(env.get("Z_FOCUS_SAMPLE_EVERY") or 5))

    with nd2.ND2File(nd2_path) as f:
        sizes = dict(f.sizes)
        acq = read_acquisition(f, log)
        names = acq["channel_names"]
        log.info(f"{nd2_path.name}")
        log.info(f"  sizes    {sizes}")
        log.info(f"  channels {names}")

        n_t = int(sizes.get("T", 1))
        n_p = int(sizes.get("P", 1))
        n_z = int(sizes.get("Z", 1))
        has_z_any = n_z > 1

        ch_idx = {
            c: resolve_channel(f"CH_{c.upper()}", env[f"CH_{c.upper()}"], names, log)
            for c in CHANNELS
        }

        if args.positions.strip():
            positions = [int(x) for x in args.positions.replace(",", " ").split()]
        else:
            positions = list(range(n_p))
        log.info(f"  splitting positions {positions} of 0-{n_p - 1}")

        if n_z == 1:
            log.info("this file has a single plane, nothing to reduce")
            if has_z:
                log.warning(
                    "HAS_Z_STACK is TRUE but this file has only one "
                    "plane — the z settings are being ignored."
                )
        elif not has_z:
            log.warning(
                f"this file has {n_z} z planes but HAS_Z_STACK is FALSE, so "
                f"plane {z_plane} is being used for every frame. Set "
                f"HAS_Z_STACK=TRUE and run with --plan-only to choose "
                f"properly."
            )
            z_method = "fixed"

        try:
            arr = f.to_dask()  # lazy: never loads the whole file
            lazy = True
        except Exception:
            log.warning(
                "could not open lazily, reading the whole file into "
                "memory — this needs a lot of RAM"
            )
            arr = f.asarray()
            lazy = False
        log.info(f"  reading {'lazily' if lazy else 'all at once'}")

        out_root.mkdir(parents=True, exist_ok=True)

        for pos in positions:
            pdir = out_root / f"pos_{pos:02d}"
            # Test for the data, not the folder: --inspect-focus creates the
            # folder, and that must not make the real split skip itself.
            done = (
                all((pdir / f"{c}.tif").exists() for c in CHANNELS)
                and (pdir / "acquisition.json").exists()
            )

            # The z choice is baked into the TIFFs, so a split made under a
            # different Z_METHOD is stale. Skipping it silently means the
            # config says one thing and the data on disk is another — which
            # looks like the setting having no effect at all.
            if done:
                try:
                    prev = json.load(open(pdir / "acquisition.json"))
                except Exception:
                    prev = {}
                was = prev.get("z_method")
                was_plane = prev.get("z_plane_fixed")
                changed = n_z > 1 and was is not None and was != z_method
                if (
                    not changed
                    and z_method == "fixed"
                    and n_z > 1
                    and was_plane is not None
                    and int(was_plane) != z_plane
                ):
                    changed = True
                if changed and not args.force:
                    log.warning(
                        f"position {pos}: on disk it was split with "
                        f"Z_METHOD='{was}', but the config now says "
                        f"'{z_method}'. RE-SPLITTING, because the z choice is "
                        f"baked into the TIFFs and keeping the old ones would "
                        f"silently ignore the change."
                    )
                    done = False

            if done and not args.force and not args.inspect_focus:
                log.info(
                    f"position {pos}: already split with "
                    f"Z_METHOD='{z_method}', skipping (--force to redo)"
                )
                continue
            pdir.mkdir(parents=True, exist_ok=True)
            log.info(f"--- position {pos} ---")

            plan_path = pdir / "focus_plan.csv"
            saved = (
                None
                if (args.replan or args.plan_only)
                else read_focus_plan(plan_path, n_t, log)
            )
            if saved is not None and n_z > 1:
                chosen, scores = saved, None
            else:
                chosen, scores = choose_planes(
                    arr,
                    sizes,
                    pos if n_p > 1 else 0,
                    ch_idx["bf"],
                    n_z,
                    n_t,
                    z_method if n_z > 1 else "fixed",
                    z_plane,
                    z_max_step,
                    z_sample,
                    log,
                )
                if n_z > 1:
                    write_focus_plan(plan_path, chosen, scores, z_method, n_z, log)

            if n_z > 1:
                focus_figure(
                    arr,
                    sizes,
                    pos if n_p > 1 else 0,
                    ch_idx["bf"],
                    n_z,
                    chosen,
                    scores,
                    pdir / "focus_qc.png",
                    log,
                )
            if args.plan_only:
                log.info("--plan-only: the plan is saved, no images written")
                continue
            if args.inspect_focus:
                log.info("--inspect-focus: stopping before writing images")
                continue

            with Timer(log, f"position {pos}"):
                for name, ci in ch_idx.items():
                    planes = []
                    for t in range(n_t):
                        fixed = {"T": t, "C": ci}
                        if "P" in sizes:
                            fixed["P"] = pos if n_p > 1 else 0
                        if n_z > 1 and z_method in ("focus", "fixed", "consensus"):
                            fixed["Z"] = int(chosen[t])
                            plane = np.asarray(arr[axis_indexer(sizes, **fixed)])
                        elif n_z > 1:
                            stack = np.asarray(arr[axis_indexer(sizes, **fixed)])
                            plane = reduce_z(stack, z_method, 0)
                        else:
                            plane = np.asarray(arr[axis_indexer(sizes, **fixed)])
                        planes.append(np.asarray(plane, dtype=np.uint16))
                    stack = np.stack(planes)
                    tifffile.imwrite(pdir / f"{name}.tif", stack)
                    log.info(
                        f"  {name}.tif  {stack.shape}  " f"{stack.nbytes / 1e6:.0f} MB"
                    )

            info = dict(acq)
            info.update(
                {
                    "position": pos,
                    "n_frames": n_t,
                    "shape_yx": list(stack.shape[1:]),
                    "channel_index": ch_idx,
                    "z_method": z_method,
                    "z_plane_fixed": z_plane if z_method == "fixed" else None,
                    "z_planes_used": [int(z) for z in chosen] if n_z > 1 else None,
                    "n_z_in_file": n_z,
                    "source_nd2": str(nd2_path),
                }
            )
            with open(pdir / "acquisition.json", "w") as fh:
                json.dump(info, fh, indent=2)
            if n_z > 1:
                with open(pdir / "focus_planes.csv", "w") as fh:
                    fh.write("frame,z_plane\n")
                    for t, z in enumerate(chosen):
                        fh.write(f"{t},{z}\n")

    if args.plan_only and not has_z_any:
        log.warning(
            "no field had more than one z plane, so there is no focus " "plan to make."
        )
        return 0

    if args.plan_only:
        log.info("")
        log.info(f"focus plans written under {out_root}")
        log.info(
            "Check focus_qc.png, and edit focus_plan.csv by hand if a "
            "frame picked the wrong plane."
        )
        log.info("Then split for real (the plan is reused, not recomputed):")
        log.info(f"    python tools/split_nd2.py -c {args.config}")
        return 0

    if args.inspect_focus:
        log.info(f"focus figures written under {out_root}")
        log.info(
            "If the chosen plane looks right, run again without "
            "--inspect-focus to write the images."
        )
        return 0

    log.info(f"done -> {out_root}")
    log.info("Now set these in the config and re-run the pipeline:")
    log.info(f'    INPUT_SOURCE="split"')
    log.info(f'    SPLIT_DIR="{out_root}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

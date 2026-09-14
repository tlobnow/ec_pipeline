#!/usr/bin/env python3
"""
Step 01 — load one field of view from the nd2 and write per-channel TIFFs.

Outputs (in <pos>/01_load/):
    bf.tif, gfp.tif, rfp.tif   uint16, shape (T, Y, X)
    acquisition.json           pixel size, frame interval, channel names
    meta.json                  provenance

Also does the channel sanity check that used to be a manual notebook cell:
PI/RFP should rise over time and GFP should not. A warning is logged if the
channel assignment looks reversed, rather than leaving it to be noticed later.

Utility mode:
    01_load_nd2.py --config config.sh --count      # print number of positions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    Timer,
    base_parser,
    die,
    frame_times_from_events,
    get_logger,
    load_config,
    programmed_interval_min,
    reduce_z,
    require,
    resolve_channel,
    run_safely,
    step_dir,
    write_meta,
)


def axis_indexer(sizes, **fixed):
    """Index tuple following the file's own axis order, from f.sizes."""
    return tuple(fixed.get(a, slice(None)) for a in sizes)


def load_from_split(cfg, pos, out, log):
    """Copy a pre-split position into place, instead of re-reading the nd2."""
    import shutil

    src = Path(cfg["load"]["split_dir"]).expanduser() / f"pos_{pos:02d}"
    require(
        src,
        "the pre-split position folder",
        "Run  python tools/split_nd2.py -c <config>  first, or set "
        'INPUT_SOURCE="nd2" to read the nd2 directly.',
    )

    acq_src = require(
        src / "acquisition.json",
        "the split metadata",
        "Re-run tools/split_nd2.py for this position.",
    )
    with open(acq_src) as fh:
        acq = json.load(fh)

    written = {}
    for name in ("bf", "gfp", "rfp"):
        s = require(
            src / f"{name}.tif",
            f"the {name} channel",
            "Re-run tools/split_nd2.py for this position.",
        )
        shutil.copy2(s, out / f"{name}.tif")
        import tifffile

        written[name] = tifffile.imread(out / f"{name}.tif")
        log.info(f"{name}.tif  {written[name].shape}  from the split folder")

    if acq.get("z_method"):
        planes = acq.get("z_planes_used")
        log.info(
            f"z was reduced by '{acq['z_method']}'"
            + (
                f", planes {min(planes)}-{max(planes)} of "
                f"0-{acq['n_z_in_file'] - 1}"
                if planes
                else ""
            )
        )
    return written, acq, src


def n_positions(nd2_path) -> int:
    import nd2

    with nd2.ND2File(nd2_path) as f:
        return int(dict(f.sizes).get("P", 1))


def read_acquisition(f, position, log) -> dict:
    """Pixel size, channel names, and both versions of the frame interval."""
    acq = {
        "pixel_size_um": None,
        "frame_interval_min": None,
        "channel_names": [],
        "sizes": dict(f.sizes),
        "frame_interval_recorded_min": None,
        "frame_interval_programmed_min": None,
        "frame_interval_uneven": False,
        "frame_times_s": None,
    }
    try:
        acq["pixel_size_um"] = float(f.voxel_size().x)
    except Exception:
        pass
    try:
        acq["channel_names"] = [c.channel.name for c in f.metadata.channels]
    except Exception:
        pass

    times = frame_times_from_events(f, position)
    if times is not None and len(times) > 1:
        d = np.diff(times)
        acq["frame_interval_recorded_min"] = float(np.median(d) / 60.0)
        acq["frame_times_s"] = [round(float(x - times[0]), 3) for x in times]
        if d.max() - d.min() > 0.1 * np.median(d):
            acq["frame_interval_uneven"] = True
            log.warning(
                f"frame spacing is uneven: {d.min()/60:.2f} to {d.max()/60:.2f} "
                f"min (median {np.median(d)/60:.2f}). A single interval is an "
                f"approximation here; the real times are in frame_times.csv."
            )

    acq["frame_interval_programmed_min"] = programmed_interval_min(f)
    return acq


def main(argv=None):
    p = base_parser(__doc__.split("\n")[1])
    p.add_argument(
        "--count",
        action="store_true",
        help="print number of positions in the nd2 and exit",
    )
    args = p.parse_args(argv)
    cfg = load_config()
    log = get_logger("01_load", args.quiet)

    nd2_path = require(
        Path(cfg["experiment"]["nd2_path"]).expanduser(),
        "the nd2 file",
        "Check ND2_PATH in config.sh.",
    )

    if args.count:
        print(n_positions(nd2_path))
        return 0

    lc = cfg["load"]
    params = dict(lc, nd2_path=str(nd2_path), position=args.position)
    out = step_dir(cfg, args.position, "01_load", create=True)

    if lc["input_source"] == "split":
        written, acq, src = load_from_split(cfg, args.position, out, log)
        T = int(acq.get("n_frames") or written["bf"].shape[0])
        acq.setdefault("pixel_size_source", "split folder")
        acq.setdefault("frame_interval_source", "split folder")
        if lc.get("frame_interval_min_override"):
            acq["frame_interval_min"] = float(lc["frame_interval_min_override"])
            acq["frame_interval_source"] = "config override"
        if lc.get("pixel_size_um_override"):
            acq["pixel_size_um"] = float(lc["pixel_size_um_override"])
            acq["pixel_size_source"] = "config override"
        if not acq.get("frame_interval_min"):
            die(
                "the split folder does not record a frame interval.",
                "Set FRAME_INTERVAL_MIN in config.sh.",
            )
        if not acq.get("pixel_size_um"):
            die(
                "the split folder does not record a pixel size.",
                "Set PIXEL_SIZE_UM in config.sh.",
            )
        log.info(
            f"pixel size {acq['pixel_size_um']:.4f} um/px, "
            f"interval {acq['frame_interval_min']:.3f} min, {T} frames"
        )
        with open(out / "acquisition.json", "w") as fh:
            json.dump(acq, fh, indent=2)
        with open(out / "frame_times.csv", "w") as fh:
            fh.write("frame,time_min,source\n")
            for i in range(T):
                fh.write(f"{i},{i * acq['frame_interval_min']:.4f},assumed\n")
        write_meta(
            out,
            "01_load",
            params,
            {"split": src},
            {
                "n_frames": T,
                "shape_yx": list(written["bf"].shape[1:]),
                "input_source": "split",
                "pixel_size_um": acq["pixel_size_um"],
                "frame_interval_min": acq["frame_interval_min"],
            },
        )
        log.info(f"done -> {out}")
        return 0

    import nd2
    import tifffile

    with Timer(log, "read nd2"):
        with nd2.ND2File(nd2_path) as f:
            acq = read_acquisition(f, args.position, log)
            sizes = dict(f.sizes)
            names = acq.get("channel_names") or []
            log.info(f"{nd2_path.name}  sizes={sizes}  channels={names}")

            n_t = int(sizes.get("T", 1))
            n_p = int(sizes.get("P", 1))
            n_z = int(sizes.get("Z", 1))
            n_c = int(sizes.get("C", 1))
            if args.position >= n_p:
                die(
                    f"position {args.position} was requested but this nd2 has "
                    f"only {n_p} position(s), numbered 0 to {n_p - 1}.",
                    'Fix POSITIONS in config.sh, or set POSITIONS="all".',
                )

            # Channels by name where possible: this matters because files are
            # not always ordered phase first.
            idx = {
                k: resolve_channel(f"CH_{k.upper()}", lc[f"ch_{k}"], names, log)
                for k in ("bf", "gfp", "rfp")
            }
            if max(idx.values()) >= n_c:
                die(
                    f"the config asks for channel {max(idx.values())} but this "
                    f"file has only {n_c} channel(s), numbered 0 to {n_c - 1}.",
                    "Fix CH_BF / CH_GFP / CH_RFP in config.sh.",
                )

            if n_z > 1:
                log.warning(
                    f"this file has {n_z} z planes. They are being reduced by "
                    f"'{lc['z_method']}' while the whole file is read, which "
                    f"is slow. Running tools/split_nd2.py once is much faster "
                    f"for repeated analysis."
                )

            # Index by axis NAME. Deriving the layout from the shape is what
            # failed on this file: a z axis made (T, Z, C, Y, X) look wrong.
            try:
                arr = f.to_dask()
                lazy = True
            except Exception:
                arr = f.asarray()
                lazy = False
            log.info(
                f"reading {'lazily' if lazy else 'all at once'}, "
                f"T={n_t} C={n_c} Z={n_z} P={n_p}"
            )

            chosen = [int(lc["z_plane"])] * n_t
            if n_z > 1 and lc["z_method"] == "focus":
                from common import best_plane, smooth_plane_choices

                step = max(1, int(lc["z_focus_sample_every"]))
                frames = sorted({*range(0, n_t, step), n_t - 1})
                meas = {}
                for tt in frames:
                    fixed = {"T": tt, "C": idx["bf"]}
                    if n_p > 1:
                        fixed["P"] = args.position
                    meas[tt] = best_plane(
                        np.asarray(arr[axis_indexer(sizes, **fixed)])
                    )[0]
                chosen = [
                    (
                        meas.get(tt)
                        if tt in meas
                        else meas[max(x for x in meas if x <= tt)]
                    )
                    for tt in range(n_t)
                ]
                chosen = smooth_plane_choices(chosen, int(lc["z_max_step"]), log)
                log.info(
                    f"focus: using planes {min(chosen)}-{max(chosen)} "
                    f"of 0-{n_z - 1}"
                )

            raw = {}
            for name, ci in idx.items():
                planes = []
                for tt in range(n_t):
                    fixed = {"T": tt, "C": ci}
                    if n_p > 1:
                        fixed["P"] = args.position
                    if n_z > 1 and lc["z_method"] in ("focus", "fixed"):
                        fixed["Z"] = int(chosen[tt])
                        planes.append(np.asarray(arr[axis_indexer(sizes, **fixed)]))
                    elif n_z > 1:
                        planes.append(
                            reduce_z(
                                np.asarray(arr[axis_indexer(sizes, **fixed)]),
                                lc["z_method"],
                                0,
                            )
                        )
                    else:
                        planes.append(np.asarray(arr[axis_indexer(sizes, **fixed)]))
                raw[name] = np.stack(planes)

    T = n_t
    Y, X = raw["bf"].shape[1], raw["bf"].shape[2]
    acq["z_method"] = lc["z_method"] if n_z > 1 else None
    acq["z_planes_used"] = [int(z) for z in chosen] if n_z > 1 else None
    acq["n_z_in_file"] = n_z

    # Centre crop, clipped to the image so an oversized crop cannot silently
    # produce an empty or asymmetric array.
    half = lc.get("crop_half_size")
    if half:
        cy, cx = Y // 2, X // 2
        h = int(min(half, cy, cx))
        if h != half:
            log.warning(f"crop_half_size {half} exceeds image; clipped to {h}")
        sl = np.s_[:, cy - h : cy + h, cx - h : cx + h]
    else:
        sl = np.s_[:, :, :]

    written = {}
    for name in ("bf", "gfp", "rfp"):
        a = raw[name][sl].astype(np.uint16)
        tifffile.imwrite(out / f"{name}.tif", a)
        written[name] = a
        log.info(f"{name}.tif  shape={a.shape}  " f"range=[{a.min()}, {a.max()}]")

    # Channel diagnostic: PI/RFP must rise, GFP must not.
    def trend(a):
        n = max(1, len(a) // 5)
        return float(np.median(a[-n:]) - np.median(a[:n]))

    gfp_t = trend([float(x.max()) for x in written["gfp"]])
    rfp_t = trend([float(x.max()) for x in written["rfp"]])
    log.info(f"channel trend (late - early max):  gfp={gfp_t:+.0f}  rfp={rfp_t:+.0f}")
    swapped = rfp_t <= 0 < gfp_t
    if swapped:
        log.warning(
            "RFP does not rise while GFP does — ch_gfp/ch_rfp may be "
            "swapped. Check before trusting downstream death calls."
        )

    # ── frame interval ─────────────────────────────────────────────────
    # Order of preference:
    #   1. FRAME_INTERVAL_MIN in config.sh, if set (an explicit override)
    #   2. the timestamps the microscope actually recorded
    #   3. the period programmed into the acquisition software
    rec = acq["frame_interval_recorded_min"]
    prog = acq["frame_interval_programmed_min"]
    override = lc.get("frame_interval_min_override")

    if rec is not None and prog is not None:
        ratio = rec / prog if prog else 0
        if abs(rec - prog) > 0.02 * max(rec, prog):
            # A recorded value that is a neat fraction of the programmed one
            # usually means the timestamps were divided by the wrong number of
            # intervals, not that the microscope ran fast.
            near = min(
                (n for n in (2, 3, 4, 5, 6, 8, 10) if abs(ratio - 1.0 / n) < 0.02),
                default=None,
            )
            if near:
                log.warning(
                    f"the recorded interval ({rec:.3f} min) is almost exactly "
                    f"1/{near} of the programmed one ({prog:.3f} min). That "
                    f"pattern means the timestamps were counted wrongly, not "
                    f"that the run was faster — using the programmed value."
                )
                rec = None
            else:
                log.warning(
                    f"the nd2 disagrees with itself about timing: recorded "
                    f"{rec:.3f} min/frame, but {prog:.3f} min/frame was "
                    f"programmed. The recorded timestamps are what actually "
                    f"happened."
                )

    if override:
        acq["frame_interval_min"] = float(override)
        acq["frame_interval_source"] = "config override"
    elif rec is not None:
        acq["frame_interval_min"] = rec
        acq["frame_interval_source"] = "recorded timestamps"
    elif prog is not None:
        acq["frame_interval_min"] = prog
        acq["frame_interval_source"] = "programmed period (no timestamps found)"
        log.warning(
            "no per-frame timestamps in this file, so the programmed "
            "period is being used. That is what was asked for, not "
            "necessarily what happened."
        )
    else:
        die(
            "the frame interval could not be found in the nd2 file.",
            "Set it yourself in config.sh, for example  FRAME_INTERVAL_MIN=2.5",
        )

    # ── pixel size ─────────────────────────────────────────────────────
    px_override = lc.get("pixel_size_um_override")
    if px_override:
        acq["pixel_size_um"] = float(px_override)
        acq["pixel_size_source"] = "config override"
    elif acq["pixel_size_um"]:
        acq["pixel_size_source"] = "nd2"
    else:
        die(
            "the pixel size could not be found in the nd2 file.",
            "Set it yourself in config.sh, for example  PIXEL_SIZE_UM=0.065",
        )

    acq["n_frames"] = T
    acq["shape_yx"] = list(written["bf"].shape[1:])
    log.info(
        f"pixel size {acq['pixel_size_um']:.4f} um/px " f"({acq['pixel_size_source']})"
    )
    log.info(
        f"frame interval {acq['frame_interval_min']:.4f} min "
        f"({acq['frame_interval_source']})"
    )
    if rec is not None:
        log.info(
            f"  recorded timestamps say {rec:.4f} min/frame; "
            f"total run {rec * (T - 1):.1f} min"
        )
    if prog is not None:
        log.info(f"  programmed period says   {prog:.4f} min/frame")

    # Real time of every frame, for anything that needs exact timing.
    times = acq.pop("frame_times_s", None)
    with open(out / "frame_times.csv", "w") as fh:
        fh.write("frame,time_min,source\n")
        for i in range(T):
            if times is not None and i < len(times):
                fh.write(f"{i},{times[i] / 60.0:.4f},recorded\n")
            else:
                fh.write(f"{i},{i * acq['frame_interval_min']:.4f},assumed\n")

    with open(out / "acquisition.json", "w") as fh:
        json.dump(acq, fh, indent=2)

    write_meta(
        out,
        "01_load",
        params,
        {"nd2": nd2_path},
        {
            "n_frames": T,
            "n_channels": n_c,
            "shape_yx": acq["shape_yx"],
            "n_z_in_file": n_z,
            "z_method": acq["z_method"],
            "channels_possibly_swapped": bool(swapped),
            "pixel_size_um": acq["pixel_size_um"],
            "frame_interval_min": acq["frame_interval_min"],
        },
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "load nd2"))

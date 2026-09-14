#!/usr/bin/env python3
"""
Show every timing number an nd2 file carries, so you can see which one is
right and where a disagreement comes from.

    python tools/check_nd2_timing.py /path/to/file.nd2
    python tools/check_nd2_timing.py /path/to/file.nd2 --position 0

An nd2 holds two different things:

  PROGRAMMED  what was typed into NIS-Elements before the run. Fast to read,
              but it is an intention, not a measurement. If the microscope
              could not keep up, or the operator changed the interval, this
              number stays as it was.

  RECORDED    a timestamp written for each frame as it was actually taken.
              This is the ground truth.

If the two disagree, trust RECORDED.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "scripts"))
try:
    from common import _find_key as find_key, _find_time_key
except ImportError:  # standalone copy

    def find_key(keys, *musts, forbid=()):
        for k in keys:
            low = str(k).lower()
            if all(m in low for m in musts) and not any(f in low for f in forbid):
                return k
        return None

    _find_time_key = None


def dump_experiment(f):
    print("\n--- PROGRAMMED (acquisition loop definition) --------------------")
    try:
        loops = list(f.experiment)
    except Exception as exc:
        print(f"  could not read: {exc}")
        return
    if not loops:
        print("  none recorded")
        return
    for i, loop in enumerate(loops):
        kind = type(loop).__name__
        count = getattr(loop, "count", "?")
        print(f"  loop {i}: {kind}, count={count}")
        params = getattr(loop, "parameters", None)
        if params is None:
            continue
        for attr in ("periodMs", "durationMs", "periodDiff", "startMs"):
            if hasattr(params, attr):
                print(f"      {attr} = {getattr(params, attr)}")
        # NETimeLoop keeps its phases in a list
        for j, ph in enumerate(getattr(params, "periods", []) or []):
            print(
                f"      phase {j}: count={getattr(ph, 'count', '?')} "
                f"periodMs={getattr(ph, 'periodMs', '?')} "
                f"durationMs={getattr(ph, 'durationMs', '?')}"
            )


def frame_times_from_events(f, position=None, verbose=True):
    """Actual per-frame timestamps, in seconds, or None."""
    try:
        evs = f.events()
    except Exception as exc:
        if verbose:
            print(f"  events() failed: {exc}")
        return None
    evs = list(evs) if evs is not None else []
    if not evs:
        if verbose:
            print("  events() returned nothing")
        return None

    rows = []
    for e in evs:
        if isinstance(e, dict):
            rows.append(e)
        else:
            d = getattr(e, "__dict__", None)
            if d:
                rows.append(d)
    if not rows:
        if verbose:
            print("  events present but not readable as records")
        return None

    keys = list(rows[0].keys())
    if verbose:
        print(f"  {len(rows)} event records, fields: {keys}")

    if _find_time_key is not None:
        tkey, scale = _find_time_key(keys)
    else:
        t_ms = find_key(keys, "time", "ms", forbid=("exposure",))
        tkey, scale = (
            (t_ms, 1e-3)
            if t_ms
            else (find_key(keys, "time", forbid=("ms", "exposure")), 1.0)
        )
    if tkey is None:
        if verbose:
            print("  no time field found in events")
        return None
    if verbose:
        others = [k for k in keys if "time" in str(k).lower() and k != tkey]
        if others:
            print(
                f"  using '{tkey}'; ignoring {others} — a field such as "
                f"exposure time is the same on every frame and would make "
                f"the run appear to span zero minutes"
            )

    ikey = find_key(keys, "t", "index") or find_key(keys, "index", forbid=("p ", "z "))
    pkey = find_key(keys, "p", "index") or find_key(keys, "position", "name")

    times = {}
    for r in rows:
        val = r.get(tkey)
        if val is None:
            continue
        if position is not None and pkey is not None:
            pv = r.get(pkey)
            try:
                if int(pv) != int(position):
                    continue
            except (TypeError, ValueError):
                pass  # position name, not an index
        idx = r.get(ikey) if ikey else len(times)
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            idx = len(times)
        times.setdefault(idx, []).append(float(val) * scale)

    if len(times) < 2:
        if verbose:
            print(f"  only {len(times)} distinct timepoint(s) after filtering")
        return None
    ordered = [np.mean(times[k]) for k in sorted(times)]
    if verbose:
        print(
            f"  time field '{tkey}', index field '{ikey}', "
            f"position field '{pkey}' -> {len(ordered)} timepoints"
        )
    return np.array(ordered)


def dump_frame_metadata(f, n_frames, n_pos=1):
    """
    Second opinion: relative timestamps stored per frame.

    The index here counts EVERY frame, so in a 5-position file the frames go
    p0 p1 p2 p3 p4 for timepoint 0, then again for timepoint 1. Treating that
    index as a timepoint divides the elapsed time by five times too many
    intervals — which is how a 2 min interval came out as 0.4 min.
    """
    print("\n--- RECORDED (per-frame metadata) ------------------------------")
    if n_pos > 1:
        print(f"  {n_pos} positions, so frame index = timepoint x {n_pos}")
    got = []
    for i in (
        0,
        n_pos,
        2 * n_pos,
        max(0, (n_frames - 2) * n_pos),
        max(0, (n_frames - 1) * n_pos),
    ):
        try:
            fm = f.frame_metadata(i)
            ch = fm.channels[0]
            got.append((i, float(ch.time.relativeTimeMs) / 1000.0))
        except Exception as exc:
            print(f"  frame {i}: not readable ({type(exc).__name__})")
            return
    for i, t in got:
        print(f"  frame {i:5d}  (timepoint {i // max(n_pos, 1):4d}): " f"{t:10.2f} s")
    if len(got) >= 2 and got[-1][0] > got[0][0]:
        span = got[-1][1] - got[0][1]
        n_int = (got[-1][0] - got[0][0]) / max(n_pos, 1)
        if n_int > 0:
            print(
                f"  -> mean interval between timepoints: "
                f"{span / n_int / 60.0:.4f} min ({span / n_int:.2f} s)"
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("nd2")
    ap.add_argument("--position", type=int, default=None)
    args = ap.parse_args()

    try:
        import nd2
    except ImportError:
        sys.exit("the 'nd2' package is not installed in this environment")

    with nd2.ND2File(args.nd2) as f:
        sizes = dict(f.sizes)
        print(f"file  : {args.nd2}")
        print(f"sizes : {sizes}")
        n_frames = int(sizes.get("T", 1))

        try:
            print(f"pixel : {f.voxel_size()}")
        except Exception:
            pass

        dump_experiment(f)

        print("\n--- RECORDED (event timestamps) --------------------------------")
        times = frame_times_from_events(f, args.position)
        if times is not None and len(times) > 1:
            d = np.diff(times)
            print(
                f"  {len(times)} timepoints spanning "
                f"{(times[-1] - times[0]) / 60.0:.2f} min"
            )
            print(
                f"  interval  median {np.median(d) / 60.0:.4f} min"
                f"   mean {d.mean() / 60.0:.4f} min"
            )
            print(
                f"            min {d.min() / 60.0:.4f}"
                f"   max {d.max() / 60.0:.4f} min"
            )
            if d.max() - d.min() > 0.1 * np.median(d):
                print(
                    "  NOTE: the spacing between frames is uneven — the "
                    "microscope did not keep a constant rhythm."
                )

        dump_frame_metadata(f, n_frames, int(sizes.get("P", 1)))

        print("\n--- WHAT TO DO -------------------------------------------------")
        print("  If PROGRAMMED and RECORDED disagree, the recorded timestamps")
        print("  are what actually happened. Put that value in config.sh as")
        print("  FRAME_INTERVAL_MIN=<value> to pin it explicitly.")


if __name__ == "__main__":
    main()

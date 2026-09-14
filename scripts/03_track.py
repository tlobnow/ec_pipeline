#!/usr/bin/env python
"""
Step 03 — track cells over time and relabel the masks with track ids.

Outputs (in <pos>/03_track/):
    tracked_masks.npz   int32 stack, pixel value = track id
    lineage.csv         one row per track: parent, root, generation, lifespan
    divisions.csv       one row per division event
    qc_tracking.png
    meta.json

Division handling
-----------------
A cell keeps ONE track id from the moment it appears until it divides. At a
division the mother track ends and the two daughters start as NEW ids, each
recording parent_id. So in the QC overlay a cell holds its colour until it
splits, then the daughters take two new colours — which is what makes division
events visible at a glance.

lineage.csv also carries root_id (constant for an entire lineage, including
through every division) and generation, so downstream analysis can group by
individual cell or by founder lineage without re-deriving anything.

Relabelling
-----------
Each track is mapped back to the exact segmentation label it was built from,
via btrack's object references. The old nearest-centroid lookup (which
majority-voted over a +/-8 px window and could mis-assign in dense fields) is
kept only as an automatic fallback, and the log records which was used.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    require_settings,
    Timer,
    base_parser,
    build_cell_names,
    die,
    get_logger,
    load_config,
    require,
    run_safely,
    step_dir,
    write_meta,
)

# The steps and common.py are updated together. Checked here so a half-copied
# folder says so, instead of failing later with a KeyError on a setting that
# the older common.py never read.
_EXPECT_COMMON = "0.10.0"
import common as _common  # noqa: E402

if getattr(_common, "PIPELINE_VERSION", "0") != _EXPECT_COMMON:
    die(
        f"this script expects common.py version {_EXPECT_COMMON}, but the "
        f"one next to it is "
        f"{getattr(_common, 'PIPELINE_VERSION', 'older than 0.8.0')}.",
        "Copy the whole pipeline folder across, not single files — "
        "scripts/ and tools/ have to match.\n"
        "            The file to update is scripts/common.py.",
    )


# ── object construction ─────────────────────────────────────────────────────
def build_objects(masks, log):
    """PyTrackObjects with explicit IDs, plus an id -> (frame, label) map."""
    import btrack
    from skimage import measure

    objects, id_map = [], {}
    oid = 0
    for t in range(masks.shape[0]):
        if not masks[t].any():
            continue
        for reg in measure.regionprops(masks[t]):
            cy, cx = reg.centroid
            o = btrack.btypes.PyTrackObject()
            o.ID = oid
            o.x, o.y, o.z, o.t = float(cx), float(cy), 0.0, int(t)
            o.label = 0
            try:
                o.properties = {
                    "area": float(reg.area),
                    "eccentricity": float(reg.eccentricity),
                    "solidity": float(reg.solidity),
                }
            except Exception:
                pass
            objects.append(o)
            id_map[oid] = (int(t), int(reg.label))
            oid += 1
    log.info(
        f"objects: {len(objects)} "
        f"({len(objects) / max(masks.shape[0], 1):.0f}/frame)"
    )
    return objects, id_map


# ── relabelling ─────────────────────────────────────────────────────────────
def relabel_by_refs(tracks, id_map, masks, log):
    """Exact mapping: track -> the segmentation labels it was built from."""
    per_frame = defaultdict(dict)  # frame -> {seg_label: track_id}
    unresolved = 0
    for tr in tracks:
        for ref in getattr(tr, "refs", []):
            ref = int(ref)
            if ref < 0:  # dummy/interpolated detection
                unresolved += 1
                continue
            tl = id_map.get(ref)
            if tl is None:
                unresolved += 1
                continue
            per_frame[tl[0]][tl[1]] = int(tr.ID)
    log.info(
        f"refs mapping: {sum(len(v) for v in per_frame.values())} "
        f"assignments, {unresolved} dummy/unmatched"
    )
    return per_frame


def relabel_by_centroid(tracks, masks, tol, log):
    """Fallback: nearest-label vote around each track's centroid."""
    H, W = masks.shape[1], masks.shape[2]
    per_frame = defaultdict(dict)
    for tr in tracks:
        for t, x, y in zip(tr.t, tr.x, tr.y):
            t = int(t)
            if t >= masks.shape[0]:
                continue
            cy, cx = int(round(y)), int(round(x))
            win = masks[t][
                max(0, cy - tol) : min(H, cy + tol + 1),
                max(0, cx - tol) : min(W, cx + tol + 1),
            ]
            win = win[win > 0]
            if win.size:
                per_frame[t][int(np.bincount(win).argmax())] = int(tr.ID)
    log.info("centroid fallback mapping built")
    return per_frame


def apply_mapping(masks, per_frame, keep_ids):
    """Vectorised per-frame label -> track id substitution."""
    out = np.zeros(masks.shape, dtype=np.int32)
    for t in range(masks.shape[0]):
        m = masks[t]
        if not m.any():
            continue
        lut = np.zeros(int(m.max()) + 1, dtype=np.int32)
        for lbl, tid in per_frame.get(t, {}).items():
            if lbl < lut.size and tid in keep_ids:
                lut[lbl] = tid
        out[t] = lut[m]
    return out


def split_suspect_tracks(per_frame, masks, px_um, tc, log):
    """
    Break a track where it stops being physically plausible.

    When a cell ruptures it vanishes, and the tracker will happily continue
    that id on whichever neighbour is closest — so one id ends up covering two
    different cells and any single-cell movie built from it is a chimera.

    Compared PER FRAME, on the union of whatever labels carry that id in each
    frame. A track legitimately holds several labels in one frame when
    segmentation has split a cell, and comparing those labels against each
    other instead of against the next frame gives an overlap of exactly zero
    — they are disjoint by construction — which cut half of all links.
    """
    from collections import defaultdict
    from skimage import measure

    min_overlap = float(tc["min_overlap"])
    max_step = float(tc["max_step_um"])
    max_ratio = float(tc["max_area_ratio"])
    max_gap = int(tc["max_gap_frames"])

    # frame -> track id -> the labels carrying that id in that frame
    per_tid = defaultdict(dict)
    for fr, mapping in per_frame.items():
        by_tid = defaultdict(list)
        for lbl, tid in mapping.items():
            by_tid[int(tid)].append(int(lbl))
        for tid, labels in by_tid.items():
            per_tid[tid][int(fr)] = labels

    def footprint(fr, labels):
        return np.isin(masks[fr], labels)

    def stats(fr, labels):
        m = footprint(fr, labels)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            return None
        return (
            ys.mean(),
            xs.mean(),
            int(ys.size),
            (ys.min(), xs.min(), ys.max() + 1, xs.max() + 1),
        )

    next_id = max(per_tid) + 1 if per_tid else 1
    breaks, segments, links = [], defaultdict(list), []

    for tid in sorted(per_tid):
        frames = sorted(per_tid[tid])
        if len(frames) < 2:
            segments[tid].append((frames[0], frames[0], tid))
            continue
        current, seg_start = tid, frames[0]

        for i in range(1, len(frames)):
            t0, t1 = frames[i - 1], frames[i]
            a = stats(t0, per_tid[tid][t0])
            b = stats(t1, per_tid[tid][t1])
            if a is None or b is None:
                continue
            cy0, cx0, area0, bb0 = a
            cy1, cx1, area1, bb1 = b

            gap = t1 - t0 - 1
            step_um = float(np.hypot(cy1 - cy0, cx1 - cx0)) * px_um / max(t1 - t0, 1)
            ratio = max(area0, area1) / max(min(area0, area1), 1)

            y0, x0 = max(bb0[0], bb1[0]), max(bb0[1], bb1[1])
            y1, x1 = min(bb0[2], bb1[2]), min(bb0[3], bb1[3])
            if y1 <= y0 or x1 <= x0:
                ov = 0.0
            else:
                m0 = footprint(t0, per_tid[tid][t0])[y0:y1, x0:x1]
                m1 = footprint(t1, per_tid[tid][t1])[y0:y1, x0:x1]
                ov = float((m0 & m1).sum()) / max(min(area0, area1), 1)

            links.append(
                {
                    "track_id": tid,
                    "frame": t1,
                    "overlap": ov,
                    "step_um": step_um,
                    "area_ratio": ratio,
                    "gap": gap,
                }
            )

            reason = None
            if gap >= max_gap:
                reason = f"missing for {gap} frame(s)"
            elif ov < min_overlap:
                reason = (
                    f"overlaps the previous frame by only "
                    f"{ov * 100:.0f}% (limit {min_overlap * 100:.0f}%)"
                )
            elif step_um > max_step:
                reason = f"moved {step_um:.2f} um/frame (limit {max_step})"
            elif ratio > max_ratio:
                reason = f"area changed {ratio:.1f}x (limit {max_ratio})"

            if reason:
                segments[tid].append((seg_start, t0, current))
                new_id = next_id
                next_id += 1
                breaks.append(
                    {
                        "original_id": tid,
                        "new_id": new_id,
                        "frame": t1,
                        "reason": reason,
                        "overlap": round(ov, 3),
                        "step_um": round(step_um, 3),
                        "area_ratio": round(ratio, 2),
                        "gap": gap,
                    }
                )
                current, seg_start = new_id, t1
            if current != tid:
                for lbl in per_tid[tid][t1]:
                    per_frame[t1][lbl] = current
        segments[tid].append((seg_start, frames[-1], current))

    if breaks:
        kinds = defaultdict(int)
        for b in breaks:
            kinds[b["reason"].split()[0]] += 1
        detail = ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
        log.warning(
            f"{len(breaks)} suspect link(s) cut ({detail}), affecting "
            f"{len(set(b['original_id'] for b in breaks))} track(s). "
            f"These are cells the tracker joined that are probably not "
            f"the same cell — see track_breaks.csv."
        )
        per_track = defaultdict(int)
        for b in breaks:
            per_track[b["original_id"]] += 1
        worst = sorted(per_track.items(), key=lambda kv: -kv[1])[:5]
        log.info(
            "most-cut track(s): " + ", ".join(f"id {k} cut {v}x" for k, v in worst)
        )
        if worst and worst[0][1] > 5:
            log.warning(
                f"track {worst[0][0]} was cut {worst[0][1]} times. A cell does "
                f"not change identity repeatedly — loosen TRACK_MIN_OVERLAP, "
                f"TRACK_MAX_STEP_UM and TRACK_MAX_AREA_RATIO, or set "
                f"TRACK_SPLIT_SUSPECT=FALSE and rely on the overlap tracker."
            )
    else:
        log.info("no suspect links found")

    if links:
        ov_all = np.array([l["overlap"] for l in links])
        log.info(
            f"link continuity: median overlap {np.median(ov_all):.2f}, "
            f"{100 * (ov_all < min_overlap).mean():.1f}% below the limit "
            f"of {min_overlap}"
        )
    return per_frame, breaks, segments, links


def reparent_after_split(parent, segments, span_start, log):
    """
    Point each daughter at the segment of its mother that it actually came
    from, so a division that happened after a cut is still attached correctly.
    """
    fixed = 0
    for child, mother in list(parent.items()):
        if mother is None or mother not in segments:
            continue
        birth = span_start.get(child)
        if birth is None:
            continue
        for start, end, seg_id in segments[mother]:
            if start <= birth - 1 <= end and seg_id != mother:
                parent[child] = seg_id
                fixed += 1
                break
    if fixed:
        log.info(
            f"{fixed} daughter(s) re-attached to the correct part of a "
            f"track that was cut"
        )
    return parent


def track_with_btrack(masks, tc, btrack_cfg, log):
    """The original Bayesian tracker, kept as an alternative."""
    import btrack

    objects, id_map = build_objects(masks, log)
    if not objects:
        die(
            "the segmentation contains no cells at all, so there is nothing "
            "to track.",
            "Look at 02_segment/qc_segmentation.png. If it is blank, lower "
            "MASK_THRESHOLD in config.sh and segment again.",
        )

    with Timer(log, "btrack"):
        with btrack.BayesianTracker() as tracker:
            tracker.configure(str(btrack_cfg))
            tracker.max_search_radius = tc["search_radius"]
            tracker.append(objects)
            tracker.volume = ((0, masks.shape[2]), (0, masks.shape[1]), (-1e5, 1e5))
            tracker.track()
            if tc.get("optimize", True):
                log.info("optimising (this is what resolves divisions)")
                tracker.optimize()
            tracks = tracker.tracks
    log.info(f"{len(tracks)} tracks")

    parent, children, root, generation = build_lineage(tracks, log)

    # ── relabel ────────────────────────────────────────────────────────────
    method = tc.get("relabel_method", "refs")
    per_frame = None
    if method == "refs":
        per_frame = relabel_by_refs(tracks, id_map, masks, log)
        n_seg = sum(int(len(np.unique(masks[t])) - 1) for t in range(masks.shape[0]))
        assigned = sum(len(v) for v in per_frame.values())
        if n_seg and assigned / n_seg < 0.5:
            log.warning(
                f"refs covered only {assigned / n_seg:.0%} of segmented "
                f"objects — falling back to centroid matching"
            )
            method, per_frame = "centroid", None
    if per_frame is None:
        method = "centroid"
        per_frame = relabel_by_centroid(
            tracks, masks, tc.get("centroid_tolerance_px", 8), log
        )
    return per_frame, parent, children, root, generation, method


def track_by_overlap(
    masks,
    min_overlap,
    log,
    min_daughter_frac=0.30,
    max_division_growth=1.4,
    min_persist=4,
    merge_frac=0.5,
):
    """
    Link cells by how much their masks overlap between consecutive frames.

    Bacteria on agar barely move from one frame to the next, but they grow
    and they divide. A tracker built on centroid distance has to cope with a
    centroid that shifts as the cell elongates, and with two daughters
    appearing where one mother was — in a packed microcolony the nearest
    object is often a neighbour rather than the same cell.

    Overlap sidesteps all of that: the same cell one frame later occupies
    almost the same pixels, and a neighbour occupies none of them. A mother
    claimed by two objects in the next frame is a division, which is exactly
    the signal wanted, rather than something to be inferred from a motion
    model.

    Returns per_frame[frame][segmentation label] = track id, and the parent
    of every track.
    """
    from scipy.optimize import linear_sum_assignment

    T = masks.shape[0]
    per_frame = defaultdict(dict)
    parent = {}

    first = next((t for t in range(T) if masks[t].any()), None)
    if first is None:
        die(
            "the segmentation is empty in every frame.",
            "Check 02_segment/qc_segmentation.png.",
        )

    next_id = 1
    for lbl in np.unique(masks[first]):
        if lbl:
            per_frame[first][int(lbl)] = next_id
            parent[next_id] = None
            next_id += 1
    log.info(f"overlap tracking: {next_id - 1} cells in the first frame")

    n_div = n_new = n_end = n_frag = 0
    prev_areas = {}
    merges = []
    for t in range(first + 1, T):
        prev, cur = masks[t - 1], masks[t]
        if not cur.any():
            continue

        # how many pixels each (previous label, current label) pair shares
        both = (prev > 0) & (cur > 0)
        pairs = defaultdict(int)
        if both.any():
            a = prev[both].astype(np.int64)
            b = cur[both].astype(np.int64)
            key = a * (int(cur.max()) + 1) + b
            uniq, counts = np.unique(key, return_counts=True)
            for k, c in zip(uniq, counts):
                pairs[
                    (int(k // (int(cur.max()) + 1)), int(k % (int(cur.max()) + 1)))
                ] = int(c)

        cur_area = {
            int(l): int(n) for l, n in zip(*np.unique(cur[cur > 0], return_counts=True))
        }
        prev_areas = {
            int(l): int(n)
            for l, n in zip(*np.unique(prev[prev > 0], return_counts=True))
        }

        # ONE-TO-ONE assignment between last frame's cells and this one's.
        #
        # Letting each object independently pick the previous cell it overlaps
        # most lets a grown cell win its neighbour's object too: the neighbour
        # then matches nothing, its track ends, and one id covers two cells.
        # Solving it as an assignment keeps every previous cell with its own
        # best available partner instead.
        prev_labels = sorted(prev_areas)
        cur_labels = sorted(cur_area)
        assigned = {}  # current label -> previous label
        if prev_labels and cur_labels and pairs:
            cost = np.zeros((len(prev_labels), len(cur_labels)))
            pi = {l: i for i, l in enumerate(prev_labels)}
            ci = {l: i for i, l in enumerate(cur_labels)}
            for (pl, cl), n in pairs.items():
                if pl in pi and cl in ci:
                    # symmetric overlap, so neither a shrinking nor a growing
                    # cell is favoured
                    denom = min(prev_areas.get(pl, 1), cur_area.get(cl, 1))
                    cost[pi[pl], ci[cl]] = -n / max(denom, 1)
            rows_i, cols_i = linear_sum_assignment(cost)
            for r, c in zip(rows_i, cols_i):
                if -cost[r, c] >= min_overlap:
                    assigned[cur_labels[c]] = prev_labels[r]

        # anything left over may be a daughter, a fragment, or a new cell
        leftovers = [cl for cl in cur_labels if cl not in assigned]
        extra = defaultdict(list)
        for cl in leftovers:
            best_pl, best_frac = None, 0.0
            for pl in prev_labels:
                n = pairs.get((pl, cl), 0)
                if not n:
                    continue
                frac = n / max(cur_area.get(cl, 1), 1)
                if frac > best_frac:
                    best_pl, best_frac = pl, frac
            if best_pl is not None and best_frac >= min_overlap:
                extra[best_pl].append(cl)
            else:
                per_frame[t][cl] = next_id
                parent[next_id] = None
                next_id += 1
                n_new += 1

        for cl, pl in assigned.items():
            mother = per_frame[t - 1].get(pl)
            if mother is None:
                per_frame[t][cl] = next_id
                parent[next_id] = None
                next_id += 1
            else:
                per_frame[t][cl] = mother

        for pl, kids in extra.items():
            mother = per_frame[t - 1].get(pl)
            partner = next((c for c, q in assigned.items() if q == pl), None)
            if mother is None:
                for cl in kids:
                    per_frame[t][cl] = next_id
                    parent[next_id] = None
                    next_id += 1
                continue

            areas = sorted(
                [cur_area[c] for c in kids] + ([cur_area[partner]] if partner else []),
                reverse=True,
            )
            prev_a = prev_areas.get(pl, sum(areas))
            smallest_share = areas[-1] / max(sum(areas), 1)
            grew = sum(areas) / max(prev_a, 1)
            real = smallest_share >= min_daughter_frac and grew <= max_division_growth

            if real and partner is not None:
                # a division: mother ends, both pieces become new tracks
                n_div += 1
                for cl in [partner] + kids:
                    per_frame[t][cl] = next_id
                    parent[next_id] = mother
                    next_id += 1
            else:
                # Not a division. These pieces get their OWN ids rather than
                # the mother's: handing one id to several objects in a frame
                # is what made a whole cluster share a colour, and it grows
                # worse each frame as that id covers more ground.
                n_frag += 1
                for cl in kids:
                    per_frame[t][cl] = next_id
                    parent[next_id] = mother if partner is not None else None
                    next_id += 1

        ended = set(per_frame[t - 1].values()) - set(per_frame[t].values())
        n_end += len(ended)

    multi = 0
    for fr, mapping in per_frame.items():
        counts = defaultdict(int)
        for lbl, tid in mapping.items():
            counts[tid] += 1
        multi += sum(1 for v in counts.values() if v > 1)
    if multi:
        log.info(
            f"{multi} case(s) where one track id covers several "
            f"segmented objects in a frame — normally a cell that "
            f"segmentation split, but check the tracking QC: a whole "
            f"cluster sharing one colour means ids are being merged"
        )

    if merges:
        by_frame = defaultdict(int)
        for m in merges:
            by_frame[m["frame"]] += 1
        worst = sorted(by_frame.items(), key=lambda kv: -kv[1])[:3]
        log.warning(
            f"{len(merges)} frame(s) where one mask covers two or more "
            f"previously separate cells — segmentation merged them. Those "
            f"masks are not one cell, so their length and phenotype are the "
            f"sum of several. Worst frames: "
            + ", ".join(f"{f} ({n} merges)" for f, n in worst)
        )
        log.warning(
            "  listed in 03_track/merged_cells.csv. Exclude those "
            "cell-frames from per-cell measurements, or improve "
            "segmentation there — blurred frames are the usual cause."
        )

    if min_persist > 1:
        per_frame, parent = undo_brief_divisions(per_frame, parent, min_persist, log)
        n_div = sum(1 for v in parent.values() if v is not None) // 2

    log.info(
        f"overlap tracking: {len(parent)} tracks, {n_div} divisions, "
        f"{n_new} cells appearing mid-movie, {n_end} ending"
    )
    if n_frag:
        log.info(
            f"overlap tracking: {n_frag} split(s) rejected as segmentation "
            f"fragments rather than divisions, so those tracks continued"
        )
    n_cells0 = len(per_frame[first])
    if n_cells0 and n_div > 6 * n_cells0:
        log.warning(
            f"{n_div} divisions from {n_cells0} starting cells over {T} "
            f"frames. Bacteria manage about one division per 20-30 min, so "
            f"this is likely segmentation splitting cells rather than real "
            f"divisions — raise TRACK_MIN_DAUGHTER_FRAC, or look at "
            f"02_segment/qc_segmentation.png."
        )
    # Where a track begins says a lot about whether it is real. A cell
    # entering from outside the field starts at the border; one that appears
    # in open space is nearly always a broken link.
    Y, X = masks.shape[1], masks.shape[2]
    margin = max(8, int(0.03 * min(Y, X)))
    start_frame, start_border = {}, {}
    for t in sorted(per_frame):
        for lbl, tid in per_frame[t].items():
            if tid in start_frame:
                continue
            start_frame[tid] = t
            ys, xs = np.nonzero(masks[t] == lbl)
            start_border[tid] = bool(
                ys.size
                and (
                    ys.min() < margin
                    or xs.min() < margin
                    or ys.max() > Y - margin
                    or xs.max() > X - margin
                )
            )
    late = [
        tid for tid, f in start_frame.items() if f > first and parent.get(tid) is None
    ]
    if late:
        at_edge = sum(1 for tid in late if start_border.get(tid))
        log.info(
            f"{len(late)} track(s) start after the first frame with no "
            f"parent: {at_edge} at the image border (cells moving into "
            f"view), {len(late) - at_edge} in open field"
        )
        if len(late) - at_edge > 2 * max(len(per_frame[first]), 1):
            log.warning(
                f"{len(late) - at_edge} tracks begin in open field, away from "
                f"the border. A cell cannot appear there — those are links "
                f"the tracker failed to make. Lower TRACK_MIN_OVERLAP_LINK."
            )

    n_founder = sum(1 for v in parent.values() if v is None)
    n_start = len(set(per_frame[first].values()))
    if n_founder > 3 * max(n_start, 1):
        log.warning(
            f"{n_founder} tracks have no parent but only {n_start} cells are "
            f"present in the first frame. A cell appearing from nowhere "
            f"mid-movie is nearly always a broken link — check "
            f"02_segment/qc_segmentation.png, and lower "
            f"TRACK_MIN_OVERLAP_LINK."
        )
    if n_new > 3 * len(per_frame[first]):
        log.warning(
            f"{n_new} cells appear from nowhere after the first "
            f"frame. That usually means links are being missed — try "
            f"lowering TRACK_MIN_OVERLAP_LINK."
        )
    return per_frame, parent, merges


def undo_brief_divisions(per_frame, parent, min_persist, log):
    """
    Undo divisions whose daughters do not stay divided.

    A cell split into two equal halves by a momentary segmentation error looks
    exactly like a division: same total area, two similar pieces. No
    single-frame geometry can separate the two cases.

    What does separate them is time. A real division is permanent; a
    segmentation fragment merges back within a frame or two. So a division is
    only kept if both daughters survive min_persist frames, and otherwise the
    mother's id is put back on both pieces and the track continues.
    """
    from collections import defaultdict

    span = defaultdict(list)
    for fr, mapping in per_frame.items():
        for lbl, tid in mapping.items():
            span[tid].append(int(fr))

    kids = defaultdict(list)
    for tid, mother in parent.items():
        if mother is not None:
            kids[mother].append(tid)

    undone = 0
    # deepest first, so undoing a division cannot orphan one below it
    for mother in sorted(kids, key=lambda m: -min(span[m] or [0])):
        ch = [c for c in kids[mother] if span[c]]
        if len(ch) < 2:
            continue
        if min(len(span[c]) for c in ch) >= min_persist:
            continue  # it stuck: a real division
        for c in ch:
            for fr, mapping in per_frame.items():
                for lbl, tid in list(mapping.items()):
                    if tid == c:
                        mapping[lbl] = mother
            parent.pop(c, None)
            span[mother].extend(span.pop(c, []))
        undone += 1

    if undone:
        log.info(
            f"overlap tracking: {undone} division(s) undone because the "
            f"daughters did not stay apart for {min_persist} frames — "
            f"those were segmentation splitting a cell, not divisions"
        )
    return per_frame, parent


# ── lineage ─────────────────────────────────────────────────────────────────
def lineage_from_parents(parent, ids, log, label="lineage", continuation=None):
    """
    children / root / generation, given who each track's mother is.

    Tracks listed in `continuation` are the same cell carrying on after a
    distrusted link was cut, so they inherit their original's generation
    rather than counting as a new one. Otherwise a track cut twenty times
    reports twenty generations and the division count is meaningless.
    """
    continuation = continuation or {}
    children = defaultdict(list)
    for tid, p in parent.items():
        if p is not None and p in ids and tid not in continuation:
            children[p].append(tid)

    root, generation = {}, {}
    queue = deque()
    for tid in ids:
        if parent.get(tid) is None or parent.get(tid) not in ids:
            root[tid], generation[tid] = tid, 0
            queue.append(tid)
    seen = set(queue)
    while queue:
        cur = queue.popleft()
        for ch in children.get(cur, []):
            if ch in seen:  # cycle guard
                continue
            root[ch] = root[cur]
            generation[ch] = generation[cur] + 1
            seen.add(ch)
            queue.append(ch)
    for tid, orig in continuation.items():  # cuts inherit, not advance
        if tid in ids:
            root[tid] = root.get(orig, orig)
            generation[tid] = generation.get(orig, 0)
    for tid in ids:  # anything left unreached
        root.setdefault(tid, tid)
        generation.setdefault(tid, 0)

    n_div = sum(1 for v in children.values() if len(v) >= 2)
    log.info(
        f"{label}: {len(ids)} tracks, {n_div} division events, "
        f"{len(set(root.values()))} founder lineages"
    )
    return parent, children, root, generation


def build_lineage(tracks, log):
    """parent / root / generation per track, derived from btrack's graph."""
    ids = {int(tr.ID) for tr in tracks}
    parent = {}
    for tr in tracks:
        tid = int(tr.ID)
        p = getattr(tr, "parent", None)
        p = int(p) if p is not None else tid
        parent[tid] = p if (p in ids and p != tid) else None
    return lineage_from_parents(parent, ids, log)


def rebuild_lineage(parent, span, log, continuation=None):
    """Same, after tracks have been cut — the id set has changed."""
    return lineage_from_parents(
        parent, set(span), log, label="lineage after cuts", continuation=continuation
    )


# ── QC ──────────────────────────────────────────────────────────────────────
def link_qc_figure(links, tc, path, log):
    """
    What every frame-to-frame link actually looks like, with the limits on top.

    This is how to tell an over-eager rule from a real problem: if the limit
    sits inside the bulk of the distribution it is cutting normal cells, and
    if it sits in an empty gap it is only catching the outliers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not links:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, key, thr, label, side in [
        (
            axes[0],
            "overlap",
            tc["min_overlap"],
            "overlap with the previous frame",
            "min",
        ),
        (axes[1], "step_um", tc["max_step_um"], "centroid movement (um/frame)", "max"),
        (
            axes[2],
            "area_ratio",
            tc["max_area_ratio"],
            "area change between frames",
            "max",
        ),
    ]:
        v = np.array([l[key] for l in links])
        ax.hist(v, bins=60, color="steelblue", edgecolor="white")
        ax.axvline(
            thr,
            color="red",
            ls="--",
            label=f"{'>=' if side == 'min' else '<='} {thr:g}  "
            f"({100 * ((v < thr) if side == 'min' else (v > thr)).mean():.1f}% cut)",
        )
        ax.set_yscale("log")
        ax.set_xlabel(label, fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("links (log scale)")
    fig.suptitle(
        "Frame-to-frame links — a limit inside the bulk is cutting "
        "normal cells, not catching mistakes",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info(f"link QC -> {path.name}")


def qc_figure(tracked, masks, lineage_rows, qc_frames, path, log):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_id = int(tracked.max())
    rng = np.random.default_rng(0)
    colours = rng.random((max_id + 2, 3)) * 0.75 + 0.25
    colours[0] = 0.0

    n = len(qc_frames)
    fig = plt.figure(figsize=(5 * n, 11))
    gs = fig.add_gridspec(2, n, height_ratios=[2.2, 1])

    for j, t in enumerate(qc_frames):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(colours[tracked[t]], interpolation="nearest")
        ax.set_title(f"frame {t} — {len(np.unique(tracked[t])) - 1} tracked")
        ax.axis("off")

    lengths = np.array([r["n_frames"] for r in lineage_rows])
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(lengths, bins=40, color="steelblue", edgecolor="white")
    ax.set(
        xlabel="track length (frames)",
        ylabel="tracks",
        title=f"median {np.median(lengths):.0f}, "
        f"{int((lengths < 3).sum())} shorter than 3",
    )

    ax = fig.add_subplot(gs[1, 1] if n > 1 else gs[1, 0])
    seg = [int((masks[t] > 0).sum()) for t in range(masks.shape[0])]
    trk = [int((tracked[t] > 0).sum()) for t in range(tracked.shape[0])]
    ax.plot(seg, label="segmented px", color="grey")
    ax.plot(trk, label="tracked px", color="crimson")
    ax.set(xlabel="frame", ylabel="pixels", title="tracking coverage")
    ax.legend(fontsize=8)

    if n > 2:
        ax = fig.add_subplot(gs[1, 2])
        gens = np.array([r["generation"] for r in lineage_rows])
        ax.hist(
            gens,
            bins=np.arange(gens.max() + 2) - 0.5,
            color="seagreen",
            edgecolor="white",
        )
        ax.set(
            xlabel="generation",
            ylabel="tracks",
            title="0 = founder, 1 = first daughters",
        )

    fig.suptitle(
        "Tracking QC — colour is per track id; daughters take new "
        "colours at each division",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"QC figure -> {path}")


# ── main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = base_parser(__doc__.split("\n")[1]).parse_args(argv)
    cfg = load_config()
    log = get_logger("03_track", args.quiet)

    tc = cfg["track"]
    require_settings(
        tc,
        (
            "method",
            "min_overlap_link",
            "min_daughter_frac",
            "max_division_growth",
            "division_min_persist",
            "merge_frac",
            "split_suspect",
            "min_overlap",
            "max_step_um",
            "max_area_ratio",
            "max_gap_frames",
        ),
        "tracking",
    )
    params = dict(tc, position=args.position)
    out = step_dir(cfg, args.position, "03_track", create=True)

    seg_dir = step_dir(cfg, args.position, "02_segment")
    mask_file = require(
        seg_dir / "masks.npz",
        "the masks written by the segmentation step",
        "Set SEGMENT_CELLS=TRUE in config.sh and run again.",
    )
    meta_file = seg_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file) as fh:
            if json.load(fh).get("results", {}).get("preview"):
                die(
                    "the segmentation on disk is a PREVIEW run, covering only a "
                    "few frames, so it cannot be tracked.",
                    'Set PREVIEW_FRAMES="" and SEGMENT_CELLS=TRUE in config.sh, '
                    "then run again.",
                )

    masks = np.load(mask_file)["masks"]
    T = masks.shape[0]

    px_um, px_dt = 0.065, 1.0
    acq_file = step_dir(cfg, args.position, "01_load") / "acquisition.json"
    if acq_file.exists():
        with open(acq_file) as fh:
            _acq = json.load(fh)
            px_um = float(_acq.get("pixel_size_um") or px_um)
            px_dt = float(_acq.get("frame_interval_min") or px_dt)
    log.info(f"masks {masks.shape}, {int(masks.max())} max label")

    btrack_cfg = Path(tc["config_file"])
    if not btrack_cfg.is_absolute():
        btrack_cfg = Path(cfg["experiment"]["config_dir"]) / btrack_cfg
    if tc["method"] == "btrack":
        require(
            btrack_cfg,
            "the btrack settings file",
            "Copy btrack_config.json next to config.sh, or set "
            "BTRACK_CONFIG to its full path.",
        )

    # ── build the links ────────────────────────────────────────────────────
    if tc["method"] == "overlap":
        with Timer(log, "overlap tracking"):
            per_frame, parent, merges = track_by_overlap(
                masks,
                tc["min_overlap_link"],
                log,
                tc["min_daughter_frac"],
                tc["max_division_growth"],
                tc["division_min_persist"],
                tc["merge_frac"],
            )
        parent, children, root, generation = lineage_from_parents(
            parent, set(parent), log
        )
        method = "overlap"
    else:
        merges = []
        per_frame, parent, children, root, generation, method = track_with_btrack(
            masks, tc, btrack_cfg, log
        )

    # ── cut links that cannot be the same cell ─────────────────────────────
    breaks, segments, links = [], {}, []
    if tc.get("split_suspect", True):
        per_frame, breaks, segments, links = split_suspect_tracks(
            per_frame, masks, px_um, tc, log
        )
        with open(out / "track_links.csv", "w") as fh:
            fh.write("track_id,frame,overlap,step_um,area_ratio,gap\n")
            for l in links:
                fh.write(
                    f"{l['track_id']},{l['frame']},{l['overlap']:.4f},"
                    f"{l['step_um']:.4f},{l['area_ratio']:.3f},"
                    f"{l['gap']}\n"
                )
        link_qc_figure(links, tc, out / "qc_track_links.png", log)

    # Spans come from the final mapping, not from btrack, so newly created
    # ids after a cut appear in lineage.csv like any other track.
    from collections import defaultdict as _dd

    # DISTINCT frames. A track can hold several labels in one frame when
    # segmentation has split a cell, so counting entries made a cell present
    # in 61 frames report 205 — and MIN_TRACK_LENGTH then filtered on a
    # number that was partly a measure of how badly the cell fragmented.
    seen = _dd(set)
    for t, mapping in per_frame.items():
        for _lbl, tid in mapping.items():
            seen[int(tid)].add(int(t))
    span = {tid: (min(fs), max(fs), len(fs)) for tid, fs in seen.items()}

    split_from = {b["new_id"]: b["original_id"] for b in breaks}
    if segments:
        parent = reparent_after_split(
            parent, segments, {tid: v[0] for tid, v in span.items()}, log
        )
        # A track that was cut is a continuation whose link was distrusted,
        # not a cell that appeared from nowhere. Recording it as a founder
        # would inflate the founder count and hide how many real lineages
        # there are.
        for new_id, orig in split_from.items():
            parent.setdefault(new_id, orig)
        for tid in span:
            parent.setdefault(tid, None)
        parent, children, root, generation = rebuild_lineage(
            parent, span, log, continuation=split_from
        )

    min_len = int(tc.get("min_track_length", 0))
    keep_ids = {tid for tid, (_, _, n) in span.items() if n >= min_len}
    dropped_with_kids = [t for t in span if t not in keep_ids and children.get(t)]
    if dropped_with_kids:
        log.warning(
            f"{len(dropped_with_kids)} dropped track(s) have daughters; "
            f"their lineages are broken in the masks but preserved in "
            f"lineage.csv"
        )

    rows = []
    for tid, (f0, f1, n) in sorted(span.items()):
        rows.append(
            {
                "track_id": tid,
                "parent_id": parent.get(tid) or "",
                "root_id": root.get(tid, tid),
                "generation": generation.get(tid, 0),
                "n_children": len(children.get(tid, [])),
                "divides": int(len(children.get(tid, [])) >= 2),
                "start_frame": f0,
                "end_frame": f1,
                "n_frames": n,
                "kept": int(tid in keep_ids),
                "split_from": split_from.get(tid, ""),
            }
        )

    # Readable names: a founder keeps its id, its daughters become id-1 and
    # id-2, and so on down the generations. Everything from one founder shares
    # a colour, which is what the overlays and the overview PDF use.
    names, colours = build_cell_names(rows, continuation=split_from)
    for r in rows:
        r["cell_name"] = names.get(r["track_id"], str(r["track_id"]))
        r["lineage_colour"] = colours.get(r["track_id"], "")
    log.info(
        f"named {len(names)} tracks across "
        f"{len(set(colours.values()))} colour group(s); deepest name: "
        f"{max(names.values(), key=len)}"
    )
    with open(out / "lineage.csv", "w") as fh:
        cols = list(rows[0].keys())
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")

    with open(out / "divisions.csv", "w") as fh:
        fh.write("parent_id,division_frame,daughter_ids,n_daughters\n")
        n_div = 0
        for pid, kids in sorted(children.items()):
            if len(kids) < 2:
                continue
            n_div += 1
            dframe = min(span[k][0] for k in kids if k in span)
            fh.write(
                f"{pid},{dframe},"
                f"\"{';'.join(str(k) for k in sorted(kids))}\",{len(kids)}\n"
            )

    if breaks:
        per_track = defaultdict(int)
        for b in breaks:
            per_track[b["original_id"]] += 1
        worst = sorted(per_track.items(), key=lambda kv: -kv[1])[:5]
        log.info(
            "most-cut track(s): " + ", ".join(f"id {k} cut {v}x" for k, v in worst)
        )
        if worst and worst[0][1] > 5:
            log.warning(
                f"track {worst[0][0]} was cut {worst[0][1]} times. A cell is "
                f"not repeatedly changing identity — either its shape changes "
                f"faster than the checks allow, or segmentation is unstable "
                f"there. Loosen TRACK_MIN_OVERLAP, TRACK_MAX_STEP_UM and "
                f"TRACK_MAX_AREA_RATIO, or set TRACK_SPLIT_SUSPECT=FALSE and "
                f"rely on the overlap tracker alone."
            )

    with open(out / "merged_cells.csv", "w") as fh:
        fh.write("frame,time_min,segmentation_label,track_ids,n_merged," "area_px\n")
        for m in merges:
            ids = ";".join(str(i) for i in m["track_ids"])
            fh.write(
                f"{m['frame']},{m['frame'] * px_dt:.2f},{m['label']},"
                f"\"{ids}\",{m['n_merged']},{m['area_px']}\n"
            )

    with open(out / "track_breaks.csv", "w") as fh:
        fh.write("original_id,new_id,frame,reason,overlap,step_um," "area_ratio,gap\n")
        for b in breaks:
            fh.write(
                f"{b['original_id']},{b['new_id']},{b['frame']},"
                f"\"{b['reason']}\",{b['overlap']},{b['step_um']},"
                f"{b['area_ratio']},{b['gap']}\n"
            )

    tracked = apply_mapping(masks, per_frame, keep_ids)
    coverage = float((tracked > 0).sum()) / max(int((masks > 0).sum()), 1)
    log.info(
        f"relabel method={method}  coverage={coverage:.1%}  "
        f"kept {len(keep_ids)}/{len(span)} tracks "
        f"(min_track_length={min_len})"
    )
    if coverage < 0.8:
        log.warning(
            "coverage below 80% — raise track.search_radius, or lower "
            "track.min_track_length if many tracks are fragmented"
        )

    np.savez_compressed(out / "tracked_masks.npz", masks=tracked)

    nqc = int(tc.get("qc_frames", 4))
    qc_frames = sorted({int(round(x)) for x in np.linspace(0, T - 1, max(2, nqc))})
    qc_figure(tracked, masks, rows, qc_frames, out / "qc_tracking.png", log)

    write_meta(
        out,
        "03_track",
        params,
        {"masks": mask_file, "btrack_config": btrack_cfg},
        {
            "n_tracks": len(span),
            "n_kept": len(keep_ids),
            "n_suspect_links_cut": len(breaks),
            "n_divisions": n_div,
            "relabel_method": method,
            "coverage": round(coverage, 4),
            "median_track_length": float(np.median([r["n_frames"] for r in rows])),
            "n_founder_lineages": len(set(root.values())),
        },
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "track"))

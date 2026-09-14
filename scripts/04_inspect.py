#!/usr/bin/env python3
"""
Step 04 — make the tracked cells inspectable in Fiji.

Outputs (in <pos>/04_inspect/):
    cell_labels.csv     frame, track_id, x, y   (one row per cell per frame)
    outlines.tif        cell boundaries, 8-bit, same size as the movie
    open_in_fiji.ijm    drag onto Fiji and press Run
    find_cell.ijm       asks for a track id and jumps to that cell
    meta.json

The macro opens the channels as one composite hyperstack, draws each cell's
track id on top of it, and applies the pixel size and frame interval read from
the nd2 — so a cell of interest can be identified by eye and its id read off
directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    Timer,
    base_parser,
    die,
    env_bool,
    get_logger,
    hex_to_rgb,
    load_cell_names,
    load_config,
    read_units,
    require,
    resolve_timepoints,
    run_safely,
    step_dir,
    write_meta,
)


# ── the Fiji macro ──────────────────────────────────────────────────────────
def build_macro(
    paths, px_um, dt_min, n_frames, label_every, font_size, show_outlines
) -> str:
    """Generate a self-contained .ijm with absolute paths already filled in."""

    def ij(p):  # forward slashes for ImageJ
        return str(p).replace("\\", "/")

    open_outlines = ""
    merge_outline = ""
    n_channels = 3
    if show_outlines:
        open_outlines = f'open("{ij(paths["outlines"])}");\nrename("outlines");\n'
        merge_outline = " c5=outlines"
        n_channels = 4

    return f"""// =====================================================================
//  Open this movie with track ids drawn on the cells.
//  Generated automatically — drag onto Fiji and press Run.
//
//  Settings you may want to change:
//      SATURATED      : % of pixels allowed to clip when setting contrast.
//                       Higher = punchier, 0.35 is the Fiji default.
//      CONTRAST_FRAME : frame used to judge contrast. 0 = the middle frame,
//                       which is usually representative: the first frame has
//                       no PI signal yet and the last ones are full of it.
//      LABEL_EVERY    : draw ids on every Nth frame (1 = every frame).
//      FONT_SIZE      : size of the id text.
//      LABEL_COLOR    : yellow, white, cyan, magenta, red, green
//      INVERT_BF      : show brightfield dark-on-light.
//
//  This is for looking at only. The channels are scaled to 8-bit here, so
//  never measure intensities off this window — the measurement steps work
//  from the original 16-bit files.
// =====================================================================

SATURATED      = 0.35;
CONTRAST_FRAME = 0;
LABEL_EVERY       = {label_every};
FONT_SIZE         = {font_size};
COLOR_BY_LINEAGE  = true;      // false = every label in LABEL_COLOR
LABEL_COLOR       = "yellow";
INVERT_BF         = false;

run("Close All");
print("\\\\Clear");
setBatchMode(true);

// ---- open the channels ----------------------------------------------
open("{ij(paths["bf"])}");   rename("bf");
open("{ij(paths["gfp"])}");  rename("gfp");
open("{ij(paths["rfp"])}");  rename("rfp");
{open_outlines}
// ---- set contrast, then convert to 8-bit -----------------------------
// Merge Channels needs every input to have the same bit depth, and the
// outlines are 8-bit, so the three camera channels are brought down to
// 8-bit too. The contrast is judged on ONE frame and then held fixed for
// the whole stack, so brightness does not drift as the movie plays.
prepare("bf");
prepare("gfp");
prepare("rfp");

// ---- safety check -----------------------------------------------------
// If this ever fails, the merge below would stop with
// "The source images must have the same bit depth".
depth = 0;
titles = getList("image.titles");
for (i = 0; i < titles.length; i++) {{
    selectWindow(titles[i]);
    if (depth == 0) depth = bitDepth();
    if (bitDepth() != depth) {{
        setBatchMode(false);
        exit("Channel '" + titles[i] + "' is " + bitDepth() + "-bit but "
             + "another channel is " + depth + "-bit.\\n"
             + "All channels must match before merging.");
    }}
}}

// ---- combine into one composite --------------------------------------
run("Merge Channels...", "c1=rfp c2=gfp c4=bf{merge_outline} create");
rename("tracked");

// ---- physical units ---------------------------------------------------
run("Properties...", "channels={n_channels} slices=1 frames={n_frames}"
    + " pixel_width={px_um} pixel_height={px_um} voxel_depth=1"
    + " frame=[{dt_min} min]");
Stack.setDisplayMode("composite");

if (INVERT_BF) {{
    Stack.setChannel(3);                        // c4=bf is the third channel
    run("Invert LUT");
}}

// ---- scale bar and clock ---------------------------------------------
// Added before the ids: these commands rebuild the overlay, so drawing the
// ids afterwards keeps them.
run("Scale Bar...", "width=5 height=4 font=14 color=White background=None "
    + "location=[Lower Right] hide overlay label");
run("Label...", "format=0 starting=0 interval={dt_min} x=5 y=20 font=14 "
    + "text=min range=1-{n_frames} use overlay");

// ---- draw the cell names ---------------------------------------------
// Each line of cell_labels.csv is:
//     frame,track_id,cell_name,x,y,r,g,b
// cell_name shows the descent: 12 is a founder, 12-1 and 12-2 are its
// daughters, 12-1-1 a granddaughter. r,g,b is the colour of that lineage.
csv   = File.openAsString("{ij(paths["labels"])}");
lines = split(csv, "\\n");
setFont("SansSerif", FONT_SIZE, "bold antialiased");
if (!COLOR_BY_LINEAGE) setColor(LABEL_COLOR);

n = 0;
for (i = 1; i < lines.length; i++) {{          // row 0 is the header
    line = String.trim(lines[i]);
    if (lengthOf(line) == 0) continue;
    f = split(line, ",");
    if (f.length < 8) continue;

    frame = parseInt(f[0]) + 1;                // ImageJ counts from 1
    name  = f[2];
    x     = parseFloat(f[3]);
    y     = parseFloat(f[4]);

    if ((frame - 1) % LABEL_EVERY != 0) continue;

    if (COLOR_BY_LINEAGE)
        setColor(parseInt(f[5]), parseInt(f[6]), parseInt(f[7]));

    Overlay.setPosition(0, 0, frame);          // 0,0 = all channels
    Overlay.drawString(name, x, y);
    Overlay.add;
    Overlay.setPosition(0, 0, frame);          // harmless if already set
    n++;
}}
Overlay.show;

setBatchMode(false);
Stack.setFrame(1);
print("Ready. " + n + " labels drawn.");
print("Read a cell's id straight off the image, then put it in CELL_TRACK_IDS "
    + "in config.sh to extract that cell.");


// ---- helper used above ------------------------------------------------
// Sets the contrast from one representative frame, holds it across the whole
// stack, then converts to 8-bit so all channels can be merged.
function prepare(title) {{
    selectWindow(title);
    if (bitDepth() == 8) return;                // outlines are already 8-bit

    n = nSlices;
    f = CONTRAST_FRAME;
    if (f < 1 || f > n) f = floor(n / 2) + 1;   // 0 or nonsense -> middle
    setSlice(f);

    run("Enhance Contrast", "saturated=" + SATURATED);
    getMinAndMax(lo, hi);                       // what it chose on that frame
    if (hi <= lo) {{                             // flat frame, use the stack
        resetMinAndMax();
        getMinAndMax(lo, hi);
    }}
    setMinAndMax(lo, hi);                       // pin it for every frame

    setOption("ScaleConversions", true);        // scale, do not just truncate
    run("8-bit");
    print(title + ": contrast " + lo + "-" + hi + " taken from frame " + f
          + " of " + n);
}}
"""


def build_find_macro(labels_path, n_frames) -> str:
    def ij(p):
        return str(p).replace("\\", "/")

    return f"""// =====================================================================
//  Jump to a cell by its track id.
//  Run open_in_fiji.ijm first, then run this one.
// =====================================================================

id = getString("Track id or cell name to find (e.g. 12 or 12-1)", "1");

csv   = File.openAsString("{ij(labels_path)}");
lines = split(csv, "\\n");
first = -1; lastf = -1; fx = 0; fy = 0;

for (i = 1; i < lines.length; i++) {{
    f = split(String.trim(lines[i]), ",");
    if (f.length < 8) continue;
    if (f[1] == id || f[2] == id) {{
        fr = parseInt(f[0]);
        if (first < 0) {{ first = fr; fx = parseFloat(f[3]); fy = parseFloat(f[4]); }}
        lastf = fr;
    }}
}}

if (first < 0) {{
    showMessage("Not found",
        "No cell with track id " + id + ".\\n\\n" +
        "Ids are drawn on the image by open_in_fiji.ijm.");
}} else {{
    Stack.setFrame(first + 1);
    makeOval(fx - 25, fy - 25, 50, 50);
    print("Track " + id + ": frames " + first + " to " + lastf +
          " (" + (lastf - first + 1) + " frames), now showing its first frame.");
}}
"""


def overview_pdf(
    load_dir,
    tracked,
    centroids,
    name_of,
    colour_of,
    requested_min,
    dt_min,
    px_um,
    path,
    log,
):
    """
    One page per timepoint: the whole field of view, every cell outlined and
    named in its lineage colour. Vector text, so it stays sharp when zoomed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import tifffile
    from matplotlib.backends.backend_pdf import PdfPages
    from skimage.segmentation import find_boundaries

    T = tracked.shape[0]
    frames, skipped = [], []
    for minute in requested_min:
        f = int(round(minute / dt_min))
        (frames if 0 <= f < T else skipped).append((minute, f))
    if skipped:
        log.info(
            "overview: "
            + ", ".join(f"{m:g}" for m, _ in skipped)
            + " min are past the end of this movie, skipped"
        )
    if not frames:
        log.warning("overview: none of the requested timepoints exist here")
        return

    bf = tifffile.imread(load_dir / "bf.tif")
    lo, hi = np.percentile(bf[[f for _, f in frames]], [1, 99.5])

    with PdfPages(path) as pdf:
        for minute, f in frames:
            fig, ax = plt.subplots(figsize=(11, 11))
            ax.imshow(bf[f], cmap="gray", vmin=lo, vmax=hi)

            lbl = tracked[f]
            if lbl.any():
                edges = find_boundaries(lbl, mode="inner")
                rgba = np.zeros(lbl.shape + (4,))
                for tid in np.unique(lbl[edges]):
                    if not tid:
                        continue
                    sel = edges & (lbl == tid)
                    c = matplotlib.colors.to_rgb(colour_of(tid))
                    rgba[sel] = (*c, 1.0)
                ax.imshow(rgba, interpolation="nearest")

            for tid, cx, cy in centroids.get(f, []):
                ax.text(
                    cx,
                    cy,
                    name_of(tid),
                    fontsize=8,
                    ha="center",
                    va="center",
                    color=colour_of(tid),
                    weight="bold",
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
                )

            n = len(centroids.get(f, []))
            ax.set_title(
                f"{minute:g} min   ·   frame {f}   ·   {n} cells named"
                "\nnames show descent: a founder keeps its number, "
                "its daughters add -1 and -2",
                fontsize=11,
            )
            ax.set_xlim(0, tracked.shape[2])
            ax.set_ylim(tracked.shape[1], 0)
            ax.axis("off")

            # Scale bar, placed in axes coordinates so it cannot fall outside
            # the image. Length is the largest round number under a quarter
            # of the field width.
            width_um = tracked.shape[2] * px_um
            bar_um = next((v for v in (20, 10, 5, 2, 1) if v <= 0.25 * width_um), 1)
            frac = bar_um / width_um
            # transAxes: y=0 is the bottom of the panel, whatever the data
            # limits do, so this stays in the lower right corner.
            ax.plot(
                [0.96 - frac, 0.96],
                [0.045, 0.045],
                transform=ax.transAxes,
                color="white",
                lw=3,
                solid_capstyle="butt",
            )
            ax.text(
                0.96 - frac / 2,
                0.055,
                f"{bar_um:g} um",
                transform=ax.transAxes,
                color="white",
                ha="center",
                va="bottom",
                fontsize=9,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log.info(f"overview PDF -> {Path(path).name} ({len(frames)} pages)")


# ── main ────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = base_parser(__doc__.split("\n")[1])
    args = p.parse_args(argv)
    cfg = load_config()
    log = get_logger("04_inspect", args.quiet)

    ic = cfg["inspect"]
    pos = args.position

    load_dir = step_dir(cfg, pos, "01_load")
    track_dir = step_dir(cfg, pos, "03_track")
    tif = {
        c: require(
            load_dir / f"{c}.tif",
            f"the {c} channel",
            "Set LOAD_ND2=TRUE in config.sh and run again.",
        )
        for c in ("bf", "gfp", "rfp")
    }
    mask_file = require(
        track_dir / "tracked_masks.npz",
        "the tracked masks",
        "Set TRACK_CELLS=TRUE in config.sh and run again.",
    )

    out = step_dir(cfg, pos, "04_inspect", create=True)

    import tifffile
    from skimage import measure
    from skimage.segmentation import find_boundaries

    tracked = np.load(mask_file)["masks"]
    T = tracked.shape[0]

    # acquisition units, written by step 01
    px_um, dt_min = read_units(load_dir, log, cfg)

    # how long each track lasts, used to hide clutter from very short ones
    lifespan = {}
    for t in range(T):
        for tid in np.unique(tracked[t]):
            if tid:
                lifespan[int(tid)] = lifespan.get(int(tid), 0) + 1

    min_len = int(ic["label_min_track_length"])
    shown = {tid for tid, n in lifespan.items() if n >= min_len}
    log.info(
        f"{len(shown)} of {len(lifespan)} tracks are at least "
        f"{min_len} frames long and will be labelled"
    )

    lineage_csv = track_dir / "lineage.csv"
    if lineage_csv.exists():
        names, colours = load_cell_names(lineage_csv)
        log.info(f"{len(set(colours.values()))} lineage colour group(s)")
    else:
        names, colours = {}, {}
        log.warning(
            "lineage.csv not found — labels will be plain track ids "
            "with no lineage colouring"
        )

    def name_of(tid):
        return names.get(int(tid), str(int(tid)))

    def colour_of(tid):
        return colours.get(int(tid), "#ffe119")

    # ── label positions ────────────────────────────────────────────────────
    centroids = {}  # frame -> [(tid, x, y)] for the PDF
    with Timer(log, "label positions"), open(out / "cell_labels.csv", "w") as fh:
        fh.write("frame,track_id,cell_name,x,y,r,g,b\n")
        n_rows = 0
        for t in range(T):
            if not tracked[t].any():
                continue
            here = []
            for reg in measure.regionprops(tracked[t]):
                if reg.label not in shown:
                    continue
                cy, cx = reg.centroid
                r, g, b = hex_to_rgb(colour_of(reg.label))
                fh.write(
                    f"{t},{reg.label},{name_of(reg.label)},"
                    f"{cx:.1f},{cy:.1f},{r},{g},{b}\n"
                )
                here.append((int(reg.label), cx, cy))
                n_rows += 1
            centroids[t] = here
    log.info(f"cell_labels.csv: {n_rows} rows")

    # ── outlines ───────────────────────────────────────────────────────────
    show_outlines = bool(ic["show_outlines"])
    if show_outlines:
        with Timer(log, "outlines"):
            outl = np.zeros(tracked.shape, dtype=np.uint8)
            for t in range(T):
                if tracked[t].any():
                    outl[t] = find_boundaries(tracked[t], mode="inner") * 255
            tifffile.imwrite(out / "outlines.tif", outl)
        log.info("outlines.tif written")

    # ── overview PDF at fixed timepoints ───────────────────────────────────
    if bool(ic["overview_pdf"]):
        want = resolve_timepoints(
            os.environ.get("TIMEPOINTS_MIN", "0 30 45 60 90 120 150"), T, dt_min, log
        )
        overview_pdf(
            load_dir,
            tracked,
            centroids,
            name_of,
            colour_of,
            want,
            dt_min,
            px_um,
            out / "overview_timepoints.pdf",
            log,
        )

    # ── macros ─────────────────────────────────────────────────────────────
    paths = {
        "bf": tif["bf"],
        "gfp": tif["gfp"],
        "rfp": tif["rfp"],
        "outlines": out / "outlines.tif",
        "labels": out / "cell_labels.csv",
    }
    (out / "open_in_fiji.ijm").write_text(
        build_macro(
            paths,
            px_um,
            dt_min,
            T,
            int(ic["label_every_n_frames"]),
            int(ic["label_font_size"]),
            show_outlines,
        )
    )
    (out / "find_cell.ijm").write_text(build_find_macro(out / "cell_labels.csv", T))

    log.info(f"drag this onto Fiji:  {out / 'open_in_fiji.ijm'}")

    write_meta(
        out,
        "04_inspect",
        dict(ic, position=pos),
        {"tracked_masks": mask_file},
        {
            "n_label_rows": n_rows,
            "n_tracks_labelled": len(shown),
            "n_tracks_total": len(lifespan),
            "pixel_size_um": px_um,
            "frame_interval_min": dt_min,
        },
    )
    log.info(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_safely(main, "inspect"))

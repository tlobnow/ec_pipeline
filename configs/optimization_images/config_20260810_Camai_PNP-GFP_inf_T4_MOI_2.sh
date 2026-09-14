#!/usr/bin/env bash
# =============================================================================
#  PIPELINE CONFIGURATION
#
#  This is the only file you edit. Copy it per experiment, e.g.
#      cp config.sh config_uninf.sh
#  and run:
#      ./run_pipeline.sh -c config_uninf.sh
#
#  Rules:
#    * TRUE / FALSE must be written in capitals.
#    * Paths with spaces must stay inside "quotes".
#    * Never put a space around the = sign.   NAME="x"   not   NAME = "x"
# =============================================================================


# ── WHICH STEPS TO RUN ──────────────────────────────────────────────────────
# Set a step to FALSE to skip it. A skipped step must have been run before,
# because the next step reads its output files from disk.

# Steps always overwrite their previous output. To redo only tracking, set
# LOAD_ND2=FALSE, SEGMENT_CELLS=FALSE, TRACK_CELLS=TRUE.
#
# Usual way of working:
#   1. run steps 1-4, open 04_inspect/open_in_fiji.ijm, read the id of the
#      cell you care about off the image


### RUN MAIN, NO EXTRACTION
MAIN=TRUE ; EXTRACT=FALSE
# MAIN=FALSE ; EXTRACT=TRUE
# GFP_ONLY=TRUE

if ([ "$MAIN" = TRUE ]); then
  LOAD_ND2=TRUE            # step 1 - read the nd2, write bf/gfp/rfp tif
  SEGMENT_CELLS=TRUE       # step 2 - Omnipose segmentation
  TRACK_CELLS=TRUE         # step 3 - btrack tracking + lineage
  INSPECT_IN_FIJI=TRUE     # step 4 - Fiji macro with track ids drawn on cells
  EXTRACT_CELL=FALSE       # step 5 - follow one cell as its own small movie
  ANALYSE_GFP=TRUE         # step 6 - find GFP puncta and filaments
fi

if ([ "$EXTRACT" = TRUE ]); then
  LOAD_ND2=FALSE            # step 1 - read the nd2, write bf/gfp/rfp tif
  SEGMENT_CELLS=FALSE       # step 2 - Omnipose segmentation
  TRACK_CELLS=FALSE         # step 3 - btrack tracking + lineage
  INSPECT_IN_FIJI=FALSE     # step 4 - Fiji macro with track ids drawn on cells
  EXTRACT_CELL=TRUE         # step 5 - follow one cell as its own small movie
  ANALYSE_GFP=FALSE         # step 6 - find GFP puncta and filaments
fi

if ([ "$GFP_ONLY" = TRUE ]); then
  LOAD_ND2=FALSE            # step 1 - read the nd2, write bf/gfp/rfp tif
  SEGMENT_CELLS=FALSE       # step 2 - Omnipose segmentation
  TRACK_CELLS=FALSE         # step 3 - btrack tracking + lineage
  INSPECT_IN_FIJI=FALSE     # step 4 - Fiji macro with track ids drawn on cells
  EXTRACT_CELL=FALSE         # step 5 - follow one cell as its own small movie
  ANALYSE_GFP=TRUE         # step 6 - find GFP puncta and filaments
fi

#   2. put that id in CELL_TRACK_IDS below, set EXTRACT_CELL=TRUE and the
#      other four to FALSE, and run again
# EXTRACT ONLY
# LOAD_ND2=FALSE            # step 1 - read the nd2, write bf/gfp/rfp tif
# SEGMENT_CELLS=FALSE       # step 2 - Omnipose segmentation
# TRACK_CELLS=FALSE         # step 3 - btrack tracking + lineage
# INSPECT_IN_FIJI=FALSE     # step 4 - Fiji macro with track ids drawn on cells
# EXTRACT_CELL=TRUE         # step 5 - follow one cell as its own small movie
# ANALYSE_GFP=FALSE         # step 6 - find GFP puncta and filaments


# ── EXPERIMENT ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME="20260810_Camai_PNP-GFP_inf_T4_MOI2005.nd2"
ND2_PATH='/Users/u_lobnow/Desktop/bact_microscopy/20260810/20260810_Camai_PNP-GFP_inf_T4_MOI2005.nd2'
OUTPUT_ROOT="/Users/u_lobnow/Library/CloudStorage/Dropbox-WeizmannInstitute/Finn Lobnow/Wein lab/Projects/BacHuman_DFDs/Experiments/Microscopy/ecoli_pipeline/output"
CONDITION="infected"                # infected | uninfected

# Fields of view. "all" = every position in the nd2, otherwise list them
# separated by spaces, e.g. POSITIONS="0 1 2"
# POSITIONS="0"
POSITIONS="all"

# Conda environment to activate. Leave empty ("") to use whatever python is
# already active.
CONDA_ENV="phage_pipeline"


# ── STEP 1: LOAD ────────────────────────────────────────────────────────────
# Where the images come from:
#   nd2    read the nd2 directly. Fine for small files.
#   split  read per-position TIFFs written by tools/split_nd2.py. Much faster
#          for large files, and the only sensible option for z stacks.
#
# To split once:   python tools/split_nd2.py -c configs/your_config.sh
INPUT_SOURCE="nd2"
SPLIT_DIR=""                        # empty = <nd2 folder>/<nd2 name>_split
 
# Channels. USE THE NAMES — they are printed in the log when the file is
# opened. Numbers work too, but files are not always ordered phase first:
# one recent file is GFP, RFP, PHASE, so assuming brightfield is channel 0
# would analyse the GFP image as if it were phase contrast.
CH_BF="PHASE"
CH_GFP="GFP"
CH_RFP="RFP"                        # PI channel, must RISE over time
 
# --- z stacks ---
# Each frame is reduced to one image per channel.
#   focus  the sharpest plane, chosen from the phase channel per frame and
#          applied to every channel. The default, and what you want for a
#          long timelapse: focus drifts over hours.
#   fixed  one plane for the whole movie, set by Z_PLANE
#   max    brightest value across planes
#   mean   average across planes
#
# Check the focus first on a new dataset — it writes figures and nothing else:
#     python tools/split_nd2.py -c configs/your_config.sh --inspect-focus
Z_METHOD="focus"
Z_PLANE=0                           # only used when Z_METHOD="fixed"
Z_MAX_STEP=1                        # how far the plane may move per frame
Z_FOCUS_SAMPLE_EVERY=5              # measure focus every Nth frame and
                                    # interpolate between; 1 = every frame
 
CROP_HALF_SIZE=0                  # 200 -> 400x400 centre crop; 0 = full frame
 
# Leave these EMPTY to read them from the nd2 file, which is what you want
# almost always. Put a number in only to overrule the file.
#
# The frame interval is read from the timestamps the microscope recorded
# frame by frame, not from the period that was programmed before the run.
# Those two can differ if the microscope could not keep up, and the log says
# which was used and warns when they disagree. To check a file directly:
#     python tools/check_nd2_timing.py /path/to/file.nd2
FRAME_INTERVAL_MIN=""        # e.g. 2.5 to force it
PIXEL_SIZE_UM=""             # e.g. 0.065 to force it
 
# ── STEP 2: SEGMENTATION ────────────────────────────────────────────────────
OMNI_MODEL="bact_phase_affinity"
MASK_THRESHOLD=5                    # merged cells -> lower it
                                    # missed cells -> raise it
FLOW_THRESHOLD=0
SEGMENT_CHANNEL="bf"                # which loaded channel to segment
USE_GPU="auto"                      # auto | TRUE | FALSE
AFFINITY_SEG=TRUE
 
# Threshold tuning: list a few frames to segment only those, in seconds
# instead of minutes. Output goes to 02_segment_preview/ and step 3 refuses
# to use it. Set back to "" for a real run.
PREVIEW_FRAMES=""                   # e.g. "0 30 60"
 
 
# ── STEP 3: TRACKING ────────────────────────────────────────────────────────
BTRACK_CONFIG="btrack_config.json"  # relative paths are resolved next to this
                                    # config file
SEARCH_RADIUS=20                    # raise if tracks are fragmented
OPTIMIZE_TRACKS=TRUE                # needed for division detection
 
# How tracks are matched back to segmented cells.
#   refs     = exact, uses btrack's own object references (recommended)
#   centroid = older nearest-centroid method, kept as a fallback
RELABEL_METHOD="refs"
CENTROID_TOLERANCE_PX=8             # only used by the centroid method
 
# Tracks shorter than this are removed from the mask stack. They stay listed
# in lineage.csv with kept=0, so nothing vanishes without a record.
MIN_TRACK_LENGTH=3
 
QC_FRAMES=4                         # frames shown in the tracking QC figure
 
 
# ── TIMEPOINTS (used by step 4 and step 5) ──────────────────────────────────
# Times in minutes. The overview PDF gets one page per timepoint, and the
# single-cell montages get one column per timepoint. Times past the end of
# the movie are skipped and listed in the log.
TIMEPOINTS_MIN="0 30 45 60 90 120 150"
 
 
# ── STEP 4: LOOK AT THE CELLS IN FIJI ───────────────────────────────────────
# Writes 04_inspect/open_in_fiji.ijm — drag it onto Fiji and press Run.
 
LABEL_EVERY_N_FRAMES=1       # draw ids on every Nth frame; raise if crowded
                             # (can also be changed at the top of the macro)
LABEL_FONT_SIZE=10
LABEL_MIN_TRACK_LENGTH=3     # do not label tracks shorter than this
SHOW_OUTLINES=TRUE           # add cell outlines as an extra channel
OVERVIEW_PDF=TRUE            # 04_inspect/overview_timepoints.pdf — one page
                             # per timepoint, whole field of view, every cell
                             # outlined and named in its lineage colour.
                             # Vector text, so zoom in as far as you like.
 
# Cells are named by descent: a founder keeps its track number, its daughters
# add -1 and -2, their daughters -1-1 and so on. Everything from one founder
# shares a colour, from a set of nine chosen to stay apart. With more than
# nine founders the colours repeat, so use the names to be sure.
 
 
# ── STEP 5: FOLLOW ONE CELL ─────────────────────────────────────────────────
# What to extract:
#   ids      = the cells listed in CELL_TRACK_IDS, one output each
#   lineages = one output per founder, following the whole family tree
CELL_EXTRACT="ids"
 
# Used when CELL_EXTRACT="ids". Either form works: the track number (41) or
# the cell name from the overview PDF (41-2, 41-1-2). Several allowed,
# separated by spaces: CELL_TRACK_IDS="41-2 45 88"
#
# Picking a daughter is the normal case. With CELL_FOLLOW_ANCESTORS=TRUE
# below, asking for 41-2 gives one continuous movie: the mother 41 until it
# divides, then 41-2 onwards.
CELL_TRACK_IDS=""
 
# Used when CELL_EXTRACT="lineages". Empty = every founder whose family lasts
# at least CELL_LINEAGE_MIN_FRAMES frames in total. Founders are the names
# without a dash in them.
CELL_LINEAGE_IDS=""
CELL_LINEAGE_MIN_FRAMES=20
 
# --- how much of the cell's life to show ---
# A track id only lasts from one division to the next, so a cell picked at
# 150 min has no past of its own. With ancestors switched on, the earlier
# frames show whichever mother, grandmother and so on was there at the time,
# and the result is one cell followed from the start of the movie.
CELL_FOLLOW_ANCESTORS=TRUE   # walk BACK through the lineage
CELL_FOLLOW_DAUGHTERS=FALSE  # walk FORWARD past divisions (keeps the whole
                             # family in the box, so it stops being one cell)
 
# Which frames end up in the tif:
#   movie   = every frame of the movie. Frames before the cell appears or
#             after it is gone are blank, so the time axis matches the
#             original and the montage timepoints below always line up.
#   lineage = only the frames where the cell or an ancestor is present.
CELL_SPAN="movie"
 
# --- montage ---
# Columns come from TIMEPOINTS_MIN above. Timepoints where the cell is absent
# are still shown, marked "absent", so a cell that dies is visibly gone
# rather than silently missing.
CELL_MONTAGE_SHOW_MERGE=TRUE # add two rows combining BF, GFP and RFP: one
                             # masked, one with the surroundings kept
 
# --- appearance ---
CELL_PAD_PX=10               # empty margin around the cell, in pixels
CELL_BACKGROUND="black"      # black | white — what replaces the surroundings
                             # (the montage is always drawn on black)
CELL_MASK_OUTSIDE=TRUE       # FALSE keeps the neighbours visible
CELL_SAVE_UNMASKED=TRUE      # also save the same crop without masking,
                             # which is the honest way to check the outline
CELL_ALIGN_MAJOR_AXIS=FALSE  # TRUE turns the cell to lie horizontally.
                             # Looks tidy, but rotation interpolates pixels,
                             # so leave FALSE if the images will be measured.
CELL_KEEP_GAPS=TRUE          # only used when CELL_SPAN="lineage": keep a
                             # blank frame where tracking lost the cell
 
 
# ── STEP 6: GFP PUNCTA AND FILAMENTS ────────────────────────────────────────
# Each cell is compared against ITSELF: the frame background is subtracted,
# the cell's median becomes its diffuse pool, and a pixel counts as structure
# when it is this much brighter than that pool, as a fraction. So a cell with
# an even glow produces nothing, which is the point.
# A cell whose GFP is this many times the background noise counts as
# expressing. Below it the cell is called "none" rather than "diffuse":
# both look featureless, but one means the reporter is absent and the other
# means it is present and unassembled, and merging them would make an
# untagged line look like an assembly-free one.
GFP_MIN_SIGNAL_OVER_BG=3.0
# gfp_area_frac — how much of the cell carries GFP above the background — is
# recorded for every cell and shown on the contact sheet as "cover". An
# untagged cell reads near 0 and an evenly glowing one near 1, so it is the
# clearest separator of none from diffuse if the median test ever disagrees.
 
# A cell whose detected structure covers less than this fraction of the cell
# is called diffuse, whatever was found in it — unless it has a filament.
# On hand-labelled data this came out as the single most useful separator of
# diffuse from punctate, above every shape measure. 0 disables it.
# Set it from --train output, but check it against the contact sheet: too
# high and small genuine foci disappear.
GFP_MIN_STRUCT_AREA_FRAC=0.0
 
GFP_STRUCT_MIN_CONTRAST=0.45  # raise if too many cells come out structured
GFP_STRUCT_NOISE_K=3.0        # floor: also needs to beat the cell's own noise
GFP_STRUCT_MIN_AREA_PX=4      # ignore anything smaller than this
 
# Ignore a rim this many pixels inside the cell outline.
#
# LEAVE THIS AT 0 unless you have a specific reason. Foci at the cell pole
# are real localisation, not bleed from the neighbour, so eroding the rim
# throws away true signal. On test data even 1 px also clipped a genuine
# faint filament running near the edge. It exists only for the case where a
# particular field really is dominated by overlapping cells.
GFP_ERODE_CELL_PX=0
 
# Where a focus sits along the cell is recorded for every cell:
#   punct_axial_pos_max   0 at mid-cell, 1 at a pole
#   n_puncta_polar        how many foci are past GFP_POLAR_THRESHOLD
# Polar and mid-cell foci are both called "punctate" — the distinction is
# biological, so it is measured and left for you to analyse rather than
# folded into the phenotype.
GFP_POLAR_THRESHOLD=0.6
GFP_CLOSE_GAPS_PX=1           # bridge single-pixel breaks, so one filament
                              # stays one object instead of several fragments
                              # too short to be recognised
 
# A faint filament never rises far enough above the diffuse pool to be found
# by brightness alone, at any threshold that is not also picking up noise.
# The ridge filter asks a different question — is this pixel part of a line —
# so faint filaments are found by their shape instead. Anything it finds that
# turns out NOT to be a line is discarded rather than counted as a punctum.
GFP_USE_RIDGE=TRUE
GFP_RIDGE_K=4.0               # lower = more sensitive, more false lines
GFP_RIDGE_SIGMAS="1 2 3"      # filament half-widths to look for, in pixels
 
# How a filament is recognised:
#   shape    scale-free descriptors, no width threshold. A filament is a
#            shape far from round and extended for its area, whatever its
#            size. This is the default and needs the least tuning.
#   microns  the older rule: minimum length, maximum width, minimum aspect.
GFP_FIL_RULE="shape"
 
# Used by the "shape" rule. Measured on known shapes, foci come out at
# circularity 1.1-1.5 with elongation 0.1-0.7, while filaments — straight,
# curved and V-shaped alike — sit at 0.17-0.55 and 2.0-4.0. These limits sit
# in the gap between them.
#   circularity  4*pi*area / perimeter^2 : 1 is a circle, 0 is a line
#   elongation   skeleton length / sqrt(area) : ~0.5 for a blob, >2 for a line
GFP_FIL_MAX_CIRCULARITY=0.60
GFP_FIL_MIN_ELONGATION=1.8
 
# Used by both rules: a structure shorter than this is not a filament
# whatever its shape.
GFP_FIL_MIN_LENGTH_UM=1.2
 
# Used by the "microns" rule only.
GFP_FIL_MAX_WIDTH_UM=0.45
GFP_FIL_MIN_ASPECT=3.0
 
# Puncta
GFP_PUNCTA_UNIT_AREA_PX=7     # area of one diffraction-limited spot
GFP_PUNCTA_MAX_AREA_PX=70     # a bigger blob counts as ONE focus, not many
 
# How far a peak must stand above the ring of cell just around it, as a
# fraction of the diffuse pool. Clearing the cell's overall threshold is not
# enough: an unevenly bright cell has broad rises that pass it without being
# foci. Raise this if too many cells come out punctate.
GFP_PUNCTA_MIN_PROMINENCE=0.25
 
# A cell must hold a new state for this many frames before the change is
# accepted, which stops calls flickering when a measurement sits on a
# threshold. The frame the change really happened on is still reported.
GFP_STATE_MIN_DURATION=2
 
# A focus sitting ON a filament is still a focus, so such a cell is "mixed"
# rather than "filamentous". A peak counts only if it stands this far above
# that filament's own median: measured on known cases, a plain filament runs
# at 1.2-1.3 and a filament carrying a focus reaches 2.0.
GFP_PUNCTA_ON_FILAMENT_RATIO=1.6
 
GFP_CALIBRATION_CELLS=6       # worked examples in calibration.png
 
# ── TUNING THE GFP CALLS ────────────────────────────────────────────────────
# Do this on a few frames before running the whole movie:
#
#   python tools/tune_gfp.py -c configs/your_config.sh --frames "0 30 60"
#
# It writes 06_gfp_tuning/ with a contact sheet of every cell filed under the
# call it got, binary masks to check in Fiji, and labels.csv. Fill in the
# your_call column by hand, then:
#
#   python tools/tune_gfp.py -c configs/your_config.sh --score
#
# for precision and recall per phenotype. To see how one setting behaves:
#
#   python tools/tune_gfp.py -c configs/your_config.sh --sweep GFP_RIDGE_K=2,3,4,6
#
# Nothing in 06_gfp_tuning/ is read by the pipeline.
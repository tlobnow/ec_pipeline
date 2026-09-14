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
# GFP_ONLY=TRUE; EXTRACT=TRUE
# SEGMENT_TRACK_ONLY=TRUE

if ([ "$MAIN" = TRUE ]); then
  LOAD_ND2=TRUE            # step 1 - read the nd2, write bf/gfp/rfp tif
  SEGMENT_CELLS=TRUE       # step 2 - Omnipose segmentation
  TRACK_CELLS=TRUE         # step 3 - btrack tracking + lineage
  INSPECT_IN_FIJI=TRUE     # step 4 - Fiji macro with track ids drawn on cells
  EXTRACT_CELL=FALSE       # step 5 - follow one cell as its own small movie
  ANALYSE_GFP=FALSE         # step 6 - find GFP puncta and filaments
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

# ── EXPERIMENT ──────────────────────────────────────────────────────────────
EXPERIMENT_NAME="20260831_Camai_GFP-PNP_swap_Inf_T4_MOI_0.2__02"
ND2_PATH="/Users/u_lobnow/Desktop/bact_microscopy/20260831/20260831_Camai_GFP-PNP_swap_Inf_T4_MOI0.2003.nd2"
OUTPUT_ROOT="/Users/u_lobnow/Library/CloudStorage/Dropbox-WeizmannInstitute/Finn Lobnow/Wein lab/Projects/BacHuman_DFDs/Experiments/Microscopy/ecoli_pipeline/output"
CONDITION="infected"                # infected | uninfected

# Fields of view. "all" = every position in the nd2, otherwise list them
# separated by spaces, e.g. POSITIONS="0 1 2"

POSITIONS="0"
# POSITIONS="all"

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
CH_BF="PHASE"                       # phase contrast / brightfield
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
 
CROP_HALF_SIZE=0 #200                  # 200 -> 400x400 centre crop; 0 = full frame
 
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
# How cells are linked between frames:
#   overlap  (default) the same cell one frame later occupies almost the same
#            pixels, so link by how much the masks overlap. Bacteria on agar
#            barely move but grow and divide, which is exactly the case a
#            distance-based tracker struggles with: the centroid shifts as a
#            cell elongates, and in a packed microcolony the nearest object
#            is often a neighbour rather than the same cell. A mother claimed
#            by two objects in the next frame is simply a division.
#   btrack   the Bayesian tracker. Better when cells genuinely move around.
TRACK_METHOD="overlap"
 
# How much of a cell must be covered by the same cell in the previous frame
# for the link to be accepted. Lower it if many cells appear from nowhere
# mid-movie, which the log warns about.
TRACK_MIN_OVERLAP_LINK=0.2
 
BTRACK_CONFIG="btrack_config.json"  # only used when TRACK_METHOD="btrack";
                                    # relative paths resolve next to this file
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
# "last" means the final frame of whichever movie is being processed, so a
# config works unchanged on runs of different lengths. Times past the end are
# dropped and listed in the log.
TIMEPOINTS_MIN="0 30 45 60 90 120 150 last"
 
 
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
 
 
# ── DO YOU EVEN NEED TRACKING? ──────────────────────────────────────────────
# Tracking answers "is this the same cell as the one in the last frame". That
# matters for single-cell movies and lineages. It does NOT matter for asking
# what fraction of cells carry a filament — that is a proportion measured
# among the cells in a frame.
#
# So if tracking is struggling (dense microcolonies are the usual reason),
# the phenotype analysis still runs on the segmentation alone:
#
#     SEGMENT_CELLS=TRUE  TRACK_CELLS=FALSE  ANALYSE_GFP=TRUE
#
# The cost is that ids are per frame only, so a cell seen in 40 frames would
# be counted 40 times. Analyse ONE timepoint per field instead, and the
# counts stay honest:
#
#     python tools/compare_conditions.py --frames "30" ...
 
 
# ── STEP 6: FINDING FILAMENTS ───────────────────────────────────────────────
# Every setting has a sensible default in the code, so only the ones worth
# changing are listed here. The puncta settings still exist and still work —
# they are just not the question at the moment, so they are left at their
# defaults rather than cluttering this file. To see them all:
#     grep -o 'GFP_[A-Z_]*' scripts/common.py | sort -u
 
# Which cells count as expressing at all. Below this a cell is called "none"
# rather than "diffuse": both look featureless, but one has no reporter and
# the other has one that has not assembled.
GFP_MIN_SIGNAL_OVER_BG=3.0
 
# How much brighter than the cell's own diffuse pool a pixel must be to count
# as structure. Raise it if too much is being picked up.
GFP_STRUCT_MIN_CONTRAST=0.45
 
# A faint filament never rises far above the pool at any threshold that is
# not also catching noise, so the ridge filter looks for LINES instead —
# shape rather than brightness. This is what finds the faint ones.
GFP_USE_RIDGE=TRUE
GFP_RIDGE_K=4.0               # lower = more sensitive, more false lines
 
# What counts as a filament. "shape" uses scale-free descriptors and needs no
# width threshold, so it does not have to be retuned when magnification or
# cell size changes.
#   circularity  4*pi*area / perimeter^2 : 1 is a circle, 0 is a line
#   elongation   skeleton length / sqrt(area) : ~0.5 for a blob, >2 for a line
# Measured on known shapes: foci sit at 1.1-1.5 and 0.1-0.7, filaments —
# straight, curved and V-shaped alike — at 0.17-0.55 and 2.0-4.0.
GFP_FIL_RULE="shape"
GFP_FIL_MAX_CIRCULARITY=0.60
GFP_FIL_MIN_ELONGATION=1.8
GFP_FIL_MIN_LENGTH_UM=1.2     # nothing shorter is a filament, whatever its shape
 
# Bridge single-pixel breaks, so one filament stays one object instead of
# fragments too short to be recognised.
GFP_CLOSE_GAPS_PX=1
 
 
# ── CHECKING THE CALLS ──────────────────────────────────────────────────────
# Before running everything, look at a few frames:
#
#   python tools/tune_gfp.py -c <this config> --frames "0 30 60"
#
# It writes 06_gfp_tuning/ with a contact sheet of every cell filed under the
# call it got, and binary masks to check in Fiji. Nothing there is read by
# the pipeline. To see how one setting behaves:
#
#   python tools/tune_gfp.py -c <this config> --sweep GFP_RIDGE_K=2,3,4,6
 
 
# ── FILAMENT CLASSIFIER (run_filaments.sh) ──────────────────────────────────
# Which frames to take cells from. "last" is the final frame of the movie.
FILAMENT_FRAMES="0 20 40 60 last"
 
# ONE folder for every experiment. Point every config at the same path: the
# training set grows as more experiments are sorted, and one model then works
# for all of them. File names carry the experiment, so nothing collides.
FILAMENT_TRAINING_DIR="$OUTPUT_ROOT/filament_training"
 
# Where the trained model lives. Also shared across experiments.
FILAMENT_MODEL="$OUTPUT_ROOT/filament_model.joblib"
 
# Only export cells the model is at least this sure about, when cutting
# crops to measure. Lower it if you want a length distribution that
# represents the whole population rather than the clearest cases.
FILAMENT_MIN_CONFIDENCE=0.8
 
# Cells per field when cutting crops to measure (0 = every one).
FILAMENT_MAX_PER_FIELD=20
 
# How many uncertain cells a review round exports.
FILAMENT_REVIEW_N=100
 
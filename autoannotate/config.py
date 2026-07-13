"""Environment, path and device configuration for AutoAnnotate.

Importing this module has side effects on purpose: it sets the environment
variables that must exist BEFORE HuggingFace / transformers are imported,
loads .env, and optionally logs into HuggingFace. Every other module gets
its paths from here so the app runs the same from `python -m autoannotate`,
run_app.py, or a future frozen executable.
"""
import os
import platform as _platform

from dotenv import load_dotenv

# Silence the huggingface/tokenizers fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# This file lives at <repo>/autoannotate/config.py, so the repo root is one
# directory up. The notebook-era cwd walk is gone: modules know where they are.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# .env is loaded HERE, before the device block below, and not further down where
# it used to sit. load_dotenv does not overwrite a variable that already exists,
# so the old order lost: the device block ran first, saw no AUTOANNOTATE_SD_DEVICE
# (because .env had not been read yet), pinned it to "cpu" on macOS, and the
# AUTOANNOTATE_SD_DEVICE=mps that the user had put in .env could never take
# effect. .env is the documented place to configure this, so it has to be
# readable before anything defaults it.
load_dotenv(os.path.join(REPO_ROOT, ".env"))

# ---------------------------------------------------------------------------
# PER-OS COMPUTE-DEVICE CONFIG  (review when moving to a new machine)
#
# AutoAnnotate runs on macOS, Windows and Linux. File paths are derived
# portably below, so the ONLY setting that differs between operating systems
# is which device runs the Stable Diffusion background-variation step.
# pipeline.sd._sd_select_device() reads the AUTOANNOTATE_SD_DEVICE env var set
# here; the detection/segmentation models (DINO, YOLOE, SAM2/SAM3) pick their
# own device automatically.
#
# To FORCE a device, UNCOMMENT exactly ONE line for your OS -- an explicit
# value wins over the platform auto-detect that follows.
#
#   --- macOS (Apple Silicon / Intel) ---
# os.environ["AUTOANNOTATE_SD_DEVICE"] = "cpu"   # 8GB Macs: MPS hangs SD-1.5, use CPU
# os.environ["AUTOANNOTATE_SD_DEVICE"] = "mps"   # Macs with >8GB unified memory (faster)
#   --- Windows / Linux with an NVIDIA GPU ---
# os.environ["AUTOANNOTATE_SD_DEVICE"] = "cuda"  # fastest; fp16 on the GPU
#   --- Windows / Linux, CPU only ---
# os.environ["AUTOANNOTATE_SD_DEVICE"] = "cpu"
#
# Auto-detect default (when none of the above is uncommented and the var is
# not already set in the shell environment):
#   macOS         -> "cpu"  (safe on 8GB; MPS has hung SD-1.5 for 15+ min)
#   Windows/Linux -> leave UNSET so _sd_select_device() prefers CUDA, then
#                    MPS, then CPU -- correct for a GPU box and a CPU box alike.
if "AUTOANNOTATE_SD_DEVICE" not in os.environ:
    if _platform.system() == "Darwin":
        os.environ["AUTOANNOTATE_SD_DEVICE"] = "cpu"
# ---------------------------------------------------------------------------

# Windows: register the CUDA runtime DLL directories. Since Python 3.8,
# compiled extension modules (.pyd) no longer search PATH for their dependent
# DLLs, only directories registered via os.add_dll_directory(). Without this,
# `import groundingdino._C` fails with "DLL load failed" even when CUDA's bin
# is on PATH, and DINO silently falls back to slow pure-Python deformable
# attention. No-op on macOS/Linux and on Windows machines without CUDA_PATH.
# add_dll_directory returns a handle that UNREGISTERS the directory when it is
# closed, and it closes on garbage collection. Dropping it would remove the
# search path again before groundingdino._C is ever imported, so the handles are
# parked here for the lifetime of the process.
_CUDA_DLL_HANDLES = []

if _platform.system() == "Windows":
    _cuda_path = os.environ.get("CUDA_PATH", "")
    if _cuda_path:
        for _sub in ("bin", os.path.join("bin", "x64")):
            _dll_dir = os.path.join(_cuda_path, _sub)
            if os.path.isdir(_dll_dir):
                try:
                    _CUDA_DLL_HANDLES.append(os.add_dll_directory(_dll_dir))
                except (OSError, AttributeError):
                    pass

    # Windows consoles default sys.stdout to the ANSI code page (cp1252), so a
    # single print of a non-ASCII image filename raises UnicodeEncodeError and
    # kills the run mid-batch. The app's own log lines are all ASCII; the
    # filenames it interpolates are not under our control. errors="replace"
    # degrades an unprintable character to '?' instead of aborting.
    # No-op on macOS/Linux, which are already UTF-8.
    import sys as _sys

    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass   # not a real console (pytest capture, pythonw, a pipe)

# Model weights (sam3.pt, sam2_t.pt, yoloe-11l-seg.pt, ...) and the default
# label output dirs live in the historical "GUI and Pipeline" folder. Working
# directory no longer matters: resolve everything against this.
WEIGHTS_DIR = os.path.join(REPO_ROOT, "GUI and Pipeline")

# Where cwd-relative artifacts (DINO-labels, optimizer save files) land when a
# caller does not pass an explicit directory. Kept at "GUI and Pipeline" for
# continuity with where earlier versions of the app wrote them.
BASE_DIR = WEIGHTS_DIR

# Per-user settings that must outlive a single run and follow the user rather
# than the checkout. expanduser resolves to ~/.autoannotate on macOS/Linux and
# %USERPROFILE%\.autoannotate on Windows, so the same code path serves every OS.
#
# NOTHING currently writes here. The box-prompt class names used to, and are now
# deliberately session-only (see gui/session_state.py: they outlive the window
# but never the app). This is kept as the one place a future persisted setting
# should go, and check_environment probes it so a read-only home directory is
# caught before something depends on it.
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".autoannotate")


def user_config_path(filename):
    """Absolute path to `filename` inside USER_CONFIG_DIR, creating the
    directory on first use. Returns the path even when creation fails so the
    caller's own try/except decides whether a missing settings file matters."""
    try:
        os.makedirs(USER_CONFIG_DIR, exist_ok=True)
    except OSError:
        pass
    return os.path.join(USER_CONFIG_DIR, filename)

# Master switch for the chatty [TAG] debug prints ([_get_model], [YOLOE-LOAD],
# [DINO-FILTER], [ROUND-TRIP CHECK] OK). False keeps normal runs quiet; real
# errors and [ROUND-TRIP CHECK] FAILED always print regardless of this flag.
# Read from the environment (.env or shell) so it can be toggled WITHOUT
# editing code; truthy = 1/true/yes/on (case-insensitive). Default off.
AUTOANNOTATE_DEBUG = os.environ.get("AUTOANNOTATE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

# Hugging Face login is OPTIONAL. Every model the GUI uses downloads
# anonymously: SAM2 / YOLOE come from ultralytics' own asset server, and
# bert-base-uncased (DINO text encoder), SmolVLM, and SD-1.5 are PUBLIC HF
# repos. A token is only needed for a FIRST-TIME download of the gated
# sam3.pt (facebook/sam3); once sam3.pt sits in "GUI and Pipeline/" the
# app runs fully token-free.
hf_token = os.environ.get("HF_TOKEN")
if hf_token and hf_token != "paste_your_hf_token_here":
    from huggingface_hub import login
    login(token=hf_token)
else:
    print("[HF] No HF_TOKEN set -- running token-free. Public models "
          "download anonymously; SAM3 needs sam3.pt already present.")

# Auto-derive GROUNDING_DINO_DIR: GroundingDINO sits at a fixed location under
# the repo root, so a fresh clone needs no .env edit. A .env value still wins.
# The study folder is git-tracked as "autoannotate study" (with a space), but
# zips/transfers sometimes mangle it into "autoannotate_study"; accept both so
# neither checkout style bricks startup.
_gd_candidates = []
_gd_env = os.environ.get("GROUNDING_DINO_DIR")
if _gd_env:
    _gd_candidates.append(_gd_env)
for _study in ("autoannotate study", "autoannotate_study"):
    _gd_candidates.append(os.path.join(REPO_ROOT, _study, "GroundingDINO"))

GROUNDING_DINO_DIR = None
for _cand in _gd_candidates:
    if os.path.isdir(_cand):
        GROUNDING_DINO_DIR = _cand
        break
if GROUNDING_DINO_DIR is None:
    raise EnvironmentError(
        "GroundingDINO not found. Tried: "
        + ", ".join(repr(c) for c in _gd_candidates)
        + ". Set GROUNDING_DINO_DIR in AutoAnnotate/.env to its absolute path.")
if _gd_env and GROUNDING_DINO_DIR != _gd_env:
    print(f"[config] GROUNDING_DINO_DIR from .env is not a directory "
          f"({_gd_env!r}); using {GROUNDING_DINO_DIR!r} instead.")

# Default cap on detection area as a fraction of the image, shared by the GUI
# (_max_area_frac) and documented in HOW_TO_RUN. Override per session with the
# AUTOANNOTATE_MAX_AREA_FRAC env var (.env or shell).
DEFAULT_MAX_AREA_FRAC = 0.5


def effective_max_area_frac():
    """The max-area fraction the app will actually use this session.

    Single source of truth for parsing AUTOANNOTATE_MAX_AREA_FRAC: an unset,
    unparseable, or out-of-range (0, 1] value falls back to
    DEFAULT_MAX_AREA_FRAC. check_environment reports this rather than the raw
    env string, so what it prints is what inference will really apply.
    """
    try:
        val = float(os.environ.get("AUTOANNOTATE_MAX_AREA_FRAC",
                                   str(DEFAULT_MAX_AREA_FRAC)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_AREA_FRAC
    if not (0.0 < val <= 1.0):
        return DEFAULT_MAX_AREA_FRAC
    return val

# Default image source directory. Subfolders here (Berry, Buds, Fescue,
# Red_leaf) are the image categories.
CUMULATIVE_DIR = os.path.normpath(os.path.join(REPO_ROOT, "cumulative"))


def weights_path(name):
    """Resolve a checkpoint filename against WEIGHTS_DIR when it exists there.

    Falls back to the bare name so ultralytics can still auto-download the
    public checkpoints (sam2_t.pt, yoloe-*.pt) into the working directory on a
    fresh machine.
    """
    if os.path.isabs(name):
        return name
    cand = os.path.join(WEIGHTS_DIR, name)
    return cand if os.path.exists(cand) else name

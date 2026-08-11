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

# Do not call huggingface_hub.login() at application startup. Every runtime
# model is public or already on disk, and huggingface_hub/transformers already
# read HF_TOKEN themselves when a request actually needs authentication.
# Proactively validating a token makes a fully cached/offline launch perform a
# network request and abort before any model is loaded.
hf_token = os.environ.get("HF_TOKEN")
if not hf_token or hf_token == "paste_your_hf_token_here":
    print("[HF] No HF_TOKEN set -- running token-free. Public models "
          "download anonymously; SAM3 needs sam3.pt already present.")

# Auto-derive GROUNDING_DINO_DIR: GroundingDINO sits at a fixed location under
# the repo root, so a fresh clone needs no .env edit. A .env value still wins.
# The study folder is git-tracked as "autoannotate study" (with a space), but
# zips/transfers sometimes mangle it into "autoannotate_study"; accept both so
# neither checkout style bricks startup.
#
# Picking the FIRST directory that exists is not good enough, and cost a day on
# a Windows 11 machine in 2026-08. That machine had both spellings. Python
# imported groundingdino from the underscore copy (the absolute path pip baked
# into the editable install), while this block picked the space copy because it
# is listed first. Code came from one tree, model configs and weights from
# another, and a git pull only ever refreshed one of them.
#
# So the tree the interpreter will ACTUALLY import from is asked for first, via
# find_spec, which locates the package without importing it (no torch pulled in
# before autoannotate/__init__.py has set the CUDA allocator flag, and no cost
# worth measuring). Candidates are then scored on whether they really hold the
# config modules and the weights, because both are gitignored and therefore
# exist only in the tree they were built or downloaded into.
def _gd_import_tree():
    """The GroundingDINO checkout Python would import, or None."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("groundingdino")
    except (ImportError, ValueError, AttributeError):
        # ValueError when something has put a spec-less stub in sys.modules,
        # which is exactly what the headless test suite does.
        return None
    origin = getattr(spec, "origin", None) if spec else None
    if not origin:
        return None
    # <tree>/groundingdino/__init__.py -> <tree>
    return os.path.dirname(os.path.dirname(os.path.abspath(origin)))


def _gd_score(tree):
    """How complete a checkout is: (has config modules, has weights)."""
    has_cfg = os.path.isdir(os.path.join(tree, "groundingdino", "config"))
    weights_dir = os.path.join(tree, "weights")
    try:
        has_weights = any(n.endswith(".pth") for n in os.listdir(weights_dir))
    except OSError:
        has_weights = False
    return (has_cfg, has_weights)


def gd_candidates(repo_root, env=None, import_dir=None):
    """Candidate GroundingDINO trees, best-informed first, without duplicates."""
    out = []
    if env:
        out.append(env)
    if import_dir:
        out.append(import_dir)
    for study in ("autoannotate study", "autoannotate_study"):
        out.append(os.path.join(repo_root, study, "GroundingDINO"))
    seen = set()
    unique = []
    for c in out:
        key = os.path.normcase(os.path.abspath(c))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def resolve_gd_dir(repo_root, env=None, import_dir=None):
    """Choose the GroundingDINO tree, or None when none exists.

    An explicit GROUNDING_DINO_DIR wins outright: that is a decision, not a
    guess. Otherwise the most COMPLETE existing candidate wins, with ties broken
    by candidate order, which puts the tree Python actually imports from ahead
    of a bare spelling guess. Completeness matters because the config modules
    and the weights are both gitignored, so a tree can exist and still be the
    wrong answer.
    """
    if env and os.path.isdir(env):
        return env
    existing = [c for c in gd_candidates(repo_root, env, import_dir)
                if os.path.isdir(c)]
    if not existing:
        return None
    return max(existing, key=_gd_score)


_gd_env = os.environ.get("GROUNDING_DINO_DIR")
_gd_import_dir = _gd_import_tree()
_gd_candidates = gd_candidates(REPO_ROOT, _gd_env, _gd_import_dir)
GROUNDING_DINO_DIR = resolve_gd_dir(REPO_ROOT, _gd_env, _gd_import_dir)

if GROUNDING_DINO_DIR is None:
    raise EnvironmentError(
        "GroundingDINO not found. Tried: "
        + ", ".join(repr(c) for c in _gd_candidates)
        + ". Set GROUNDING_DINO_DIR in AutoAnnotate/.env to its absolute path.")
if _gd_env and GROUNDING_DINO_DIR != _gd_env:
    print(f"[config] GROUNDING_DINO_DIR from .env is not a directory "
          f"({_gd_env!r}); using {GROUNDING_DINO_DIR!r} instead.")

def _gd_same_tree(a, b):
    """True when two paths are the same directory, through a symlink or a
    Windows junction as well, since linking the two spellings together is the
    recommended permanent fix and must not be reported as a split."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return (os.path.normcase(os.path.realpath(a))
                == os.path.normcase(os.path.realpath(b)))


# Loud, because a silent split is what made the Windows failure so hard to read.
if (_gd_import_dir and os.path.isdir(_gd_import_dir)
        and not _gd_same_tree(_gd_import_dir, GROUNDING_DINO_DIR)):
    print("[config] WARNING: GroundingDINO code and data come from DIFFERENT trees.")
    print(f"[config]   code    : {_gd_import_dir}")
    print(f"[config]   configs : {GROUNDING_DINO_DIR}")
    print('[config]   Run: python "GUI and Pipeline/check_environment.py" for the fix.')

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

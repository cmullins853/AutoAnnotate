"""Environment, path and device configuration for AutoAnnotate.

Importing this module has side effects on purpose: it sets the environment
variables that must exist BEFORE HuggingFace / transformers are imported,
loads .env, and optionally logs into HuggingFace. Every other module gets
its paths from here so the app runs the same from `python -m autoannotate`,
run_app.py, or a future frozen executable.
"""
import os
import platform as _platform

# Silence the huggingface/tokenizers fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

from dotenv import load_dotenv

# This file lives at <repo>/autoannotate/config.py, so the repo root is one
# directory up. The notebook-era cwd walk is gone: modules know where they are.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model weights (sam3.pt, sam2_t.pt, yoloe-11l-seg.pt, ...) and the default
# label output dirs live in the historical "GUI and Pipeline" folder. Working
# directory no longer matters: resolve everything against this.
WEIGHTS_DIR = os.path.join(REPO_ROOT, "GUI and Pipeline")

# Where cwd-relative artifacts (DINO-labels, optimizer save files) land when a
# caller does not pass an explicit directory. Kept at "GUI and Pipeline" for
# continuity with where earlier versions of the app wrote them.
BASE_DIR = WEIGHTS_DIR

load_dotenv(os.path.join(REPO_ROOT, ".env"))

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
GROUNDING_DINO_DIR = os.environ.get("GROUNDING_DINO_DIR")
if not GROUNDING_DINO_DIR:
    GROUNDING_DINO_DIR = os.path.join(REPO_ROOT, "autoannotate study", "GroundingDINO")
if not os.path.isdir(GROUNDING_DINO_DIR):
    raise EnvironmentError(
        f"GroundingDINO not found at {GROUNDING_DINO_DIR!r}. Set GROUNDING_DINO_DIR "
        "in AutoAnnotate/.env to its absolute path.")

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

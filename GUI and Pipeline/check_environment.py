#!/usr/bin/env python3
"""
check_environment.py - cross-OS readiness check for AutoAnnotate.

Run this on macOS, Windows, or Linux BEFORE launching the app to confirm
the machine can run the app. It does NOT load any model (fast, no downloads); it
only verifies the Python deps, reports the compute device the app will pick, and
checks that the weight files are where the GUI expects them.

Usage:
    python "GUI and Pipeline/check_environment.py"

Exit code 0 = all critical checks passed; 1 = something required is missing.
The weight-file checks are advisory (you can fetch them later), so a missing
weight warns but does not by itself fail the run; missing Python packages do.
"""
import importlib
import os
import platform
import sys

OK   = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

_failed = False


def _mark_fail():
    global _failed
    _failed = True


# ── locate repo root by walking up from this file ─────────────────────────────
def find_repo_root(start):
    d = os.path.abspath(start)
    for _ in range(6):
        # The study folder is tracked as "autoannotate study" but transfers
        # sometimes mangle it into "autoannotate_study"; accept both.
        has_study = any(os.path.isdir(os.path.join(d, s))
                        for s in ("autoannotate study", "autoannotate_study"))
        if has_study and os.path.exists(os.path.join(d, ".env.example")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.normpath(os.path.join(os.path.abspath(start), ".."))


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = find_repo_root(HERE)


def load_dotenv_if_present():
    """Load REPO_ROOT/.env into os.environ if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(REPO_ROOT, ".env"))
    except Exception:
        pass


def header(title):
    print(f"\n{title}\n" + "-" * len(title))


def main():
    print("=" * 60)
    print(" AutoAnnotate environment check")
    print("=" * 60)

    header("System")
    print(f"  OS           : {platform.system()} ({platform.platform()})")
    print(f"  Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"  Repo root    : {REPO_ROOT}")

    load_dotenv_if_present()

    # ── required Python packages ──────────────────────────────────────────────
    header("Python packages (required)")
    required = [
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PyQt5", "PyQt5"),
        ("PIL", "Pillow"),
        ("shapely", "shapely"),
        ("ultralytics", "ultralytics"),
        ("dotenv", "python-dotenv"),
        ("transformers", "transformers"),
    ]
    for mod, pip_name in required:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {OK} {mod:<14} {ver}")
        except Exception as e:
            print(f"  {FAIL} {mod:<14} missing -> pip install {pip_name}  ({e.__class__.__name__})")
            _mark_fail()

    header("Python packages (optional)")
    for mod, note in [("diffusers", "Stable Diffusion variations"),
                      ("groundingdino", "DINO detector (pip install -e the GroundingDINO dir)")]:
        try:
            importlib.import_module(mod)
            print(f"  {OK} {mod:<14} present")
        except Exception:
            print(f"  {WARN} {mod:<14} not found -> {note}")

    # ── per-user settings + image I/O ─────────────────────────────────────────
    # The box-prompt class names persist to a per-user JSON file, and every
    # image read/write goes through numpy so paths with non-ASCII characters
    # work on Windows (OpenCV's own imread/imwrite use the ANSI API there and
    # fail silently). Both are exercised here rather than trusted.
    header("Per-user settings and image I/O")
    try:
        sys.path.insert(0, REPO_ROOT)
        from autoannotate.config import USER_CONFIG_DIR, user_config_path
        probe = user_config_path(".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        print(f"  {OK} settings dir writable : {USER_CONFIG_DIR}")
    except Exception as e:
        print(f"  {WARN} settings dir not writable ({e.__class__.__name__}: {e})")
        print( "       Box class names will not persist between runs.")

    try:
        import tempfile

        import numpy as _np

        from autoannotate.imageio import imread_unicode, imwrite_unicode
        with tempfile.TemporaryDirectory() as _t:
            _p = os.path.join(_t, "bäi字", "probe.png")
            wrote = imwrite_unicode(_p, _np.zeros((4, 4, 3), dtype=_np.uint8))
            read_back = imread_unicode(_p) is not None
        if wrote and read_back:
            print(f"  {OK} non-ASCII image paths : read + write OK")
        else:
            print(f"  {FAIL} non-ASCII image paths : write={wrote} read={read_back}")
            _mark_fail()
    except Exception as e:
        print(f"  {FAIL} image I/O check failed ({e.__class__.__name__}: {e})")
        _mark_fail()

    # ── compute device the app will use ───────────────────────────────────────
    header("Compute device")
    try:
        import torch
        cuda = torch.cuda.is_available()
        mps = bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available())
        print(f"  CUDA available : {cuda}")
        print(f"  MPS available  : {mps}")

        # Mirror config.py's auto-detect + pipeline.sd._sd_select_device.
        override = os.environ.get("AUTOANNOTATE_SD_DEVICE", "").lower().strip()
        if not override and platform.system() == "Darwin":
            override = "cpu"  # config.py auto-detect default on macOS
        if override == "cuda" and cuda:
            sd = "cuda (fp16)"
        elif override == "mps" and mps:
            sd = "mps (fp32)"
        elif override == "cpu":
            sd = "cpu (fp32)"
        elif cuda:
            sd = "cuda (fp16)"
        elif mps:
            sd = "mps (fp32)"
        else:
            sd = "cpu (fp32)"
        env_show = os.environ.get("AUTOANNOTATE_SD_DEVICE") or "(unset -> auto)"
        print(f"  AUTOANNOTATE_SD_DEVICE env : {env_show}")
        print(f"  Stable Diffusion will run on: {sd}")
        budget = os.environ.get("AUTOANNOTATE_MODEL_BUDGET_GB")
        print(f"  Model RAM budget : {budget + ' GB' if budget else '(unset -> unbounded cache)'}")
        # Effective .env/shell tuning values, so users can verify their .env
        # actually took effect (the app loads .env itself via python-dotenv;
        # editor warnings about terminal env injection are unrelated).
        # Report the value inference will ACTUALLY apply, not the raw string: an
        # unparseable or out-of-range .env entry silently falls back, and printing
        # it verbatim would tell the user their setting took effect when it did not.
        maf_raw = os.environ.get("AUTOANNOTATE_MAX_AREA_FRAC")
        from autoannotate.config import effective_max_area_frac
        maf_eff = effective_max_area_frac()
        if maf_raw is None:
            maf_show = f"{maf_eff:.2f} (unset -> default)"
        else:
            try:
                accepted = abs(float(maf_raw) - maf_eff) < 1e-9
            except (TypeError, ValueError):
                accepted = False
            maf_show = (f"{maf_eff:.2f}" if accepted
                        else f"{maf_eff:.2f} ({maf_raw!r} rejected -> default)")
        print(f"  AUTOANNOTATE_MAX_AREA_FRAC : {maf_show}")
    except Exception as e:
        print(f"  {WARN} could not query torch devices: {e}")

    # ── weight files (advisory) ───────────────────────────────────────────────
    header("Weight files (advisory)")
    # Mirror config.py's resolution: env value first, then either study
    # folder spelling (space-named as git tracks it, or underscore).
    gd_candidates = []
    if os.environ.get("GROUNDING_DINO_DIR"):
        gd_candidates.append(os.environ["GROUNDING_DINO_DIR"])
    for study in ("autoannotate study", "autoannotate_study"):
        gd_candidates.append(os.path.join(REPO_ROOT, study, "GroundingDINO"))
    gd_dir = next((c for c in gd_candidates if os.path.isdir(c)), gd_candidates[-2])
    weights = [
        (os.path.join(HERE, "sam2_t.pt"), "SAM2 tiny segmenter (auto-downloads on first run)"),
        (os.path.join(HERE, "sam3.pt"), "SAM3 detector/segmenter (gated; fetch from HF)"),
        (os.path.join(HERE, "yoloe-11l-seg.pt"), "YOLOE detector"),
        (os.path.join(HERE, "yoloe-11l-seg-pf.pt"), "YOLOE prompt-free dependency"),
        (os.path.join(gd_dir, "weights", "groundingdino_swint_ogc.pth"), "DINO SwinT"),
        (os.path.join(gd_dir, "weights", "groundingdino_swinb_cogcoor.pth"), "DINO SwinB (optional)"),
    ]
    print(f"  GROUNDING_DINO_DIR : {gd_dir}"
          + ("" if os.path.isdir(gd_dir) else f"   {WARN} not a directory"))
    for path, note in weights:
        if os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            # >=1 MB = a real checkpoint; <1 MB = a failed/stub download
            # (these models legitimately range from ~40 MB SAM2-tiny upward).
            tag = OK if mb >= 1 else WARN
            print(f"  {tag} {os.path.basename(path):<34} {mb:7.0f} MB  {note}")
        else:
            print(f"  {WARN} {os.path.basename(path):<34} {'missing':>10}  {note}")

    # ── verdict ───────────────────────────────────────────────────────────────
    header("Result")
    if _failed:
        print(f"  {FAIL} Required packages are missing. Install them, then re-run.")
        return 1
    print(f"  {OK} All required packages present. Missing weights (if any) above")
    print("       can be fetched per HOW_TO_RUN.txt STEP 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


# ── GroundingDINO install integrity ───────────────────────────────────────────
# The study folder is tracked as "autoannotate study" but transfers mangle the
# space into an underscore, so a machine can end up holding BOTH spellings. That
# is where this gets dangerous, because two independent things resolve the
# folder separately:
#
#   * `import groundingdino` follows the absolute path baked into the editable
#     install (site-packages/__editable___groundingdino_*_finder.py), which is
#     whichever spelling was passed to `pip install -e` on that machine.
#   * config.GROUNDING_DINO_DIR re-derives its own path and prefers the SPACE
#     spelling, because that is first in its candidate list.
#
# Only the space folder is git-tracked. So a `git pull` refreshes one tree while
# the other silently rots, and if the editable install points at the untracked
# one, the app imports stale or incomplete code. Seen in the field on Windows 11
# 2026-08: an update refilled the tracked tree while the underscore tree kept
# only its compiled leftovers (_C.pyd, .obj, __pycache__) and lost its .py
# sources, so the splash screen came up and entering Manual mode died with
# ModuleNotFoundError: No module named 'groundingdino'.
#
# None of that is visible from a plain import check, so it is diagnosed here.

# Relative to <tree>/groundingdino/. Enough to prove the sources are really
# there rather than just the compiled extension: these are the modules the
# app's own import chain (pipeline.dino -> groundingdino.util.inference) walks.
GD_REQUIRED_SOURCES = (
    "__init__.py",
    os.path.join("util", "__init__.py"),
    os.path.join("util", "inference.py"),
    os.path.join("config", "__init__.py"),
    os.path.join("models", "__init__.py"),
    os.path.join("datasets", "__init__.py"),
)

GD_COMPILED_SUFFIXES = (".pyd", ".so", ".obj", ".dll")


def gd_tree_state(tree):
    """Describe one GroundingDINO checkout at `tree`.

    Returns a dict with the package dir, which required sources are missing,
    and whether compiled artifacts are present. Sources missing WITH compiled
    artifacts present is the signature of a partially-synced copy.
    """
    pkg = os.path.join(tree, "groundingdino")
    state = {"tree": tree, "pkg": pkg, "exists": os.path.isdir(tree),
             "pkg_exists": os.path.isdir(pkg), "missing": [], "compiled": False,
             "ext": False, "weights": []}
    if not state["pkg_exists"]:
        return state
    state["missing"] = [rel for rel in GD_REQUIRED_SOURCES
                        if not os.path.exists(os.path.join(pkg, rel))]
    try:
        for name in os.listdir(pkg):
            if name.endswith(GD_COMPILED_SUFFIXES) or name == "__pycache__":
                state["compiled"] = True
            # The _C extension specifically: this is the slow CUDA build, and
            # .gitignore excludes *.so/*.pyd, so it exists ONLY in the tree it
            # was built in. Any advice that moves someone off that tree has to
            # say so, or it silently costs them the build.
            if name.startswith("_C") and name.endswith(GD_COMPILED_SUFFIXES):
                state["ext"] = True
    except OSError:
        pass
    # weights/ is gitignored too, so the .pth files also live in exactly one
    # tree. Losing track of which is the second half of the same trap.
    try:
        state["weights"] = sorted(n for n in os.listdir(os.path.join(tree, "weights"))
                                  if n.endswith(".pth"))
    except OSError:
        state["weights"] = []
    return state


def editable_target(search_dirs=None):
    """The path `pip install -e` baked into the GroundingDINO editable install.

    pip writes a finder module into site-packages holding an absolute MAPPING
    to the source tree. Reading it answers "which copy is Python configured to
    import?" even when the import FAILS, which is exactly the case where the
    question matters most and where nothing else can answer it.

    Returns the mapped `groundingdino` package directory, or None.
    """
    import re as _re
    if search_dirs is None:
        try:
            import site
            search_dirs = list(site.getsitepackages())
            user = site.getusersitepackages()
            if isinstance(user, str):
                search_dirs.append(user)
        except Exception:
            search_dirs = []
        search_dirs = [d for d in search_dirs if d]
    for d in search_dirs:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not (name.startswith("__editable__") and "groundingdino" in name
                    and name.endswith(".py")):
                continue
            try:
                with open(os.path.join(d, name), encoding="utf-8") as fh:
                    body = fh.read()
            except OSError:
                continue
            m = _re.search(r"['\"]groundingdino['\"]\s*:\s*['\"](.+?)['\"]", body)
            if m:
                return m.group(1)
    return None


def _same_tree(a, b):
    """True when two paths are the same directory, including through a symlink
    or a Windows junction (which is the durable fix for the split, so it must
    not be reported as a second copy)."""
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def diagnose_groundingdino(repo_root, import_origin, import_error, config_dir,
                           editable_dir=None):
    """Work out what is wrong with the GroundingDINO install, if anything.

    `import_origin` is the directory the imported `groundingdino` package was
    loaded from (the parent of its __init__.py), or None if the import failed.
    Passed in rather than imported here so every layout can be tested without
    reproducing it on the machine running the tests.

    Returns (lines, fatal): report lines as (tag, text), and whether the state
    guarantees a crash once the app reaches the detector.
    """
    lines = []
    fatal = False

    trees = []
    for study in ("autoannotate study", "autoannotate_study"):
        tree = os.path.join(repo_root, study, "GroundingDINO")
        if os.path.isdir(tree):
            trees.append(gd_tree_state(tree))

    tracked = next((t for t in trees if "autoannotate study" in t["tree"]), None)

    # The import is the thing that actually decides whether the app runs.
    origin_tree = None
    if import_origin:
        # import_origin points at <tree>/groundingdino; step up to the tree.
        origin_tree = os.path.dirname(os.path.abspath(import_origin))
        # An import can succeed off a tree that has still lost submodules the
        # app imports later, so this line must not read as an all-clear when
        # the detail below is about to fail it.
        origin_state = next((t for t in trees
                             if _same_tree(t["tree"], origin_tree)), None)
        origin_tag = WARN if (origin_state and origin_state["missing"]) else OK
        lines.append((origin_tag, f"groundingdino imports from : {import_origin}"))
    else:
        fatal = True
        lines.append((FAIL, f"groundingdino cannot be imported: {import_error}"))

    # Where pip was told to look. This is the only way to name the offending
    # copy when the import itself has failed.
    target_tree = None
    if editable_dir:
        target_tree = os.path.dirname(os.path.abspath(editable_dir))
        lines.append((OK, f"editable install points at : {editable_dir}"))
    elif import_origin is None:
        lines.append((WARN, "no GroundingDINO editable install found in site-packages"))

    lines.append((OK if config_dir else WARN,
                  f"GROUNDING_DINO_DIR resolves to: {config_dir or 'unresolved'}"))

    # Two independent copies: the classic trap. A junction/symlink is fine.
    if len(trees) > 1 and not _same_tree(trees[0]["tree"], trees[1]["tree"]):
        lines.append((WARN, "two separate GroundingDINO trees exist on this machine:"))
        for t in trees:
            lines.append((WARN, f"    {t['tree']}"))
        lines.append((WARN, "    Only the \"autoannotate study\" (space) copy is tracked by git,"))
        lines.append((WARN, "    so a git pull refreshes that one and leaves the other to rot."))
    elif len(trees) > 1:
        lines.append((OK, "both folder spellings resolve to the same tree (linked, not copied)"))

    # The specific failure seen in the field: the tree Python imports from kept
    # its compiled extension but lost its sources.
    for t in trees:
        is_origin = origin_tree and _same_tree(t["tree"], origin_tree)
        is_target = target_tree and _same_tree(t["tree"], target_tree)
        if not t["missing"]:
            continue
        tag = FAIL if (is_origin or is_target or import_origin is None) else WARN
        if tag == FAIL:
            fatal = True
        if is_origin:
            who = "the tree Python imports from"
        elif is_target:
            who = "the tree the editable install points at"
        elif import_origin is None:
            # The import failed and nothing identifies a target, so this copy
            # cannot be dismissed as unused.
            who = "possibly the one Python was meant to import"
        else:
            who = "an unused copy"
        lines.append((tag, f"{t['tree']} ({who}) is missing "
                           f"{len(t['missing'])} required source file(s):"))
        for rel in t["missing"]:
            lines.append((tag, f"    groundingdino/{rel}"))
        if t["compiled"]:
            lines.append((tag, "    Compiled artifacts are still present, so this is a "
                               "partially-synced copy,"))
            lines.append((tag, "    not a missing install. The build does NOT need redoing."))

    # Code loaded from one tree while configs and weights are read from another.
    if origin_tree and config_dir and not _same_tree(origin_tree, config_dir):
        lines.append((WARN, "the imported code and GROUNDING_DINO_DIR are DIFFERENT trees:"))
        lines.append((WARN, f"    code    : {origin_tree}"))
        lines.append((WARN, f"    configs : {config_dir}"))
        lines.append((WARN, "    Model configs and weights are read from the second while the"))
        lines.append((WARN, "    code comes from the first, so the two can drift apart."))

    if len(trees) > 1 or fatal or any(tag == WARN for tag, _ in lines):
        # Which tree holds the untracked, expensive artifacts decides which
        # remedy is safe, so state it before giving instructions.
        lines.append(("", ""))
        lines.append(("", "  What each copy holds (these are gitignored and live in ONE tree only):"))
        for t in trees:
            bits = []
            bits.append("compiled _C extension" if t["ext"] else "no _C extension")
            bits.append(f"{len(t['weights'])} weight file(s)" if t["weights"]
                        else "no weights")
            lines.append(("", f"    {t['tree']}"))
            lines.append(("", f"        {', '.join(bits)}"))

        keeper = next((t for t in trees if t["ext"] or t["weights"]), None)
        lines.append(("", ""))
        lines.append(("", "  How to fix:"))
        if keeper and tracked and not _same_tree(keeper["tree"], tracked["tree"]):
            # The dangerous case. Repointing pip at the tracked copy first would
            # abandon a CUDA build that can take a long time to reproduce.
            lines.append(("", "    Do NOT simply reinstall against the tracked copy first: the"))
            lines.append(("", "    build and the weights above are NOT in git and would be left"))
            lines.append(("", "    behind. Move them across before switching."))
            lines.append(("", ""))
            lines.append(("", "    1. Copy the _C extension and the weights into the tracked copy:"))
            lines.append(("", f"         from: {keeper['tree']}"))
            lines.append(("", f"         to  : {tracked['tree']}"))
            lines.append(("", "         (groundingdino/_C.*  and  weights/*.pth)"))
            lines.append(("", "    2. Reinstall the editable package against the tracked copy:"))
            lines.append(("", '         pip install --no-build-isolation -e '
                              '"autoannotate study/GroundingDINO"'))
            lines.append(("", "    3. Rename the other copy aside and re-run this check. Git now"))
            lines.append(("", "       maintains the only tree that matters."))
            lines.append(("", ""))
            lines.append(("", "    Alternative, if you would rather not move anything: make the two"))
            lines.append(("", "    spellings the SAME directory, so they can never diverge again."))
            lines.append(("", "    Rename the untracked copy away first, then (Administrator):"))
            lines.append(("", '         mklink /J "autoannotate_study" "autoannotate study"'))
            lines.append(("", "    (macOS/Linux: ln -s 'autoannotate study' autoannotate_study)"))
        elif tracked:
            lines.append(("", "    1. Make sure the editable install points at the tracked copy:"))
            lines.append(("", '         pip install --no-build-isolation -e '
                              '"autoannotate study/GroundingDINO"'))
            lines.append(("", "    2. Or link the two spellings so they cannot diverge again"))
            lines.append(("", "       (Administrator, after renaming the untracked copy away):"))
            lines.append(("", '         mklink /J "autoannotate_study" "autoannotate study"'))
            lines.append(("", "       (macOS/Linux: ln -s 'autoannotate study' autoannotate_study)"))
            lines.append(("", "    3. Quick patch: copy the missing .py files across, leaving the"))
            lines.append(("", "       compiled _C extension alone. Goes stale on the next git pull."))
    return lines, fatal


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

    # The application imports the DINO wrapper during normal GUI startup, so
    # GroundingDINO is required even if the user plans to select another
    # detector. It is vendored in this repository and must be installed from
    # that local tree; suggesting a similarly named PyPI package would be wrong.
    _gd_origin = None
    _gd_error = None
    try:
        _gd = importlib.import_module("groundingdino")
        _gd_origin = os.path.dirname(os.path.abspath(_gd.__file__))
        print(f"  {OK} {'groundingdino':<14} present (local editable install)")
    except Exception as e:
        _gd_error = f"{e.__class__.__name__}: {e}"
        print(f"  {FAIL} {'groundingdino':<14} missing")
        print('       Install with: pip install --no-build-isolation -e '
              '"autoannotate study/GroundingDINO"')
        print(f"       Import error: {_gd_error}")
        _mark_fail()

    header("Python packages (optional)")
    for mod, note in [("diffusers", "Stable Diffusion variations")]:
        try:
            importlib.import_module(mod)
            print(f"  {OK} {mod:<14} present")
        except Exception:
            print(f"  {WARN} {mod:<14} not found -> {note}")

    # ── GroundingDINO install integrity ───────────────────────────────────────
    # Mirror config.py's resolution: env value first, then either study folder
    # spelling. Computed once here and reused by the weight checks below.
    gd_candidates = []
    if os.environ.get("GROUNDING_DINO_DIR"):
        gd_candidates.append(os.environ["GROUNDING_DINO_DIR"])
    for study in ("autoannotate study", "autoannotate_study"):
        gd_candidates.append(os.path.join(REPO_ROOT, study, "GroundingDINO"))
    gd_dir = next((c for c in gd_candidates if os.path.isdir(c)), gd_candidates[-2])

    header("GroundingDINO install integrity")
    gd_lines, gd_fatal = diagnose_groundingdino(
        REPO_ROOT, _gd_origin, _gd_error, gd_dir if os.path.isdir(gd_dir) else None,
        editable_dir=editable_target())
    for tag, text in gd_lines:
        print(f"  {tag} {text}" if tag else text)
    if gd_fatal:
        _mark_fail()

    # ── per-user settings + image I/O ─────────────────────────────────────────
    # The per-user settings dir has to be creatable and writable, and every image
    # read/write goes through numpy so paths with non-ASCII characters work on
    # Windows (OpenCV's own imread/imwrite use the ANSI API there and fail
    # silently). Both are exercised here rather than trusted.
    header("Per-user settings and image I/O")
    try:
        sys.path.insert(0, REPO_ROOT)
        from autoannotate.config import USER_CONFIG_DIR, user_config_path
        probe = user_config_path(".write_probe")
        with open(probe, "w", encoding="utf-8", newline="\n") as f:
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
        print(f"  PyTorch build  : {torch.__version__}")
        print(f"  Built for CUDA : {torch.version.cuda or 'no (CPU-only build)'}")
        print(f"  CUDA available : {cuda}")
        print(f"  MPS available  : {mps}")
        if cuda:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                total_gb = props.total_memory / (1024 ** 3)
                print(f"  CUDA device {index}  : {props.name} ({total_gb:.1f} GB)")

        try:
            import groundingdino._C  # noqa: F401
            print(f"  GroundingDINO _C extension : {OK} importable")
        except Exception as e:
            print(f"  GroundingDINO _C extension : {WARN} unavailable "
                  f"({e.__class__.__name__}: {e})")
            if cuda:
                print("       Rebuild GroundingDINO after installing CUDA PyTorch; "
                      "DINO will otherwise be much slower.")

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
    # gd_dir was resolved with config.py's rules in the integrity section above.
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

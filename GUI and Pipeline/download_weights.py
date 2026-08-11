#!/usr/bin/env python3
"""
download_weights.py - fetch the model weights AutoAnnotate needs.

The weights are deliberately not in the repository (they total roughly 6 GB and
.gitignore excludes *.pt / *.pth). Until now every new machine acquired them by
hand from HOW_TO_RUN.txt STEP 3, which is the slowest and most error-prone part
of setting the project up on a second computer.

Usage:
    python "GUI and Pipeline/download_weights.py"            # everything required
    python "GUI and Pipeline/download_weights.py" --all      # required + optional
    python "GUI and Pipeline/download_weights.py" --list     # show status, download nothing
    python "GUI and Pipeline/download_weights.py" --only yoloe-11l-seg.pt
    python "GUI and Pipeline/download_weights.py" --force    # re-fetch even if present

Everything except sam3.pt comes from public GitHub release assets over plain
stdlib urllib, so this script adds no dependency of its own.

sam3.pt is different: facebook/sam3 on Hugging Face is a gated repository, so
you have to accept its licence with your own account first. Pass a token with
--hf-token or set HF_TOKEN, and this script will fetch it through
huggingface_hub. Without a token it prints the manual instructions and carries
on with the rest. Note that the token is only ever needed for this one-time
download; the application itself runs token-free.

Exit code 0 = every requested weight is present and the right size.
Exit code 1 = something required is still missing.
"""
import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

OK   = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Ultralytics publishes these as immutable release assets, so the byte size is a
# usable integrity check. Every size below was confirmed against the live URL and
# against a known-good local copy on 2026-08-10.
ULTRALYTICS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
DINO_BASE = "https://github.com/IDEA-Research/GroundingDINO/releases/download"


def _grounding_dino_weights_dir():
    """Where the DINO checkpoints live, mirroring check_environment.py.

    The study folder is tracked as "autoannotate study" but file transfers
    sometimes mangle the space into an underscore, and GROUNDING_DINO_DIR can
    override the location outright.
    """
    env = os.environ.get("GROUNDING_DINO_DIR")
    if env:
        return os.path.join(env, "weights")
    for study in ("autoannotate study", "autoannotate_study"):
        candidate = os.path.join(REPO_ROOT, study, "GroundingDINO")
        if os.path.isdir(candidate):
            return os.path.join(candidate, "weights")
    return os.path.join(REPO_ROOT, "autoannotate study", "GroundingDINO", "weights")


class Weight:
    def __init__(self, name, url, dest_dir, size, required, note, gated=False,
                 hf_repo=None):
        self.name = name
        self.url = url
        self.dest_dir = dest_dir
        self.size = size            # expected bytes, or None when not pinned
        self.required = required
        self.note = note
        self.gated = gated
        self.hf_repo = hf_repo

    @property
    def path(self):
        return os.path.join(self.dest_dir, self.name)


def build_catalog():
    dino_dir = _grounding_dino_weights_dir()
    return [
        Weight("yoloe-11l-seg.pt", f"{ULTRALYTICS_BASE}/yoloe-11l-seg.pt",
               HERE, 70982416, True,
               "YOLOE detector (YOLOE-vis and YOLOE-seg one-shot)"),
        Weight("yoloe-11l-seg-pf.pt", f"{ULTRALYTICS_BASE}/yoloe-11l-seg-pf.pt",
               HERE, 74264838, True,
               "YOLOE prompt-free dependency, loaded alongside the detector"),
        # Ultralytics fetches this itself the first time a YOLOE text prompt
        # runs, which is a 600 MB surprise in the middle of someone's first
        # annotation. Getting it here makes the first run offline-safe.
        Weight("mobileclip_blt.ts", f"{ULTRALYTICS_BASE}/mobileclip_blt.ts",
               HERE, 599764649, True,
               "MobileCLIP text encoder, needed for YOLOE text prompts"),
        Weight("sam2_t.pt", f"{ULTRALYTICS_BASE}/sam2_t.pt",
               HERE, 78064050, True,
               "SAM2 tiny segmenter"),
        Weight("groundingdino_swint_ogc.pth",
               f"{DINO_BASE}/v0.1.0-alpha/groundingdino_swint_ogc.pth",
               dino_dir, 693997677, True,
               "GroundingDINO SwinT, the default text-prompt detector"),
        Weight("groundingdino_swinb_cogcoor.pth",
               f"{DINO_BASE}/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
               dino_dir, 938057991, False,
               "GroundingDINO SwinB, heavier and more accurate than SwinT"),
        # No size is pinned for sam3.pt: it is a branch file on Hugging Face
        # rather than a frozen release asset, so upstream can legitimately
        # replace it and a hard-coded size would start reporting false damage.
        Weight("sam3.pt", None, HERE, None, False,
               "SAM3 detector and segmenter (gated, needs a Hugging Face token)",
               gated=True, hf_repo="facebook/sam3"),
    ]


def human(n):
    if n is None:
        return "unknown size"
    return f"{n / (1024 * 1024):.0f} MB"


def exact(n):
    """MB plus the raw byte count.

    Used wherever two sizes are being compared. Rounded MB alone renders a
    999-byte shortfall as "5 MB, expected 5 MB", which hides the very
    discrepancy the message exists to report.
    """
    if n is None:
        return "unknown size"
    return f"{n / (1024 * 1024):.0f} MB ({n} bytes)"


def local_state(w):
    """Return (present, size_on_disk, looks_complete)."""
    if not os.path.exists(w.path):
        return False, 0, False
    size = os.path.getsize(w.path)
    if w.size is None:
        # Nothing to compare against, so anything above the "failed download or
        # HTML error page" threshold counts as real.
        return True, size, size >= 1024 * 1024
    return True, size, size == w.size


def _progress(done, total, name):
    if not sys.stdout.isatty() or not total:
        return
    pct = done * 100 // total
    sys.stdout.write(f"\r       {name}: {pct:3d}%  ({human(done)} / {human(total)})")
    sys.stdout.flush()


def download_url(w):
    """Stream w.url to w.path via a temp file. Returns True on success.

    The download lands on a temp file in the destination directory and is moved
    into place only once it is complete, so an interrupted run can never leave a
    half-written checkpoint that later looks present to check_environment.py.
    """
    os.makedirs(w.dest_dir, exist_ok=True)
    req = urllib.request.Request(w.url, headers={"User-Agent": "AutoAnnotate-setup"})
    tmp_path = None
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            if w.size and total and total != w.size:
                print(f"  {WARN} {w.name}: server offers {exact(total)}, expected "
                      f"{exact(w.size)}. Downloading anyway; verify before trusting it.")
            fd, tmp_path = tempfile.mkstemp(prefix=f".{w.name}.", suffix=".part",
                                            dir=w.dest_dir)
            done = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    _progress(done, total, w.name)
        if sys.stdout.isatty() and total:
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()
        if w.size and os.path.getsize(tmp_path) != w.size:
            got = os.path.getsize(tmp_path)
            print(f"  {FAIL} {w.name}: downloaded {exact(got)}, expected {exact(w.size)}. "
                  f"Not installing the partial file.")
            os.remove(tmp_path)
            return False
        shutil.move(tmp_path, w.path)
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        print(f"  {FAIL} {w.name}: {e.__class__.__name__}: {e}")
        return False


def download_gated(w, token):
    """Fetch a gated Hugging Face file through huggingface_hub.

    Deliberately not done with urllib: huggingface.co redirects to a CDN that
    rejects the request when the Authorization header is forwarded along with
    its own signed URL, so a hand-rolled download works right up until the
    redirect and then fails confusingly.
    """
    if not token:
        print(f"  {WARN} {w.name}: no Hugging Face token, skipping.")
        print(f"         1. Accept the licence at https://huggingface.co/{w.hf_repo}")
        print( "         2. Create a read token at https://huggingface.co/settings/tokens")
        print( "         3. Re-run with --hf-token <token>, or set HF_TOKEN")
        print(f"         Or download {w.name} by hand and put it in {w.dest_dir}")
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"  {FAIL} {w.name}: huggingface_hub is not installed "
              f"(pip install huggingface_hub), or download the file by hand.")
        return False
    try:
        print(f"       fetching {w.name} from {w.hf_repo} (about 3.2 GB, this is slow)")
        cached = hf_hub_download(repo_id=w.hf_repo, filename=w.name, token=token)
    except Exception as e:
        print(f"  {FAIL} {w.name}: {e.__class__.__name__}: {e}")
        print(f"         A 401 or 403 here means the token has not been granted access "
              f"to {w.hf_repo} yet.")
        return False
    os.makedirs(w.dest_dir, exist_ok=True)
    # Copy rather than symlink out of the HF cache: the app resolves this path
    # directly, and a cache clean would otherwise break a working install.
    shutil.copyfile(cached, w.path)
    return True


def report(catalog):
    print("\nWeight status")
    print("-" * 13)
    missing_required = 0
    for w in catalog:
        present, size, complete = local_state(w)
        tag = "required" if w.required else "optional"
        if present and complete:
            print(f"  {OK} {w.name:<34} {human(size):>9}  {tag}")
        elif present:
            print(f"  {WARN} {w.name:<34} {human(size):>9}  {tag}, expected "
                  f"{exact(w.size)}, got {size} bytes; re-run with --force")
            if w.required:
                missing_required += 1
        else:
            print(f"  {WARN} {w.name:<34} {'missing':>9}  {tag}: {w.note}")
            if w.required:
                missing_required += 1
    return missing_required


def main(argv=None):
    catalog = build_catalog()
    names = [w.name for w in catalog]

    p = argparse.ArgumentParser(
        description="Download the model weights AutoAnnotate needs.")
    p.add_argument("--all", action="store_true",
                   help="include the optional weights (SwinB, and SAM3 if a token is set)")
    p.add_argument("--only", metavar="NAME", action="append", choices=names,
                   help="download just this weight (repeatable). Choices: " + ", ".join(names))
    p.add_argument("--force", action="store_true",
                   help="re-download even when the file is already present and complete")
    p.add_argument("--list", action="store_true",
                   help="report what is present and exit without downloading")
    p.add_argument("--hf-token", metavar="TOKEN", default=None,
                   help="Hugging Face token for the gated sam3.pt (or set HF_TOKEN)")
    args = p.parse_args(argv)

    print("=" * 60)
    print(" AutoAnnotate weight download")
    print("=" * 60)
    print(f"  Repo root      : {REPO_ROOT}")
    print(f"  Ultralytics ->   {HERE}")
    print(f"  GroundingDINO -> {_grounding_dino_weights_dir()}")

    if args.list:
        report(catalog)
        print("\n  Nothing downloaded (--list).")
        return 0

    token = args.hf_token or os.environ.get("HF_TOKEN") or None

    if args.only:
        wanted = [w for w in catalog if w.name in set(args.only)]
    elif args.all:
        wanted = list(catalog)
    else:
        wanted = [w for w in catalog if w.required]
        # A token that is already set is a clear signal the user wants SAM3, so
        # asking them to add --all as well would just be a second hoop.
        if token:
            wanted += [w for w in catalog if w.gated]

    skipped = [w for w in catalog if w not in wanted]
    if skipped:
        print("\n  Not requested this run: " + ", ".join(w.name for w in skipped))

    print("\nDownloading")
    print("-" * 11)
    failures = []
    for w in wanted:
        present, size, complete = local_state(w)
        if present and complete and not args.force:
            print(f"  {OK} {w.name:<34} already present ({human(size)})")
            continue
        if present and not complete and not args.force:
            print(f"  {WARN} {w.name:<34} present but {size} bytes, expected "
                  f"{exact(w.size)}. Re-run with --force to replace it.")
            failures.append(w)
            continue
        if w.gated:
            ok = download_gated(w, token)
        else:
            print(f"  ....  {w.name:<34} downloading {human(w.size)}")
            ok = download_url(w)
        if ok:
            print(f"  {OK} {w.name:<34} done ({human(os.path.getsize(w.path))})")
        else:
            failures.append(w)

    missing_required = report(catalog)

    print("\nResult")
    print("-" * 6)
    if missing_required:
        print(f"  {FAIL} {missing_required} required weight(s) still missing or incomplete.")
        print( "       See HOW_TO_RUN.txt STEP 3 to fetch them by hand.")
        return 1
    if failures:
        print(f"  {WARN} Every required weight is in place; optional ones were skipped "
              f"or failed: {', '.join(w.name for w in failures)}")
    else:
        print(f"  {OK} Every requested weight is in place.")
    print( "       Next: python \"GUI and Pipeline/check_environment.py\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())

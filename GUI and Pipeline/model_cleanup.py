#!/usr/bin/env python3
"""
model_cleanup.py - find and remove model files AutoAnnotate no longer uses.

Two stores eat disk:
  1. The HuggingFace hub cache (~/.cache/huggingface/hub) - the big "blobs".
  2. Local ultralytics / GroundingDINO weight files (*.pt / *.pth).

This script classifies everything it finds as KEPT (referenced by AutoAnnotate)
or ORPHAN (not referenced), shows the reclaimable space, and - only when you
ask - deletes the orphans. HF deletions go through the official cache API so the
blobs/snapshots/refs structure is handled correctly (never rm a blob by hand).

Usage:
    python model_cleanup.py            # report only (safe, default)
    python model_cleanup.py --delete   # interactively confirm each orphan
    python model_cleanup.py --delete --yes   # delete all orphans, no prompts

The KEEP sets below are the source of truth. Add a model id / filename here if
the tool wrongly flags something you still use.
"""
import argparse
import glob
import os
import sys

# --- What AutoAnnotate actually uses (edit if the project's models change) ---
KEEP_HF = {
    "HuggingFaceTB/SmolVLM-256M-Instruct",                 # prompt-suggestion VLM
    "stable-diffusion-v1-5/stable-diffusion-inpainting",   # synthetic-image SD
    "bert-base-uncased",                                   # GroundingDINO text encoder
    "facebook/sam3",                                       # SAM3 (if HF-cached)
}
KEEP_LOCAL = {
    "sam2_t.pt", "sam2_b.pt", "sam3.pt",
    "yoloe-11l-seg.pt", "yoloe-11l-seg-pf.pt",             # -pf is a YOLOE dependency
    "groundingdino_swint_ogc.pth", "groundingdino_swinb_cogcoor.pth",
}

HERE = os.path.dirname(os.path.abspath(__file__))


def _gb(n):
    return f"{n / 1e9:6.2f} GB"


def _grounding_dino_weights_dir():
    """Read GROUNDING_DINO_DIR from env or the repo .env, return its weights dir."""
    d = os.environ.get("GROUNDING_DINO_DIR")
    if not d:
        env = os.path.join(os.path.dirname(HERE), ".env")
        if os.path.isfile(env):
            for line in open(env):
                if line.startswith("GROUNDING_DINO_DIR="):
                    d = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return os.path.join(d, "weights") if d else None


def scan_hf():
    """Return (cache_obj, [(repo, size, kept, [rev_hashes]), ...]) or (None, [])."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print("  (huggingface_hub not installed - skipping HF cache scan)")
        return None, []
    try:
        cache = scan_cache_dir()
    except Exception as e:
        print(f"  (HF cache scan failed: {e})")
        return None, []
    rows = []
    for r in cache.repos:
        rows.append((r.repo_id, r.size_on_disk, r.repo_id in KEEP_HF,
                     [rev.commit_hash for rev in r.revisions]))
    rows.sort(key=lambda x: -x[1])
    return cache, rows


def scan_local():
    """Return [(path, size, kept), ...] for *.pt / *.pth in model dirs."""
    dirs = [HERE]
    wd = _grounding_dino_weights_dir()
    if wd and os.path.isdir(wd):
        dirs.append(wd)
    rows, seen = [], set()
    for d in dirs:
        for pat in ("*.pt", "*.pth"):
            for f in glob.glob(os.path.join(d, pat)):
                rp = os.path.realpath(f)
                if rp in seen:
                    continue
                seen.add(rp)
                rows.append((f, os.path.getsize(f), os.path.basename(f) in KEEP_LOCAL))
    rows.sort(key=lambda x: -x[1])
    return rows


def report(hf_rows, local_rows):
    print("\n=== HuggingFace hub cache ===")
    recl_hf = 0
    for repo, size, kept, _ in hf_rows:
        tag = "KEEP  " if kept else "ORPHAN"
        if not kept:
            recl_hf += size
        print(f"  [{tag}] {_gb(size)}  {repo}")
    if not hf_rows:
        print("  (nothing cached)")

    print("\n=== Local weight files (*.pt / *.pth) ===")
    recl_local = 0
    for path, size, kept in local_rows:
        tag = "KEEP  " if kept else "ORPHAN"
        if not kept:
            recl_local += size
        print(f"  [{tag}] {_gb(size)}  {path}")
    if not local_rows:
        print("  (none found)")

    print("\n=== Reclaimable ===")
    print(f"  HF cache orphans : {_gb(recl_hf)}")
    print(f"  Local orphans    : {_gb(recl_local)}")
    print(f"  TOTAL            : {_gb(recl_hf + recl_local)}")
    return recl_hf + recl_local


def confirm(prompt, auto_yes):
    if auto_yes:
        return True
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def do_delete(cache, hf_rows, local_rows, auto_yes):
    # HF orphans via the official cache API.
    hf_orphans = [(repo, size, hashes) for repo, size, kept, hashes in hf_rows if not kept]
    if cache is not None and hf_orphans:
        to_purge = []
        for repo, size, hashes in hf_orphans:
            if confirm(f"Delete HF cache repo {repo} ({_gb(size).strip()})?", auto_yes):
                to_purge.extend(hashes)
        if to_purge:
            strategy = cache.delete_revisions(*to_purge)
            print(f"  freeing {_gb(strategy.expected_freed_size)} from HF cache...")
            strategy.execute()
            print("  done.")

    # Local orphans via os.remove.
    for path, size, kept in local_rows:
        if kept:
            continue
        if confirm(f"Delete local file {path} ({_gb(size).strip()})?", auto_yes):
            os.remove(path)
            print(f"  removed {path}")


def main():
    ap = argparse.ArgumentParser(description="Find/remove unused AutoAnnotate model files.")
    ap.add_argument("--delete", action="store_true", help="delete orphans (interactive)")
    ap.add_argument("--yes", action="store_true", help="skip per-item confirmation")
    args = ap.parse_args()

    cache, hf_rows = scan_hf()
    local_rows = scan_local()
    reclaimable = report(hf_rows, local_rows)

    if not args.delete:
        print("\nReport only. Re-run with --delete to remove the orphans above.")
        return
    if reclaimable == 0:
        print("\nNothing to delete - everything on disk is in use.")
        return
    print()
    do_delete(cache, hf_rows, local_rows, args.yes)


if __name__ == "__main__":
    sys.exit(main())

"""Stable Diffusion background-variation pipeline (synthetic training images)."""
import os

import cv2
import numpy as np
import torch

# Stable Diffusion synthetic image generation
# Ports the inpainting pipeline from
#   stable-diffusion practice with Grounding DINO.ipynb
# into a function the GUI can call. The pipeline regenerates the
# BACKGROUND of an image while preserving the user-annotated objects
# pixel-for-pixel, so the resulting variation inherits the source's
# labels unchanged and can be used as a synthetic training sample.
#
# Pipeline choice: stable-diffusion-v1-5/stable-diffusion-inpainting
# (~5GB, public). Was originally going to use
# stabilityai/stable-diffusion-2-inpainting but that repo is gated --
# you'd need to manually accept terms on HF to download it. The 1.5
# inpainting model produces comparable results for the
# "preserve-foreground, regenerate-background" pattern this pipeline
# uses, and avoids the auth friction.
#
# Variation count per source image is fixed at 1. The notebook the
# pipeline was ported from also does 1 per image, and the obvious next
# extension if the augmentation pool needs to grow is an "N variations
# per source" spinner in the GUI, a simple loop wrapper around
# generate_variation that re-rolls with a fresh seed each pass. Left
# as a knob to add later rather than baked in here so the per-image
# UX (single side-by-side preview) doesn't have to grow into a grid.

_sd_inpaint_pipe   = None
_sd_inpaint_device = None

# Inference resolution for the SD-1.5 inpainting U-Net. SD-1.5 was
# trained at 512x512 -- that's the "native" resolution and produces the
# best quality. On an 8GB MPS Mac the 512x512 working set thrashes
# virtual memory and per-image generation balloons past 15 minutes; at
# 256x256 the U-Net does ~1/4 the work per step and stays inside the
# MPS budget, bringing generation back to ~30-60s. Output quality is
# noticeably softer than 512 -- but the user's annotated foreground is
# composited back on top at full resolution, so only the regenerated
# BACKGROUND is affected. Bump back to 512 if/when this code runs on
# bigger hardware.
# Inference resolution: SD-1.5 was trained at 512x512 and produces
# noticeably better output at native res. We had this at 256 while
# fighting MPS OOMs; on CPU the memory constraint is gone, so we bump
# back to 512 for quality. Generation is slower (~4-8 min/image on M2
# CPU) but the output is dramatically less artifacted.
_SD_INPAINT_RES = 512

# Default negative prompt: SD-1.5 has a strong tendency to hallucinate
# text, illustrations, and diagrams when given a vague positive prompt
# (the original notebook used SDXL refiner where this wasn't needed;
# SD-1.5 inpainting NEEDS negatives to behave). Override per-call by
# passing negative_prompt= into generate_variation.
_SD_DEFAULT_NEGATIVE = (
    "text, letters, numbers, watermark, illustration, cartoon, diagram, "
    "chart, table, spreadsheet, blurry, low quality, low resolution, "
    "distorted, deformed, artifacts, jpeg artifacts, "
    "bowl, plate, pile of berries, picked berries, loose berries, "
    "fruit close-up, studio, plain background, solid color background, "
    "isolated object, indoor"
)

# Inpainting strength: fraction of the masked (background) region SD
# regenerates. 1.0 = full regeneration from noise -- it DISCARDS the
# original scene and repaints the background purely from the prompt,
# which tends to give flat/unnatural fills (a solid-color backdrop or a
# pile of loose berries) and makes the preserved foreground look pasted
# on. Lower values keep the REAL background from the source image and
# only lightly vary it, so the preserved berries sit naturally in an
# actual field. 0.4-0.7 is the sweet spot for realistic variations
# (lower = more realistic but less diverse). Override per-call via
# generate_variation(strength=...).
_SD_STRENGTH = 0.2


def _sd_select_device():
    """Pick the compute device for SD inpainting, honoring the
    AUTOANNOTATE_SD_DEVICE env var when set (cell 0 sets it to 'cpu'
    on this 8GB M2 because MPS hung inference for 15+ min across
    multiple optimization attempts). Without the override, prefers
    CUDA > MPS > CPU. fp16 only on CUDA -- MPS fp16 has precision
    artifacts on diffusion U-Nets, CPU doesn't benefit."""
    override = os.environ.get("AUTOANNOTATE_SD_DEVICE", "").lower().strip()
    if override == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if override == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    if override == "cpu":
        return torch.device("cpu"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def _sd_release_resident_models(extra_caches=None):
    """Aggressively release memory before loading Stable Diffusion.

    Before this fix the SAM3 'unload' only dropped Python references;
    PyTorch's MPS caching allocator kept the underlying tensors in a
    private pool, so re-running SD on top of an earlier SAM3 / YOLOE /
    DINO session would stack ~5GB of dead-but-cached model weights on
    top of the ~5GB SD load, and the system would swap-thrash trying to
    bring SD's weights in from disk. This function:

      1. Drops the module-level SAM3 predictor reference.
      2. Drops references in any extra per-window caches (e.g. the
         ManualWindow's `_model_cache` holding YOLOE / DINO / SAM2).
      3. Runs gc.collect() so Python actually finalizes the objects.
      4. Calls torch.mps.empty_cache() (and CUDA equivalent) to return
         the now-unused memory to the OS.

    Call this from any code path that's about to load a memory-heavy
    model (SD inpainting, future similar workloads).
    """
    import gc
    from . import sam as _sam
    print("[SD] releasing resident detection/segmentation models...")
    _sam.release_sam3_text_predictor()
    if extra_caches:
        for cache in extra_caches:
            if isinstance(cache, dict):
                cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    print("[SD] release complete.")


# Backwards-compat shim: existing callers use this name.


def load_sd_inpaint(device=None, extra_caches=None):
    """Load (or return cached) Stable Diffusion inpainting pipeline.

    First call is slow (model download + init on first run, ~5GB
    download); subsequent calls reuse the cached instance.

    `extra_caches` is an optional list of dict-shaped caches (e.g. the
    ManualWindow's `_model_cache`) to clear before loading SD. Pass
    them in so MPS / CUDA actually frees their memory instead of
    keeping them resident alongside SD."""
    global _sd_inpaint_pipe, _sd_inpaint_device
    if _sd_inpaint_pipe is not None:
        return _sd_inpaint_pipe
    _sd_release_resident_models(extra_caches=extra_caches)
    try:
        from diffusers import StableDiffusionInpaintPipeline
    except ImportError as e:
        # Surface the REAL exception text -- this fires not just when
        # diffusers is missing, but also when a sub-dependency
        # (safetensors, accelerate, transformers version mismatch) is
        # broken. Hiding the underlying cause behind a generic "install
        # diffusers" message sends the user down the wrong rabbit hole.
        raise RuntimeError(
            f"Could not import diffusers.StableDiffusionInpaintPipeline.\n"
            f"Underlying error: {type(e).__name__}: {e}\n\n"
            f"If this is `No module named 'diffusers'`, install with:\n"
            f"    pip install diffusers>=0.27 safetensors\n"
            f"into the SAME Python that runs the app. The most reliable "
            f"way is with the venv's own interpreter, from the repo root:\n"
            f"    .venv/bin/python -m pip install diffusers safetensors\n"
            f"(.venv\\Scripts\\python on Windows), then relaunch the app."
        ) from e
    if device is None:
        device, dtype = _sd_select_device()
    else:
        device = torch.device(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Repo choice: stable-diffusion-v1-5/stable-diffusion-inpainting is
    # the community-hosted home of SD 1.5 inpainting (publicly accessible,
    # no license click-through needed). stabilityai/stable-diffusion-2-inpainting
    # is a GATED repo -- the diffusers download surfaces this as a confusing
    # "Repository Not Found" 404. If you want SD-2 specifically: visit
    # https://huggingface.co/stabilityai/stable-diffusion-2-inpainting,
    # click "Agree and access repository", then swap the repo ID below.
    model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    print(f"[SD] loading {model_id} on {device} ({dtype})...")
    # variant="fp16" picks the *.fp16.safetensors files (the only
    # safetensors variant this repo ships). torch_dtype controls the
    # IN-MEMORY dtype independently -- on MPS / CPU we still want fp32
    # for numerical stability, and diffusers upcasts from the fp16 file
    # on load.
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    # Cut the progress-bar spam in the terminal output.
    pipe.set_progress_bar_config(disable=True)

    # Memory savings for MPS / 8GB systems
    # Inference on an 8GB Mac OOMs in the U-Net attention layer at
    # 512x512 without these. Each knob has a small (2x at worst) speed
    # cost but no visible quality impact for the
    # preserve-foreground-regenerate-background use case.
    #
    # 1. Drop the safety checker -- a ~600MB CLIP classifier for NSFW
    #    filtering. We composite the user's annotated foreground back
    #    on top of the SD output before save, so the model's own
    #    pre-filter has no useful gate to apply.
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    # 2. Slice attention -- computes Q@K in smaller chunks instead of
    #    one giant tensor. Biggest single memory win on MPS.
    # Attention slicing on MPS: slice_size=1 (max aggression) was
    # catastrophically slow -- every attention head turned into its
    # own kernel launch, and MPS launch overhead dominated runtime
    # (single image took 15+ minutes). At 256x256 the U-Net working
    # set is small enough that we don't need slicing at all on M-series
    # silicon. Leave the call OUT; bring it back as "auto" only if
    # bumping back up to 512x512 OOMs again.
    pass
    # 3. Slice + tile the VAE pass -- avoids a big spike during the
    #    final image decode at the configured resolution.
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    _sd_inpaint_pipe   = pipe
    _sd_inpaint_device = device
    return pipe


def generate_variation(image_path, boxes_xyxy=None, polys_xyxy_pixel=None,
                       prompt="a blueberry bush growing in an outdoor field, green leaves, branches and stems, soil, natural daylight, photorealistic, sharp focus, depth of field",
                       negative_prompt=None, strength=None):
    """Generate one synthetic variation of `image_path`:
       - background is re-inpainted by Stable Diffusion under `prompt`,
       - regions inside the preserve mask are pasted back from the
         original so labels stay pixel-accurate on the variation.

    Preserve mask is the UNION of rectangles AND polygons. Polygons
    are filled at pixel accuracy (via cv2.fillPoly) so segmentation
    annotations preserve their actual shape -- not just the bounding
    box, which would over-preserve background and produce visible
    rectangular cutouts in the regenerated image.

    Args:
      image_path: source image on disk.
      boxes_xyxy: list of [x1, y1, x2, y2] in IMAGE-PIXEL coords for
        RECTANGULAR preserve regions (typical for bbox-only annotation).
      polys_xyxy_pixel: list of polygons, each [[x, y], ...] in IMAGE-
        PIXEL coords for POLYGON-shaped preserve regions (typical for
        segmentation-mode annotation). Must have >= 3 points to be
        used.
      prompt / negative_prompt: SD text prompts driving the background
        regeneration. negative_prompt defaults to _SD_DEFAULT_NEGATIVE
        when None.

    At least one of boxes_xyxy / polys_xyxy_pixel must contain valid
    regions; otherwise the entire image would be inpainted and the
    annotated objects would not be preserved at all.

    Returns:
      (variation_PIL, original_PIL). Both at the original image's size.
    """
    from PIL import Image, ImageChops
    pipe = load_sd_inpaint(extra_caches=getattr(generate_variation, "_extra_caches", None))
    original = Image.open(image_path).convert("RGB")
    iw, ih = original.size
    # Preserve-mask: white inside annotated regions. SD's mask convention
    # is "white = inpaint here", so we INVERT this before passing it.
    preserve = np.zeros((ih, iw), dtype=np.uint8)
    # Rectangles: fast path via numpy slicing.
    for x1, y1, x2, y2 in (boxes_xyxy or []):
        x1i = max(0, min(iw, int(x1))); x2i = max(0, min(iw, int(x2)))
        y1i = max(0, min(ih, int(y1))); y2i = max(0, min(ih, int(y2)))
        if x2i > x1i and y2i > y1i:
            preserve[y1i:y2i, x1i:x2i] = 255
    # Polygons: cv2.fillPoly draws the actual shape, so segmentation
    # annotations don't inflate to their bbox.
    for poly in (polys_xyxy_pixel or []):
        if not poly or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(preserve, [pts], 255)
    if not preserve.any():
        # Enforce the contract in this function's own docstring. Nothing survived
        # the clipping (every box degenerate or off-image, every poly too short),
        # so the preserve mask is empty. The mask is INVERTED below, so an empty
        # preserve becomes an all-white "inpaint here" mask and SD would repaint
        # the entire image straight over the annotated objects it was called to
        # protect. Refuse instead of destroying them.
        raise ValueError(
            "generate_variation: no valid regions to preserve (all boxes/polys "
            "were empty or degenerate); refusing to inpaint the whole image.")
    preserve_pil = Image.fromarray(preserve)
    # SD-1.5 inpaint takes its trained resolution as input. We use
    # _SD_INPAINT_RES (see module-level constant + comment) to make
    # this tunable without hunting through the function.
    res = _SD_INPAINT_RES
    image_inpaint = original.resize((res, res))
    mask_inpaint  = preserve_pil.resize((res, res))
    inpaint_mask  = ImageChops.invert(mask_inpaint.convert("L"))
    # Aggressive speed cuts for 8GB MPS: 20 steps instead of the 50-step
    # default, plus the DPM-Solver++ scheduler which converges in fewer
    # steps without quality loss for this kind of background-inpaint
    # task. Without these an 8GB Mac swaps to disk during inference
    # and a single image can take 30+ minutes.
    try:
        from diffusers import DPMSolverMultistepScheduler
        if not isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception as _e:
        print(f"[SD] could not switch scheduler ({_e}); using default")
    # Per-step progress log so the user can see inference moving
    # instead of staring at "Loading pipeline components... 100%" with
    # no idea whether it's hung. Each line prints the step number and
    # wall-clock time since the previous step.
    import time as _t
    _step_state = {"last": _t.time(), "start": _t.time()}
    def _step_cb(pipe_, step, timestep, callback_kwargs):
        now = _t.time()
        per_step = now - _step_state["last"]
        elapsed = now - _step_state["start"]
        _step_state["last"] = now
        print(f"[SD] step {step+1}/30  ({per_step:.1f}s, elapsed {elapsed:.1f}s)")
        return callback_kwargs
    # Bumped to 30 steps from 20 for sharper output now that resolution
    # is back to 512x512. Negative prompt is the big quality lever for
    # SD-1.5 inpainting -- without it the model hallucinates text /
    # diagrams when filling regenerated regions.
    neg = negative_prompt if negative_prompt is not None else _SD_DEFAULT_NEGATIVE
    strength_val = strength if strength is not None else _SD_STRENGTH
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            image=image_inpaint,
            mask_image=inpaint_mask,
            num_inference_steps=30,
            guidance_scale=8.5,
            strength=strength_val,
            callback_on_step_end=_step_cb,
        ).images[0]
    except TypeError:
        # Older diffusers don't support callback_on_step_end. Fall back
        # to no-callback (silent during inference but still works).
        result = pipe(
            prompt=prompt,
            negative_prompt=neg,
            image=image_inpaint,
            mask_image=inpaint_mask,
            num_inference_steps=30,
            guidance_scale=8.5,
            strength=strength_val,
        ).images[0]
    # Back to original resolution, then paste the preserved boxes on top.
    result_full = result.resize((iw, ih))
    variation = Image.composite(original, result_full, preserve_pil)
    return variation, original

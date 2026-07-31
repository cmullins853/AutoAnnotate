# Auto-Annotate System

### Project Description:
The Auto-Annotate system leverages state-of-the-art models such as **Grounding DINO**, **YOLO**, and **Segment Anything Model (SAM)** for automated detection and segmentation tasks. This system aims to streamline image annotation processes for precision agriculture and other applications by combining deep learning models with optimization techniques.

Key functionalities include:
- Automatic bounding box generation using Grounding DINO.
- Mask segmentation using SAM.
- Confidence tuning and prompt optimization to refine annotation performance.
- Metrics evaluation for model performance.

---

## Getting Started

### Prerequisites:
- Install the Python libraries. There is no single cross-platform file; pick the
  one for your machine:
  - macOS / Apple Silicon: `pip install -r requirements-macos.lock`
  - Windows 10, CPU only: `pip install -r requirements-windows10-cpu.txt`
  - Windows 11 + NVIDIA (CUDA): `pip install -r requirements-windows11-cuda.txt`
  - Any OS, newest resolutions (advanced): `pip install -r requirements.txt`
```bash
# macOS example
pip install -r requirements-macos.lock
```
- **Software Dependencies:**
  - Grounding DINO configuration and weight files.
  - Segment Anything (SAM) weight files.

### Installation:
The authoritative, step-by-step setup guide is
[`GUI and Pipeline/HOW_TO_RUN.txt`](GUI%20and%20Pipeline/HOW_TO_RUN.txt). It covers
the virtual environment, the `.env` file, where to put the model weight files, and
how to launch the app on macOS, Linux, and Windows. In short:

0. On a fresh machine, install the prerequisites first: Git and Python 3.13;
   Visual Studio Build Tools with the C++ workload on every Windows machine;
   and, on Windows/Linux + NVIDIA, the GPU driver and CUDA Toolkit compatible
   with the `+cuXXX` torch build. STEP 0 of HOW_TO_RUN.txt lists each one with
   its download source.
1. Clone the repository and create a virtual environment with Python 3.13.
2. Install dependencies with the file for your platform:
   `requirements-macos.lock` (macOS/MPS, exact `pip freeze`; macOS only),
   `requirements-windows10-cpu.txt` (Windows 10 CPU-only, exact proven pins),
   `requirements-windows11-cuda.txt` (Windows 11 + NVIDIA CUDA), or
   `requirements.txt` (generic floors: Linux, or any OS).
3. Install GroundingDINO as a package (the `--no-build-isolation` flag is
   **required** because its `setup.py` imports torch at build time, so torch from
   step 2 must already be installed):
   `pip install --no-build-isolation -e "autoannotate study/GroundingDINO"`
4. Place the weight files (DINO `.pth`, `sam2_t.pt`, `sam3.pt`, `yoloe-*.pt`) as
   described in HOW_TO_RUN.txt. `GROUNDING_DINO_DIR` is auto-derived, so a fresh
   clone needs no `.env` path edit.
5. Launch the app from a terminal at the repo root with `python run_app.py`
   (or `python -m autoannotate`). No Jupyter or IDE is required.

### Windows 10 without a GPU:

Use `requirements-windows10-cpu.txt`. It preserves the tested
`torch==2.6.0+cpu` / `torchvision==0.21.0+cpu` pair and uses PyTorch's CPU wheel
index. GroundingDINO is intentionally not embedded as a `pip freeze` VCS line;
install the vendored copy separately so the space in `autoannotate study` cannot
be misparsed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows10-cpu.txt
python -m pip install wheel setuptools ninja
python -m pip install --no-build-isolation -e "autoannotate study/GroundingDINO"
python "GUI and Pipeline/check_environment.py"
python run_app.py
```

GroundingDINO compiles a CPU C++ extension, so install Visual Studio Build
Tools with the "Desktop development with C++" workload first. No NVIDIA driver
or CUDA Toolkit is needed on this machine.

This machine will be slow. Prefer DINO SwinT and SAM2 tiny; SwinB is a
232.9-million-parameter detector and is materially heavier on CPU.

### Windows with an NVIDIA GPU (CUDA):
Windows setup is fully documented in HOW_TO_RUN.txt (STEP 1 for the PowerShell
venv and the CUDA torch install, STEP 4 for the GroundingDINO build). The short
version, validated on Windows 11 with an RTX 4060:

1. Create the venv with `py -3.13 -m venv .venv` and activate with
   `.venv\Scripts\Activate.ps1` (run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once first).
2. Use `requirements-windows11-cuda.txt`: it pins the CUDA torch/torchvision wheels
   (`+cuXXX`) and adds the PyTorch CUDA index, so pip installs the GPU build
   instead of the CPU one. Clear any CPU torch first with `pip uninstall -y
   torch torchvision torchaudio`. Use another CUDA wheel only when PyTorch
   publishes a compatible torch/torchvision pair for that CUDA version, and
   build GroundingDINO with a matching CUDA Toolkit. Verify with
   `torch.cuda.is_available()`; update the NVIDIA driver if it prints `False`.
3. Building GroundingDINO needs Visual Studio Build Tools (C++ workload) and
   `pip install wheel setuptools ninja`. The vendored `setup.py` adds the MSVC
   `/Zc:preprocessor` flag automatically for CUDA 12.4+/13 toolkits.
4. `transformers` must stay below 5.x (pinned in `requirements-windows11-cuda.txt`);
   transformers 5 breaks GroundingDINO's BERT wrapper.

The app runs on macOS, Windows, and Linux. The only per-OS setting is the Stable
Diffusion compute device (`AUTOANNOTATE_SD_DEVICE`), configured in a commented
per-OS block in `autoannotate/config.py`; see HOW_TO_RUN.txt ("SUPPORTED OPERATING
SYSTEMS"). Optional speed/memory tuning (model RAM budget, batch chunking,
memory-release cadence, etc.) is controlled by `AUTOANNOTATE_*` environment
variables, all documented under "TUNING KNOBS" in HOW_TO_RUN.txt, with defaults
that reproduce the original behavior.

Auto Annotate Remaining retries an empty image once at a lower threshold, then
copies any still-empty or failed images into `output/_review/{boxes,segments}/`
with a `review_report.csv` so you can find and fix them. See HOW_TO_RUN.txt
STEP 6.

A Hugging Face token is optional and the running app never uses it: every model
it loads is public or already on disk, and no `from_pretrained` call passes a
token. A token is only useful *outside* the app, to authenticate the one-time
manual download of the gated `sam3.pt` from `huggingface.co/facebook/sam3`. Once
that file is in place, nothing reads the token. See HOW_TO_RUN.txt for details.

---

## Upkeep and Contributing

- Keep all core functionalities in modular Python files, separate from other experimental or testing scripts.
- Commit frequently with well-documented messages. Follow the [commit message guidelines](#commit-message-guidelines) provided below.
- Update the `requirements.txt` file only after validating that no dependency conflicts arise from the updates.

---

## What Makes a Good Commit? [[1]](#1)

Use the following format for commit messages: `[category(what): why]`.

Examples:
- `feature(pipeline optimization): Added parallel processing for SAM-based segmentation.`
- `fix(GroundingDINO accuracy): Adjusted bounding box threshold for improved detection.`
- `docs(readme): Updated usage instructions for model configuration.`

### Categories:
- **Feature**: Adding new functionality.
- **Fix**: Resolving issues or bugs.
- **Refactor**: Improving code structure without altering functionality.
- **Test**: Adding or updating test cases.
- **Documentation**: Updating or adding documentation.
- **Style**: Modifying code formatting without functionality changes.
- **Chore**: Updating build tools or dependencies.

---

## Key Files and Usage

All live code is in the `autoannotate/` package at the repo root. Model weight
files and dev tools live under `GUI and Pipeline/`.

### Code Files:
- **`autoannotate/`**:
  - The application package. `config.py` holds env/path/device setup,
    `pipeline/` holds the model wrappers (DINO, YOLOE, SAM2/SAM3, Stable
    Diffusion) and label I/O, `gui/` holds the PyQt5 windows and canvas, and
    `optimizer.py` holds the prompt/confidence optimizers. Entry points:
    `run_app.py` at the repo root, or `python -m autoannotate`.
- **[GUI and Pipeline/auto-annotate-backend.py](GUI%20and%20Pipeline/auto-annotate-backend.py)**:
  - Legacy reference only. NOT imported by the GUI. The `autoannotate` package
    contains the corrected versions of these functions; this file has
    non-portable paths and will not run as-is. Kept for history.
- **[GUI and Pipeline/manual-tuning-test.ipynb](GUI%20and%20Pipeline/manual-tuning-test.ipynb)**:
  - Scratch notebook for manual testing and tuning of prompts and confidence levels.
- **[GUI and Pipeline/LLM implementation.ipynb](GUI%20and%20Pipeline/LLM%20implementation.ipynb)**:
  - Experiments for the LLM prompt-suggestion path used by the Automated window.
- **[GUI and Pipeline/test_semiauto_headless.py](GUI%20and%20Pipeline/test_semiauto_headless.py)**:
  - Headless regression tests for the semi-automatic segmentation GUI logic.
    Run with `QT_QPA_PLATFORM=offscreen .venv/bin/python "GUI and Pipeline/test_semiauto_headless.py"`.
- **[GUI and Pipeline/model_cleanup.py](GUI%20and%20Pipeline/model_cleanup.py)**:
  - Utility that reports and (on request) removes model weights AutoAnnotate no
    longer uses, from both the Hugging Face cache and local `.pt`/`.pth` files.
- **[GUI and Pipeline/check_environment.py](GUI%20and%20Pipeline/check_environment.py)**:
  - Cross-OS readiness check. Loads no models; verifies the Python deps, reports
    the compute device the app will use, and lists which weight files are
    present. Run it (inside the venv) on a new machine before launching:
    `python "GUI and Pipeline/check_environment.py"`.

### Instructions for Model Training and Testing:
#### Grounding DINO:
1. The config and checkpoint paths are derived from `GROUNDING_DINO_DIR` in
   `autoannotate/config.py`; the weight files live in
   `autoannotate study/GroundingDINO/weights/`.
2. Adjust `box_threshold` and `text_threshold` for specific datasets.

#### Segment Anything Model (SAM):
1. Place `sam2_t.pt` (and `sam3.pt` if using SAM3) in `GUI and Pipeline/`. SAM2
   auto-downloads from ultralytics on first run; SAM3 must be fetched manually.
2. The segmenter is selected from the GUI dropdown, not by editing a path.

---

## Editing Guidelines

### Adding New Models:
- Ensure compatibility with the pipeline.
- Add the model in the `autoannotate` package (NOT the legacy
  `auto-annotate-backend.py`, which is unused):
  - `autoannotate/pipeline/`: add the loader/inference helper in the matching
    module (mirroring `run_yoloe_*`, `run_sam3_text`, etc.).
  - `autoannotate/gui/manual_window.py`: wire it into `_get_model`,
    `_detector_keys_for_pipeline`, and the `_run_detector` dispatch, and
    classify its prompt capability (text/boxes) so the Text/Boxes radios and
    carry-forward gate it correctly.

### Training New Models:
- Update the model paths in notebooks (e.g., YOLO or Grounding DINO).
- Use pre-defined metrics for evaluation.

### Supported Metrics:
- Intersection-over-Union (IoU).
- Precision, Recall, and F1 Score.
- Pixel Accuracy.

---

## References

- Tian, Y., Zhang, Y., Stol, K.-J., Jiang, L., & Liu, H. (2022, May). *What makes a good commit message?*. Proceedings of the 44th International Conference on Software Engineering. doi:10.1145/3510003.3510205.

---

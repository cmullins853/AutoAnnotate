# AutoAnnotate Windows GPU Setup and Mac-to-Windows Adaptation Handoff

## Purpose

AutoAnnotate was originally developed and tested on a Windows machine, but a significant portion of the recent cleanup, feature improvements, and portability work was completed on macOS. The goal of this Windows setup work was to bring those newer macOS-developed changes back onto a Windows machine with an NVIDIA GPU, verify that the full application still works in that environment, and prepare it for large-scale image annotation on a research desktop.

This handoff explains what broke during the Windows setup, why the older setup steps were not enough, what had to be installed or changed, and what should be done before porting the app to the research desktop.

The important final result is that AutoAnnotate was successfully configured on Windows with CUDA acceleration. GroundingDINO, SAM2, and Stable Diffusion were verified to run on the NVIDIA GPU during inference.

---

## High-Level Summary

The Windows setup was not just a normal dependency install. The main challenge was adapting the recently improved macOS-tested version of the project back onto a Windows + NVIDIA CUDA environment.

The setup required:

* Installing a CUDA-enabled PyTorch build before the normal requirements.
* Reinstalling or updating NVIDIA GPU drivers so PyTorch could detect CUDA.
* Installing Visual Studio Build Tools with the C++ workload.
* Installing Python build helpers such as `wheel`, `setuptools`, and `ninja`.
* Building GroundingDINO as an editable package with `--no-build-isolation`.
* Fixing a GroundingDINO CUDA source compatibility issue with modern PyTorch.
* Keeping `transformers` below version 5 because Transformers 5 breaks GroundingDINO.
* Registering CUDA DLL directories on Windows so the compiled GroundingDINO extension can load.
* Making the project tolerate both `autoannotate study` and `autoannotate_study` folder names.
* Placing model weight files in the exact folders expected by the app.
* Verifying the environment with `check_environment.py` before launching the GUI.

The root issue was dependency and platform mismatch. The newer version of the project worked on macOS, but returning it to Windows exposed CUDA-specific, compiler-specific, dependency-specific, and path-specific problems that were not visible during the recent Mac development work.

---

## Final Validated Windows Environment

The setup was validated on the following machine:

* Operating system: Windows 11 Home
* GPU: NVIDIA GeForce RTX 4060 Laptop GPU
* Python: 3.13.5
* Virtual environment: project-level virtual environment
* CUDA Toolkit: 13.2
* PyTorch: CUDA-enabled build
* `torch`: `2.12.1+cu132`
* `torchvision`: `0.27.1+cu132`
* Visual Studio Build Tools: installed with C++ workload
* MSVC: 14.51
* `transformers`: `4.50.3`
* `ultralytics`: `8.4.90`
* GroundingDINO: editable install with compiled CUDA extension
* DINO SwinT weight: `groundingdino_swint_ogc.pth`
* SAM2 weight: `sam2_t.pt`

Important note: the compiled GroundingDINO extension is tied to the active Python, PyTorch, CUDA, and compiler combination. If the research desktop uses a different Python version, PyTorch CUDA build, CUDA major version, or GPU architecture, GroundingDINO should be rebuilt on that machine.

---

## Why the Recent macOS Work Did Not Directly Transfer Back to Windows

Although AutoAnnotate originally came from a Windows development environment, much of the recent cleanup and improvement work was completed on macOS. On macOS, the app can run without the NVIDIA CUDA stack. Depending on the machine and configuration, models may run on CPU or Apple MPS.

On Windows with an NVIDIA GPU, the setup is more sensitive because:

* PyTorch must be installed from the correct CUDA wheel index.
* The NVIDIA driver must be compatible with the installed CUDA/PyTorch build.
* GroundingDINO includes compiled extension code.
* GroundingDINO’s extension depends on CUDA runtime DLLs.
* Windows handles DLL loading differently from macOS and Linux.
* Modern CUDA versions require specific MSVC compiler settings.
* Modern PyTorch versions exposed outdated code inside GroundingDINO’s CUDA source.
* Loose dependency requirements allowed newer packages that were incompatible with GroundingDINO.

So the issue was not that AutoAnnotate had never run on Windows before. The issue was that the newer version, after recent macOS-based improvements, had to be made reliable again on a fresh Windows machine with CUDA.

---

## Working Windows Setup Process

### Step 0: Machine Prerequisites

The steps below assume the machine already has: Python 3.13 (python.org
installer with the `py` launcher option ticked), Git, an up-to-date NVIDIA
driver, the CUDA Toolkit matching the torch `+cuXXX` pins (13.2 on the
validation machine; the toolkit provides `nvcc`, which is required to build
GroundingDINO and does not come with the driver), and Visual Studio Build
Tools with the "Desktop development with C++" workload. On a machine with no
prior development work, install those first; STEP 0 of
`GUI and Pipeline/HOW_TO_RUN.txt` lists each with its download source.

### Step 1: Create and Activate the Virtual Environment

From the repository root:

```powershell
py -3.13 -m venv .venv
```

If using PowerShell, allow local activation scripts once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If using Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

Make sure the active virtual environment is the one inside the repository root. During debugging, there was confusion because an incomplete stray virtual environment existed in a parent folder. The research desktop should avoid multiple similar virtual environments.

---

### Step 2: Install CUDA PyTorch Before the Other Requirements

This was one of the most important Windows-specific fixes.

If PyTorch is installed through the normal `requirements.txt` path first, pip may install the CPU-only build. If that happens, the app can silently run models on CPU even though the machine has an NVIDIA GPU.

For Windows + NVIDIA GPU, install CUDA-enabled PyTorch first:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

The `cu132` part should match the CUDA/PyTorch combination supported by the machine. On another research desktop, this may need to be adjusted.

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Expected result:

```text
True
13.2
NVIDIA GeForce RTX 4060 Laptop GPU
```

If `torch.cuda.is_available()` prints `False`, fix this before continuing. During the Windows setup, reinstalling the NVIDIA driver was required before CUDA was properly detected.

If a CPU-only PyTorch build was already installed, remove it first:

```powershell
pip uninstall -y torch torchvision
```

Then reinstall the CUDA-enabled build.

---

### Step 3: Install the Project Requirements

Install the project dependencies from the Windows requirements file. It pins the
CUDA torch/torchvision wheels (+cu132) and adds the PyTorch CUDA index, so it also
covers the torch install from Step 2 (clear any CPU build first):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-windows11-cuda.txt
```

There is no one-size-fits-all requirements file. `requirements-windows11-cuda.txt` is the
Windows + NVIDIA file (pinned CUDA torch). `requirements-macos.lock` is the macOS
lock (CPU/MPS torch) and must NOT be used on Windows, or pip installs the CPU
build. `requirements.txt` only pins floors and is for advanced users who want
the newest resolutions.

For a fully reproducible lock on this exact desktop, after a verified install:

```powershell
pip freeze > requirements-windows11-cuda.freeze.txt
```

---

### Step 4: Keep Transformers Below Version 5

One major failure was caused by `transformers` resolving to version 5.x.

GroundingDINO depends on a BERT method called `get_head_mask`. That method exists in Transformers 4.x but was removed in Transformers 5.x.

When Transformers 5 was installed, GroundingDINO crashed during model loading with an error similar to:

```text
AttributeError: 'BertModel' object has no attribute 'get_head_mask'
```

The fix was:

```powershell
pip install "transformers==4.50.3"
```

Recommended project-level requirement:

```text
transformers>=4.40,<5
```

This should be pinned in `requirements.txt` so fresh installs do not accidentally install Transformers 5 and break GroundingDINO.

---

### Step 5: Install Windows Build Tools

GroundingDINO is not a pure Python dependency. It includes compiled extension code, and on Windows with CUDA it needs Microsoft compiler tools.

Install Visual Studio Build Tools with:

* Desktop development with C++ workload
* MSVC C++ toolchain
* Windows SDK

Then install Python build helpers inside the active virtual environment:

```powershell
pip install wheel setuptools ninja
```

One observed failure was:

```text
invalid command 'bdist_wheel'
```

This was fixed by installing `wheel` and `setuptools`.

`ninja` was also required on the validation machine for the extension build process.

---

### Step 6: Patch the GroundingDINO CUDA Source for Modern PyTorch

GroundingDINO failed to build against the modern PyTorch version with an error similar to:

```text
no suitable conversion function from "at::DeprecatedTypeProperties" to "c10::ScalarType"
```

The cause was an outdated call in GroundingDINO’s CUDA source.

In the GroundingDINO file:

```text
autoannotate_study/GroundingDINO/.../csrc/MsDeformAttn/ms_deform_attn_cuda.cu
```

the code needed this change on lines 65 and 135:

```cpp
value.type()
```

changed to:

```cpp
value.scalar_type()
```

This is required because modern PyTorch no longer accepts the old implicit conversion used by `value.type()` in this context.

This should be treated as a real source compatibility fix, not just a local workaround. It should be committed into the vendored GroundingDINO code so future Windows/CUDA machines can build successfully.

---

### Step 7: Build and Install GroundingDINO

GroundingDINO must be installed as an editable Python package so imports such as the following work:

```python
from groundingdino.util.inference import ...
```

The install command is:

```powershell
pip install --no-build-isolation -e "autoannotate study/GroundingDINO"
```

If the folder is named with an underscore instead:

```powershell
pip install --no-build-isolation -e "autoannotate_study/GroundingDINO"
```

The `--no-build-isolation` flag is required because GroundingDINO’s `setup.py` imports `torch` during build time. Without this flag, pip builds in an isolated temporary environment where torch is not available, causing the install to fail.

For CUDA 12.4 or CUDA 13, the Windows compiler may also need this environment variable before building:

```powershell
$env:CL = "/Zc:preprocessor"
```

Then run the editable install:

```powershell
pip install --no-build-isolation -e "autoannotate_study/GroundingDINO"
```

After the build:

```powershell
Remove-Item Env:\CL
```

The newer project setup should ideally add this compiler flag automatically in `setup.py`, but if not, the manual environment variable is required.

---

### Step 8: Fix the GroundingDINO Folder Name Issue

The original code expected this folder:

```text
autoannotate study/GroundingDINO
```

However, on the Windows checkout, the folder appeared as:

```text
autoannotate_study/GroundingDINO
```

This caused a startup crash:

```text
GroundingDINO not found at ...\autoannotate study\GroundingDINO
```

Attempting to rename the folder on Windows failed with:

```text
Access is denied
```

The likely cause was Windows Search Indexer or Windows Defender locking large files inside the folder, such as model weights or compiled binaries.

The robust fix was to make the code accept both folder names.

Recommended logic for `autoannotate/config.py`:

```python
GROUNDING_DINO_DIR = os.environ.get("GROUNDING_DINO_DIR")

if not GROUNDING_DINO_DIR:
    for _study in ("autoannotate study", "autoannotate_study"):
        _cand = os.path.join(REPO_ROOT, _study, "GroundingDINO")
        if os.path.isdir(_cand):
            GROUNDING_DINO_DIR = _cand
            break
    else:
        GROUNDING_DINO_DIR = os.path.join(
            REPO_ROOT,
            "autoannotate study",
            "GroundingDINO",
        )
```

The same dual-name logic should also exist in:

```text
GUI and Pipeline/check_environment.py
```

This prevents the environment checker from falsely reporting GroundingDINO as missing.

Long-term recommendation: avoid spaces in important source folders. A folder name like `autoannotate_study` is easier to use in scripts, shells, and cross-platform transfers.

---

### Step 9: Register CUDA DLL Directories on Windows

After GroundingDINO compiled, importing the compiled extension still failed with a DLL error:

```text
ImportError: DLL load failed while importing _C
```

The cause is Windows-specific. Since Python 3.8, compiled `.pyd` extension modules do not reliably search the normal `PATH` for dependent DLLs. CUDA may be on `PATH`, but Python extension loading may still not find the CUDA runtime DLLs.

The fix was to register the CUDA binary folder explicitly using `os.add_dll_directory()`.

Recommended logic for `autoannotate/config.py`:

```python
if _platform.system() == "Windows":
    _cuda_root = os.environ.get("CUDA_PATH")

    if _cuda_root:
        for _sub in ("bin", os.path.join("bin", "x64")):
            _dll_dir = os.path.join(_cuda_root, _sub)

            if os.path.isdir(_dll_dir):
                try:
                    os.add_dll_directory(_dll_dir)
                except (OSError, AttributeError):
                    pass
```

This is safe because it only runs on Windows and only when `CUDA_PATH` exists.

Without this fix, DINO may still appear to run, but it can silently fall back to slower pure-Python deformable attention instead of using the compiled CUDA extension.

---

## Model Weight Placement

The app needs several large model weight files that are not stored in git.

### GroundingDINO SwinT

Required file:

```text
groundingdino_swint_ogc.pth
```

Expected location:

```text
autoannotate_study/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

or:

```text
autoannotate study/GroundingDINO/weights/groundingdino_swint_ogc.pth
```

PowerShell example:

```powershell
New-Item -ItemType Directory -Force "autoannotate_study\GroundingDINO\weights" | Out-Null

curl.exe -L `
  -o "autoannotate_study\GroundingDINO\weights\groundingdino_swint_ogc.pth" `
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
```

### GroundingDINO SwinB

SwinB did not work initially because the SwinB weight file was missing.

Required file:

```text
groundingdino_swinb_cogcoor.pth
```

This file is only needed if using the DINO SwinB detector. Until it is present, the SwinB detector option should remain greyed out.

### SAM2

Required file:

```text
sam2_t.pt
```

Expected location:

```text
GUI and Pipeline/sam2_t.pt
```

Ultralytics may auto-download `sam2_t.pt` on first run, but it may place the file in the current working directory instead of `GUI and Pipeline/`.

If that happens, move it manually:

```powershell
Move-Item .\sam2_t.pt ".\GUI and Pipeline\sam2_t.pt"
```

### SAM3

Required file:

```text
sam3.pt
```

Expected location:

```text
GUI and Pipeline/sam3.pt
```

SAM3 is gated on Hugging Face and must be downloaded manually after accepting the license.

The app can run without SAM3, but SAM3 options should remain greyed out until the file is present.

### YOLOE

Expected files:

```text
GUI and Pipeline/yoloe-11l-seg.pt
GUI and Pipeline/yoloe-11l-seg-pf.pt
```

These can auto-download from Ultralytics on first use, but for the research desktop it is better to place them ahead of time so the setup does not depend on a first-run network download.

---

## `.env` and `AUTOANNOTATE_MAX_AREA_FRAC`

During setup, VS Code showed this warning:

```text
An environment file is configured but terminal environment injection is disabled.
Enable "python.terminal.useEnvFile" to use environment variables from .env files in terminals.
```

This warning is about VS Code’s terminal environment injection. It does not necessarily mean the app itself cannot read `.env`.

The intended project behaviour is that the app loads `.env` itself through `python-dotenv`, so the VS Code warning should be unrelated if the app loads the `.env` file inside `autoannotate/config.py`.

However, since `AUTOANNOTATE_MAX_AREA_FRAC` appeared not to apply, the likely causes are:

* The `.env` file was not in the repo root expected by `config.py`.
* The app was launched from a different checkout or working directory.
* The wrong virtual environment was active.
* The variable was edited after the Python process had already started.
* The `.env` format was invalid.
* The code read the variable before `load_dotenv()` ran.
* Some pipeline file used a hardcoded value instead of importing the central config value.

The robust fix is to make `autoannotate/config.py` the single source of truth. The project should use hardcoded safe defaults in Python, while still allowing `.env` or shell overrides.

Recommended pattern:

```python
# autoannotate/config.py

DEFAULT_MAX_AREA_FRAC = 0.5

def _read_float_env(name: str, default: float, min_value: float, max_value: float) -> float:
    raw = os.environ.get(name)

    if raw is None or raw.strip() == "":
        return default

    try:
        value = float(raw)
    except ValueError:
        return default

    return max(min_value, min(max_value, value))

AUTOANNOTATE_MAX_AREA_FRAC = _read_float_env(
    "AUTOANNOTATE_MAX_AREA_FRAC",
    DEFAULT_MAX_AREA_FRAC,
    0.0,
    1.0,
)
```

Detector code should then import the config value:

```python
from autoannotate.config import AUTOANNOTATE_MAX_AREA_FRAC
```

Recommended `.env` format:

```text
AUTOANNOTATE_MAX_AREA_FRAC=0.5
AUTOANNOTATE_SD_DEVICE=cuda
AUTOANNOTATE_DEBUG=1
```

No quotes are needed.

To verify the value:

```powershell
python "GUI and Pipeline\check_environment.py"
```

The checker should print the effective `AUTOANNOTATE_MAX_AREA_FRAC`. If it does not, the checker should be updated to report it clearly.

---

## Confirming CUDA Was Actually Used

A confusing part of debugging was that some models appeared to load on CPU. This did not mean inference was CPU-only.

The final finding was:

* GroundingDINO may rest on CPU after loading, but moves to CUDA during prediction.
* SAM2 may rest on CPU after loading, but runs segmentation on CUDA during inference.
* Stable Diffusion selects CUDA through the device configuration and runs with CUDA/fp16 when available.

The key lesson is that checking the model’s device immediately after load can be misleading. The important check is where the model runs during actual inference.

Basic CUDA verification:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```text
True
NVIDIA GeForce RTX 4060 Laptop GPU
```

For deeper validation, run a small end-to-end inference and confirm that DINO and SAM2 execute on `cuda:0`.

---

## Verification Steps Before Using the Research Desktop

Before running large annotation batches on the research desktop, verify the environment in this order.

### 1. Verify the Active Python Environment

```powershell
where python
python --version
pip --version
```

Confirm the Python executable is inside the project’s virtual environment.

### 2. Verify PyTorch CUDA

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

Expected result:

```text
True
<CUDA version>
<NVIDIA GPU name>
```

### 3. Verify Project Environment

```powershell
python "GUI and Pipeline\check_environment.py"
```

This should confirm:

* Python dependencies are installed.
* CUDA is available.
* GroundingDINO can be found.
* Required model weights are present.
* `AUTOANNOTATE_MAX_AREA_FRAC` is being read correctly.
* Stable Diffusion device selection is correct.

### 4. Run Headless Regression Tests

From the repo root:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python "GUI and Pipeline\test_semiauto_headless.py"
```

The validation machine passed the full headless test suite.

### 5. Launch the GUI

```powershell
python run_app.py
```

or:

```powershell
python -m autoannotate
```

The app should launch without requiring Jupyter, VS Code, or an IDE.

---

## Functional Improvements Identified During the Windows Adaptation

The Windows setup work also revealed workflow issues in AutoAnnotate itself. Several improvements were identified or added so the app is more useful for real annotation work.

### Multi-Class Prompts

Instead of running the model separately for each object class, the app supports comma-separated class prompts.

Example:

```text
blueberry, leaf, stem
```

Each class is detected in a single model pass. Class IDs are preserved in the saved labels, and a `class_colors.txt` table is written so the labels remain interpretable.

This matters because the research desktop may process large image folders. Running one combined pass is more efficient than running the same model repeatedly for each class.

### Negative Prompts

The app supports negative classes.

Example:

```text
Positive: blueberry
Negative: leaf
```

This allows unwanted detections to be filtered during the same detection pass. It helps reduce false positives and cleanup work.

### Prompt/Class Dropdown

A class dropdown was added so hand-drawn annotations can be assigned to the correct class. This is important when mixing automatic detections with manual corrections.

### Include Earlier Images / Recycle Option

Previously, if the user started “Auto Annotate Remaining” halfway through a folder, earlier images could be skipped.

The include-earlier or recycle option allows earlier images to be appended to the end of the batch. This is useful when the user tunes prompts and thresholds in the middle of a folder but still wants the entire folder processed.

### Previous Image Without Rerunning the Model

A “Previous Image” workflow was needed because going back should not rerun the model.

If a user had already trimmed or corrected annotations, rerunning the model could overwrite that manual work.

The safer behaviour is:

* Save the current image’s annotations.
* Move back to the previous image.
* Reload the saved labels exactly as edited.
* Do not run inference again.

This preserves manual corrections and avoids repeated cleanup.

### Review Folder for Failed or Empty Images

For batch processing, images with no detections or errors are copied into a review folder instead of being silently ignored.

Expected structure:

```text
output/_review/boxes/
output/_review/segments/
output/_review/review_report.csv
```

This is important for research use because failed images need to be auditable. The user should know which images need manual review and why.

---

## Root Causes Identified

### Root Cause 1: CPU PyTorch Can Be Installed Accidentally

On Windows, installing PyTorch through the normal requirements path can result in a CPU-only build. This makes the app run much slower and defeats the purpose of using a GPU workstation.

Fix:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

---

### Root Cause 2: NVIDIA Driver/CUDA Mismatch

Even with CUDA installed, PyTorch may report:

```text
False
```

for:

```python
torch.cuda.is_available()
```

Fix:

* Update or reinstall the NVIDIA driver.
* Reinstall the matching CUDA-enabled PyTorch wheel.
* Verify CUDA before installing the rest of the project.

---

### Root Cause 3: GroundingDINO Requires Compiled Extension Setup

GroundingDINO is not a simple pure-Python dependency. It requires build tools, PyTorch availability during setup, and compatibility with the local CUDA/compiler stack.

Fix:

* Install Visual Studio Build Tools.
* Install CUDA-enabled PyTorch first.
* Install `wheel`, `setuptools`, and `ninja`.
* Build GroundingDINO with `--no-build-isolation`.

---

### Root Cause 4: CUDA 13 Requires a Newer MSVC Preprocessor Mode

CUDA 12.4 and CUDA 13 headers require MSVC’s conforming preprocessor. Without it, GroundingDINO can fail to build.

Fix:

```powershell
$env:CL = "/Zc:preprocessor"
```

or patch `setup.py` to add the flag automatically on Windows.

---

### Root Cause 5: GroundingDINO CUDA Source Was Outdated for Modern PyTorch

The vendored CUDA source used `value.type()`, which fails with newer PyTorch.

Fix:

```cpp
value.type()
```

should become:

```cpp
value.scalar_type()
```

---

### Root Cause 6: Python 3.8+ Windows DLL Loading Behaviour

Even after compiling, `groundingdino._C` failed because Python did not find CUDA DLLs through `PATH`.

Fix:

Use `os.add_dll_directory()` for `CUDA_PATH\bin` on Windows.

---

### Root Cause 7: Folder Naming Was Fragile

The project expected:

```text
autoannotate study
```

but the Windows checkout had:

```text
autoannotate_study
```

Fix:

Support both folder names in `autoannotate/config.py` and `GUI and Pipeline/check_environment.py`.

---

### Root Cause 8: Dependency Drift From Loose Requirements

The requirements file allowed Transformers 5, which broke GroundingDINO.

Fix:

Use `requirements-windows11-cuda.txt` (which pins `transformers==4.50.3`) or pin:

```text
transformers>=4.40,<5
```

---

### Root Cause 9: Model Weights Were Not Always in the Expected Location

SAM2 downloaded to the working directory instead of `GUI and Pipeline/`.

Fix:

Move model weights into the canonical folders and make the environment checker report missing weights clearly.

---

### Root Cause 10: `.env` Behaviour Was Unclear

The VS Code warning made it seem like `.env` was not loaded. The app should load `.env` itself, but the effective value still needs to be printed and verified.

Fix:

* Centralize config values in `autoannotate/config.py`.
* Provide hardcoded safe defaults.
* Allow `.env` overrides.
* Print effective config values in `check_environment.py`.

---

## Recommended Research Desktop Setup Process

Use this process for the research desktop.

```powershell
# 1. Clone the repository
git clone <repo-url>
cd AutoAnnotate

# 2. Create and activate the virtual environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install all requirements (requirements-windows11-cuda.txt pins the CUDA torch
#    wheels and adds the PyTorch CUDA index, so it installs the GPU torch too)
pip uninstall -y torch torchvision torchaudio
pip install -r requirements-windows11-cuda.txt

# 4. Verify CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"

# 6. Install build helpers
pip install wheel setuptools ninja

# 7. Build GroundingDINO
$env:CL = "/Zc:preprocessor"
pip install --no-build-isolation -e "autoannotate_study/GroundingDINO"
Remove-Item Env:\CL

# 8. Place model weights in the documented folders
# DINO weights -> autoannotate_study/GroundingDINO/weights/
# SAM2/SAM3/YOLOE weights -> GUI and Pipeline/

# 9. Verify environment
python "GUI and Pipeline\check_environment.py"

# 10. Run tests
$env:QT_QPA_PLATFORM = "offscreen"
python "GUI and Pipeline\test_semiauto_headless.py"

# 11. Run the app
python run_app.py
```

Adjust the PyTorch CUDA wheel index if the research desktop requires a different CUDA build.

---

## Project Changes That Should Be Upstreamed

The following changes should be committed so the research desktop setup is repeatable.

### Required Changes

1. Pin Transformers below 5 in `requirements.txt`.

```text
transformers>=4.40,<5
```

2. Keep the GroundingDINO CUDA source patch.

```cpp
value.type() -> value.scalar_type()
```

3. Keep dual folder-name resolution.

```text
autoannotate study
autoannotate_study
```

4. Keep Windows CUDA DLL registration using `os.add_dll_directory()`.

5. Update `check_environment.py` so it reports:

* CUDA availability.
* PyTorch CUDA version.
* GPU name.
* GroundingDINO path.
* Required model weights.
* Effective `AUTOANNOTATE_MAX_AREA_FRAC`.
* Stable Diffusion device.

6. Document Windows setup clearly in `HOW_TO_RUN.txt` and the README.

### Strongly Recommended Changes

1. Standardize the repo folder name to avoid spaces.
2. Use `requirements-windows11-cuda.txt` for research desktop installs (it pins the
   CUDA torch wheels); freeze it to `requirements-windows11-cuda.freeze.txt` once verified.
3. Add a small CUDA inference test.
4. Add clear warnings when PyTorch is CPU-only.
5. Keep model weights out of git but document exact filenames and target folders.
6. Add a troubleshooting section for common Windows errors.

---

## Common Windows Errors and Fixes

### `torch.cuda.is_available()` prints `False`

Likely cause:

* CPU-only PyTorch build.
* NVIDIA driver issue.
* CUDA/PyTorch mismatch.

Fix:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Then verify:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

---

### `invalid command 'bdist_wheel'`

Likely cause:

* `wheel` is missing.

Fix:

```powershell
pip install wheel setuptools
```

---

### MSVC preprocessor error during GroundingDINO build

Likely cause:

* CUDA 12.4 or CUDA 13 requires MSVC’s conforming preprocessor.

Fix:

```powershell
$env:CL = "/Zc:preprocessor"
pip install --no-build-isolation -e "autoannotate_study/GroundingDINO"
Remove-Item Env:\CL
```

---

### `DeprecatedTypeProperties` conversion error

Likely cause:

* GroundingDINO CUDA source is incompatible with modern PyTorch.

Fix:

```cpp
value.type()
```

should become:

```cpp
value.scalar_type()
```

---

### `DLL load failed while importing _C`

Likely cause:

* Windows did not find CUDA runtime DLLs when loading the compiled GroundingDINO extension.

Fix:

* Ensure `CUDA_PATH` is set.
* Register `CUDA_PATH\bin` with `os.add_dll_directory()` in `autoannotate/config.py`.

---

### `BertModel has no attribute get_head_mask`

Likely cause:

* Transformers 5.x is installed.

Fix:

```powershell
pip install "transformers>=4.40,<5"
```

or:

```powershell
pip install "transformers==4.50.3"
```

---

### GroundingDINO Folder Not Found

Likely cause:

* Folder name mismatch between `autoannotate study` and `autoannotate_study`.

Fix:

* Support both folder names in config.
* Set `GROUNDING_DINO_DIR` manually only if the folder is somewhere unusual.

---

### SAM2 Missing

Likely cause:

* `sam2_t.pt` downloaded to the working directory instead of `GUI and Pipeline/`.

Fix:

```powershell
Move-Item .\sam2_t.pt ".\GUI and Pipeline\sam2_t.pt"
```

---

### SAM3 Options Are Greyed Out

Likely cause:

* `sam3.pt` is missing.

Fix:

* Accept the SAM3 license on Hugging Face.
* Download `sam3.pt`.
* Place it in:

```text
GUI and Pipeline/sam3.pt
```

---

### DINO SwinB Option Is Greyed Out

Likely cause:

* SwinB weight is missing.

Fix:

* Download `groundingdino_swinb_cogcoor.pth`.
* Place it in the GroundingDINO weights folder.

---

### DINO SwinB Weight Exists but Loading or Inference Still Fails

Run:

```powershell
python "GUI and Pipeline/check_environment.py"
python -c "from autoannotate.pipeline.dino import load_dino_model; m=load_dino_model('swinb'); print(type(m).__name__, sum(p.numel() for p in m.parameters()))"
```

The second command should finish with:

```text
GroundingDINO 232903808
```

If it does, the SwinB config and checkpoint match and load correctly. Save the
full environment-check output and the complete inference traceback. In
particular, check the reported PyTorch CUDA build, GPU memory, and
`GroundingDINO _C extension` result. SwinB is substantially heavier than SwinT;
an error that occurs only during prediction is more likely to be CUDA/host
memory pressure or a compiled-extension problem than a missing checkpoint.

---

### `.env` Appears Ignored

Likely cause:

* Wrong `.env` location.
* Wrong working directory.
* Wrong virtual environment.
* Variable loaded after startup.
* Value not imported from central config.
* VS Code terminal environment injection warning causing confusion.

Fix:

* Put `.env` in the repo root.
* Load `.env` in `autoannotate/config.py`.
* Use hardcoded defaults in config.
* Print effective values in `check_environment.py`.

---

## Final Conclusion

AutoAnnotate was originally developed and tested on Windows, but the more recent cleanup, portability improvements, and feature updates were completed on macOS. The work described here brought that newer version back onto a Windows NVIDIA GPU setup and resolved the issues that appeared during that process.

The main challenge was that Windows CUDA execution requires stricter alignment between PyTorch, CUDA, NVIDIA drivers, compiler tools, Python package versions, model weights, and compiled extension loading.

The research desktop should not be configured by following only the older setup steps, because they do not fully account for the newer code changes, Windows CUDA requirements, and dependency issues found during this setup.

The correct process is:

1. Install CUDA-enabled PyTorch first.
2. Verify GPU access.
3. Install dependencies with compatible pinned versions.
4. Install Windows build tools.
5. Patch and build GroundingDINO.
6. Place model weights in the expected folders.
7. Verify the environment with `check_environment.py`.
8. Run the headless tests.
9. Launch the GUI.
10. Only then begin large batch annotation work.

The most important project-level fixes are:

* Pin `transformers<5`.
* Patch GroundingDINO’s CUDA source for modern PyTorch.
* Register CUDA DLL directories on Windows.
* Support both GroundingDINO folder names.
* Verify `.env` and config values through `check_environment.py`.
* Document the Windows CUDA path as a first-class setup path.

With these changes, the research desktop should be able to process large image folders using GPU acceleration and reduce the amount of manual annotation work required.

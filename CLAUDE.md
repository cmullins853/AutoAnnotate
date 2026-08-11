# AutoAnnotate

Semi-automatic image annotation. A PyQt5 desktop app wrapping GroundingDINO,
YOLOE and SAM2/SAM3: load a folder of images, prompt by text or by drawing
boxes, review and correct the results, then batch annotate the rest.

Runs on macOS, Windows and Linux. Development happens on a Mac; the GPU paths
are exercised on Windows with NVIDIA.

## Running it

From the repo root with the venv active:

    python run_app.py            # or: python -m autoannotate

There is no notebook launcher any more. All the code lives in the
`autoannotate/` package.

## Setting up a machine

Pick the requirements file for the machine, they are not interchangeable:

| Machine | File |
| --- | --- |
| macOS / Apple Silicon | `requirements-macos.lock` |
| Windows 10, CPU only | `requirements-windows10-cpu.txt` |
| Windows 11 + NVIDIA | `requirements-windows11-cuda.txt` |
| Linux, or anything else | `requirements.txt` |

Then, and the quoting matters because of the space in the folder name:

    pip install --no-build-isolation -e "autoannotate study/GroundingDINO"
    python "GUI and Pipeline/download_weights.py"
    python "GUI and Pipeline/check_environment.py"

`--no-build-isolation` is required: GroundingDINO's `setup.py` imports torch at
build time, so torch has to be installed first. Windows also needs Visual
Studio Build Tools with the C++ workload, because GroundingDINO builds a C++
extension. `GUI and Pipeline/HOW_TO_RUN.txt` is the authoritative setup guide
and covers every step in detail.

`sam3.pt` is gated on Hugging Face and needs a token for the one time download
(`download_weights.py --hf-token ...`). The app itself never uses a token.

## The GroundingDINO folder trap

Read this before touching anything to do with GroundingDINO. It has already
cost one evening.

The folder is tracked as `autoannotate study` (with a space). Copying or
unzipping the project sometimes produces `autoannotate_study` instead, and a
machine can end up with both. Only the space version is tracked by git, so the
other one goes stale on every pull. If pip's editable install points at the
untracked copy, the app imports stale or incomplete code. That presents as the
splash screen opening normally and then `ModuleNotFoundError: No module named
'groundingdino'` the moment you enter Manual mode.

Rules:

- Always install the editable package against the **space** spelling.
- Do not copy the project folder around. Clone it.
- `.gitignore` excludes `*.so`, `*.pyd` and `weights/`, so the compiled `_C`
  extension and the `.pth` checkpoints exist only in the tree they were built or
  downloaded into. Never move someone to a different tree without copying those
  across first. Rebuilding the CUDA extension is slow.
- If both spellings already exist, linking them is the durable fix:
  `mklink /J "autoannotate_study" "autoannotate study"` on Windows as
  Administrator, or `ln -s "autoannotate study" autoannotate_study` elsewhere.
  A link reads as one tree everywhere in the code and triggers no warnings.

`check_environment.py` diagnoses all of this under "GroundingDINO install
integrity": where the package imports from, which copy pip was installed
against, what `GROUNDING_DINO_DIR` resolved to, and which files are missing.

## Tests

    python "GUI and Pipeline/test_semiauto_headless.py"

Around 900 checks, no GUI, no models, no network. It stubs the heavy ML
packages and runs the real GUI logic under an offscreen Qt platform. It should
be run before and after any change, and it must stay at zero failures.
Skips are reported separately and are worth reading, since a skip means that
machine could not test something.

CI runs the same suite on Linux and Windows for every push
(`.github/workflows/headless-tests.yml`). The Windows job exists to cover path,
encoding and newline handling, which cannot fail on a Mac.

Neither the suite nor CI covers CUDA, VRAM behaviour, the out of memory retry
path, model loading, or anything visual. Those need a real GPU machine and a
walk through `GUI and Pipeline/MANUAL_TEST_CHECKLIST.md`, which is still
unticked.

## Layout

    autoannotate/
      config.py        paths, device selection, tuning knobs, GROUNDING_DINO_DIR
      imageio.py       the ONLY correct way to read and write images
      coco.py          YOLO output folder to COCO JSON converter
      optimizer.py     prompt and confidence scoring behind the Automated window
      palette.py       per class colours
      pipeline/        model wrappers and label IO, no Qt
      gui/             the PyQt5 app, built on pipeline
    GUI and Pipeline/  scripts, docs, checklists, weights, the test suite

`autoannotate/pipeline/__init__.py` imports dino, sam, sd and yoloe eagerly, so
anything that does not need models should not live under `pipeline/`.

## Conventions and gotchas

- **Image IO**: never call `cv2.imread` or `cv2.imwrite`. Use
  `autoannotate/imageio.py`. OpenCV uses the ANSI API on Windows and fails
  silently on non-ASCII paths.
- **Text IO**: every text handle needs `encoding="utf-8"`, and writers need
  `newline="\n"` so Windows does not produce different label files than macOS.
- **Writing**: no emojis and no em or en dashes in code, comments, commit
  messages or docs. Reword rather than substituting a different dash. Functional
  GUI glyphs such as the collapse arrows are fine.
- **Label writes** go through the atomic helper in `pipeline/labels.py`. Opening
  a label file in `'w'` and formatting rows as you go destroys the previous
  labels if anything raises partway.
- **Parallel lists**: boxes, classes and masks are kept index aligned in several
  places, and misalignment here has caused real bugs. Be careful with any
  filtering that drops from one list and not the others.
- **SAM3 box prompts are one concept per call**. Ultralytics forces `nc = 1` the
  moment `bboxes` is passed. Multi-class means one pass per class.
- Memory is tight in both directions: SAM3 is about 3.3 GB, which can silently
  kill an 8 GB Mac, and the CUDA path derives a model budget from total VRAM so
  DINO, YOLOE and SAM3 do not all sit on an 8 GB card at once. Suspect memory
  first when something dies without an error.

## Commit style

Plain sentences describing what changed and why, in the imperative or past
tense, with a body explaining the reasoning where it is not obvious. No emojis,
no dashes as punctuation, and no AI or model listed as an author or co-author.

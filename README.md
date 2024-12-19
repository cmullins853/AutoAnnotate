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
- Install required Python libraries from `requirements.txt`.
```bash
pip install -r requirements.txt
```
- **Software Dependencies:**
  - Grounding DINO configuration and weight files.
  - Segment Anything (SAM) weight files.

### Installation:
1. Clone the repository and navigate to the directory.
2. Ensure `requirements.txt` is updated and install all dependencies.

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

### Code Files:
- **[auto-annotate-gui.ipynb](GUI%20and%20Pipeline/auto-annotate-gui.ipynb)**:
  - Graphical User Interface for managing and interacting with the annotation system.
- **[auto-annotate-backend.py](auto-annotate-backend.py)**:
  - Core backend logic for image annotation using Grounding DINO and SAM models.
- **[manual-tuning-test.ipynb](manual-tuning-test.ipynb)**:
  - Jupyter notebook for manual testing and tuning of prompts and confidence levels.
- **[LLM implementation.ipynb](LLM%20implementation.ipynb)**:
  - Notebook for leveraging large language models in the annotation pipeline.
- **[yolo-world.ipynb](yolo-world.ipynb)**:
  - YOLO-world model integration for annotation.

### Instructions for Model Training and Testing:
#### Grounding DINO:
1. Specify the `config_path` and `checkpoint_path` in the backend code.
2. Adjust `box_threshold` and `text_threshold` for specific datasets.

#### Segment Anything Model (SAM):
1. Download the required SAM weights.
2. Update the model path (`mobile_sam.pt`) in `auto-annotate-backend.py`.

---

## Editing Guidelines

### Adding New Models:
- Ensure compatibility with the pipeline.
- For ML models in `auto-annotate-backend.py`:
  - Add the model with relevant hyperparameters in the codebase.

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

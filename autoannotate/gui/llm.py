"""SmolVLM prompt-suggestion model: lazy load, cache, and prompt generation."""
import os
import re

import torch
from PIL import Image
from PyQt5 import QtCore

class LLMWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)  # Signal to pass model and processor
    log = QtCore.pyqtSignal(str)  # Signal to send log text

    def run(self):
        # SmolVLM now loads LAZILY (on first use of prompt generation, via
        # ensure_llm/generate_prompts), NOT eagerly here -- so a Manual-only
        # session never pays its ~0.5GB resident cost or its startup load wait.
        # Emit (None, None) so the splash proceeds immediately; the model loads
        # the first time it is actually needed, from whichever window uses it
        # (the Automated window's Generate Prompts, or the Manual window's Auto
        # Annotate Remaining if prompt generation is invoked there).
        self.log.emit("VLM (SmolVLM) will load on first use; skipping eager load.\n")
        self.finished.emit(None, None)


def sort_largest_file(folder_path):
    # Dictionary to store file names and their line counts
    file_line_counts = {}

    # Iterate through files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .txt file
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            # Open the file and count lines
            with open(file_path, 'r', encoding='utf-8') as file:
                line_count = sum(1 for line in file)
            # Add the file and line count to the dictionary
            file_line_counts[file_name] = line_count
        else:
            print("File encountered not in .txt format.")
    # Sort files by line count in descending order and return as list of file names
    sorted_files = sorted(file_line_counts, key=file_line_counts.get, reverse=True)
    return sorted_files

def extract_descriptions(response):
    lines = response.split("\n")
    unwanted_keywords = ["user", "assistant", "describe", "text & image output", "model"]
    descriptions = []
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        if any(keyword in clean_line.lower() for keyword in unwanted_keywords):
            continue
        clean_line = re.sub(r"^\s*\d+[\.\)\-]\s*", "", clean_line)
        if clean_line:
            descriptions.append(clean_line)
    return descriptions

# Process-wide lazy cache for the prompt-suggestion VLM. None until first load;
# (model, processor) after a successful load; (None, None) if loading failed.
_llm_cache = None


def ensure_llm(log=None):
    """Load (once) and return (model, processor) for the SmolVLM prompt-suggestion
    model, caching it process-wide. Called lazily on first use so a Manual-only
    session never loads it. Returns (None, None) on failure so callers degrade
    gracefully (no prompts). `log` is an optional callable(str) for UI status."""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    def _say(m):
        try:
            if log:
                log(m)
        except Exception:
            pass
        print(m, end="")
    try:
        _say("Loading SmolVLM model (first use)...\n")
        # transformers is imported lazily so the app (and headless tests)
        # start without paying its multi-second import unless the VLM is used.
        from transformers import AutoModelForVision2Seq, AutoProcessor
        model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device)
        processor = AutoProcessor.from_pretrained(model_id)
        _say("SmolVLM loaded.\n")
        _llm_cache = (model, processor)
    except Exception as e:
        _say(f"SmolVLM load failed: {e}\n")
        _llm_cache = (None, None)
    return _llm_cache


def release_llm():
    """Drop the cached VLM and free its ~0.5GB (e.g. on shutdown). The next
    ensure_llm() reloads it."""
    global _llm_cache
    _llm_cache = None
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                and hasattr(torch.mps, "empty_cache")):
            torch.mps.empty_cache()
    except Exception:
        pass


def generate_prompts(image_path, manual_entry, model=None, processor=None):
    # Lazy-load the VLM if the caller didn't supply it (it's no longer loaded on
    # the splash). This is what lets ANY window -- including the Manual window's
    # Auto Annotate Remaining -- use prompt generation: pass None and it loads
    # (and caches) here. Returns [] if the VLM can't load.
    if model is None or processor is None:
        model, processor = ensure_llm()
    if model is None or processor is None:
        print("Prompt generation unavailable: VLM could not be loaded.")
        return []
    try:
        raw_image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            f"Describe the {manual_entry} of the image in 3 words maximum for prompt use in a zero-shot detection model, "
                            "and give 5 separate entries, each separated by a new line, and its own separate descriptor of the target. "
                            "Number each prompt. Then simply new line. Strictly the prompts, no other response is required. "
                            "Use visual description of the target in the image only. Do not duplicate responses."
                        ),
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)
        # do_sample=True or generate() runs GREEDY and silently ignores both
        # temperature and top_p, which is the opposite of the varied,
        # non-duplicate suggestions the prompt above asks for.
        output = model.generate(**inputs, do_sample=True, temperature=0.7,
                                top_p=0.9, max_new_tokens=512)
        response = processor.decode(output[0], skip_special_tokens=True)
        return extract_descriptions(response)
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return []

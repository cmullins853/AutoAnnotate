"""ManualWindow: the main annotation workflow (detect, segment, edit, save, augment)."""
import os
import tempfile

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5 import QtWidgets, QtGui, QtCore

from ..config import AUTOANNOTATE_DEBUG, CUMULATIVE_DIR, GROUNDING_DINO_DIR, WEIGHTS_DIR
from ..pipeline import sam as sam_module
from ..pipeline.dino import load_dino_model, run_dino_from_model
from ..pipeline.labels import (_mask_to_polys, result_clean_polys, save_boxes_yolo,
                               save_masks, save_polys_yolo, verify_boxes_round_trip)
from ..pipeline.overlay import adjust_masks, draw_boxes_on_image, overlay_with_borders
from ..pipeline.sam import (load_sam, run_sam3_boxes, run_sam3_text,
                            segment_with_boxes)
from ..pipeline.sd import _SD_STRENGTH, generate_variation
from ..pipeline.yoloe import load_yoloe, run_yoloe_text, run_yoloe_vis
from .canvas import AnnotationCanvas
from .dialogs import (BatchVariationViewer, InfoBadge, SDPromptDialog,
                      SemiAutoSettingsDialog, VariationPreviewDialog)
from .spatial import SpatialGrid
from .style import (BTN_BLUE, BTN_GAP, BTN_GREEN, BTN_GREY, BTN_ORANGE, BTN_PURPLE,
                    BTN_RED, TGL_DRAW_ON, TGL_EDIT_ON, TOOLTIP_AUTODRAW, TOOLTIP_BOX,
                    TOOLTIP_SEMIAUTO, TOOLTIP_SEMIAUTO_EDIT, _SD_DEFAULT_NEG,
                    _SD_DEFAULT_PROMPT, btn_qss, lock_during, slider_qss, toggle_qss,
                    tool_toggle_qss)

class ManualWindow(QtWidgets.QWidget):
    # Detector and segmenter are picked independently. YOLOE-seg is a one-shot
    # detector+segmenter; when chosen, the segmenter dropdown locks to "(none)".
    DETECTORS = [
        "DINO (SwinT)",
        "DINO (SwinB)",
        "YOLOE-vis",
        "YOLOE-seg (one-shot)",
        "SAM3 (one-shot)",
    ]
    SEGMENTERS = [
        "SAM2 (tiny)",
        "SAM3",
        "(none)",   # only valid when detector is YOLOE-seg one-shot
    ]
    # Back-compat shim, old code paths still reference PIPELINE_PRESETS strings.
    PIPELINE_PRESETS = [
        "DINO (SwinT) + SAM2",
        "DINO (SwinT) + SAM3",
        "DINO (SwinB) + SAM2",
        "DINO (SwinB) + SAM3",
        "YOLOE-vis standalone",
        "YOLOE-seg standalone",
        "SAM3 standalone",
    ]

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.current_image_index = 0
        self.images = []
        self.output_folder = ""
        self.base_cv2_image = None
        # per-mode cached results
        self.baked_bbox_cv2 = None
        self.baked_seg_cv2  = None
        self.bbox_anns      = []
        self.seg_anns       = []
        self.current_mode   = None
        # Single source of truth for the current image's box set (image-coord xyxy).
        # Always reflects: detector output ∪ manual draws − deletions.
        self.live_boxes = []
        # Parallel array: 'detector' or 'manual' for each entry of live_boxes.
        # Used to tag rect/poly annotations on regen and on mode switch.
        self.live_box_sources = []
        # Last segmentation polygons (normalized 0-1) keyed by box index in live_boxes,
        # so switching seg->bbox->seg without edits doesn't re-run SAM unnecessarily.
        self.live_polys_cache = None
        # Lazy model cache (key -> model) + per-key LRU timestamps. Eviction only
        # happens when AUTOANNOTATE_MODEL_BUDGET_GB is set; unset = the original
        # unbounded cache. DINO now loads on first use (see the DINO property),
        # not eagerly, so a YOLOE/SAM3-only session never pays its ~0.7GB cost.
        self._model_cache = {}
        self._model_lru = {}
        self._model_lru_tick = 0
        # Active tool for the Draw Boxes split button: "box" or "semiauto".
        self._draw_tool = "box"
        # Which mode the Edit button engages: "boxes" (handle/X editing of
        # boxes+masks) or "masks" (drawn-mask vertex/point editing). The Edit
        # dropdown picks it; the button's ON/OFF + colour reflect the choice.
        self._edit_tool = "boxes"
        # True while a mode toggle is programmatically flipping other buttons, so
        # the discard guard does not re-prompt for the same switch.
        self._in_mode_switch = False
        # GUI state for prompt mode + model selection (set by init_ui()).
        self.prompt_mode = "text"
        self.detector_choice  = self.DETECTORS[0]
        self.segmenter_choice = self.SEGMENTERS[0]
        # Back-compat: keep current_pipeline updated for legacy readers.
        self.current_pipeline = self.PIPELINE_PRESETS[0]
        self.init_ui()
        # DINO loads lazily on first use via _get_model("dino_swint") (the DINO
        # property), not eagerly here, so a session that only runs YOLOE/SAM3
        # never pays DINO's ~0.7GB resident cost -- it matters on an 8GB box
        # where SAM3 (~3.3GB) already dominates the budget.
        # Hook canvas edits so live_boxes stays in sync.
        self.image_label.boxes_changed.connect(self._on_canvas_changed)
        # Semi-auto SAM mask drawing: live SAM re-run on each point, commit on Enter.
        self.image_label.mask_point_added.connect(self._on_mask_point_added)
        self.image_label.mask_commit_requested.connect(self._commit_mask_object)
        self.image_label.mask_close_requested.connect(self._close_mask_object)
        self.image_label.semiauto_apply_requested.connect(self._apply_semiauto_edit)
        self.image_label.semiauto_settings_requested.connect(self._open_semiauto_settings)
        self.image_label.semiauto_delete_requested.connect(self._delete_semiauto_mask)
        self.image_label.mask_selected.connect(self._on_mask_selected)
        self.image_label.semiauto_min_vertex_delete.connect(self._on_min_vertex_delete)

    def init_ui(self):
        self.setWindowTitle("Manual Prompt and Confidence Tuning")
        self.showFullScreen()
        self.setStyleSheet("background-color: #454545;")

        # Scale UI to screen height so everything fits on any Mac display
        _geo = QtWidgets.QApplication.primaryScreen().geometry()
        screen_h, screen_w = _geo.height(), _geo.width()
        # Buttons trimmed ~20% (was screen_h // 16) so the controls take less
        # room and the image canvas can claim more of the window.
        btn_h  = max(34, screen_h // 20)
        font   = max(13, screen_h // 58)
        # Cap the left control column so it can't grow into the image; the
        # image column gets the horizontal stretch instead (see right_layout).
        left_panel_w = max(300, min(360, screen_w // 6))

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(10)

        # Left column: controls
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.setSpacing(BTN_GAP)

        lbl_style      = f"color: white; font-size: {font}px;"

        back_btn = QtWidgets.QPushButton("Back")
        back_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        back_btn.setFixedHeight(btn_h)
        back_btn.setToolTip("Return to the main menu.")
        back_btn.clicked.connect(self.go_back)
        left_layout.addWidget(back_btn)

        folder_btn = QtWidgets.QPushButton("Select Image Folder")
        folder_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        folder_btn.setFixedHeight(btn_h)
        folder_btn.setToolTip("Choose the folder of images you want to annotate.")
        folder_btn.clicked.connect(self.select_folder)
        left_layout.addWidget(folder_btn)

        output_folder_btn = QtWidgets.QPushButton("Select Output Folder")
        output_folder_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        output_folder_btn.setFixedHeight(btn_h)
        output_folder_btn.setToolTip("Choose where saved labels, segments and annotated images are written.")
        output_folder_btn.clicked.connect(self.select_output_folder)
        left_layout.addWidget(output_folder_btn)

        # Model selection: independent Detector + Segmenter dropdowns
        # Each combo gets its own label directly above it, with horizontal
        # breathing room between the two columns.
        models_row = QtWidgets.QHBoxLayout()
        models_row.setSpacing(18)

        detector_col = QtWidgets.QVBoxLayout()
        detector_col.setSpacing(4)
        detector_label = QtWidgets.QLabel("Detector:")
        detector_label.setStyleSheet(lbl_style)
        detector_col.addWidget(detector_label)
        self.detector_combo = QtWidgets.QComboBox()
        self.detector_combo.addItems(self.DETECTORS)
        self.detector_combo.setStyleSheet(f"font-size: {font}px; color: white; background-color: black;")
        self.detector_combo.setFixedHeight(int(btn_h * 0.75))
        self.detector_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                          QtWidgets.QSizePolicy.Fixed)
        self.detector_combo.setToolTip("Pick the object detector. DINO is text-prompted; YOLOE-vis "
                                       "is box-prompted; YOLOE-seg and SAM3 accept either. Greyed "
                                       "items are missing their weights on disk.")
        self.detector_combo.currentTextChanged.connect(self._on_detector_changed)
        detector_col.addWidget(self.detector_combo)
        models_row.addLayout(detector_col, 1)

        segmenter_col = QtWidgets.QVBoxLayout()
        segmenter_col.setSpacing(4)
        segmenter_label = QtWidgets.QLabel("Segmenter:")
        segmenter_label.setStyleSheet(lbl_style)
        # "Segmenter:" label with a hover info dot that explains the annotation
        # colour code (provenance + state at a glance).
        seg_label_row = QtWidgets.QHBoxLayout()
        seg_label_row.setSpacing(6)
        seg_label_row.setContentsMargins(0, 0, 0, 0)
        seg_label_row.addWidget(segmenter_label)
        self.color_info_dot = InfoBadge("i")
        _dot_d = int(font * 1.6)
        self.color_info_dot.setFixedSize(_dot_d, _dot_d)
        self.color_info_dot.setAlignment(QtCore.Qt.AlignCenter)
        self.color_info_dot.setStyleSheet(
            f"color: {BTN_BLUE}; font-size: {int(font * 0.85)}px; font-weight: bold; "
            f"border: 1px solid {BTN_BLUE}; border-radius: {_dot_d // 2}px;")
        self.color_info_dot.setCursor(QtCore.Qt.WhatsThisCursor)
        self.color_info_dot.set_info_text(
            "Annotation colors:\n"
            "Yellow = prompt input\n"
            "Green = your hand work\n"
            "Magenta = model output\n"
            "Cyan = selected / preview\n"
            "Red = delete / negative\n\n"
            "Button colors:\n"
            "Green = short to medium press time\n"
            "Yellow/Orange = medium press time\n"
            "Red = long press time\n"
            "Blue = configuration\n"
            "Light purple = edit\n"
            "Deep purple = synthetic image generation\n\n"
            "Sliders (best values):\n"
            "SD: 0.1 to 0.2\n"
            "Detector confidence: around 20 to 30\n"
            "Segmenter confidence: around 20 to 40 (depends on object size)")
        seg_label_row.addWidget(self.color_info_dot)
        seg_label_row.addStretch(1)
        segmenter_col.addLayout(seg_label_row)
        self.segmenter_combo = QtWidgets.QComboBox()
        self.segmenter_combo.addItems(self.SEGMENTERS)
        self.segmenter_combo.setStyleSheet(f"font-size: {font}px; color: white; background-color: black;")
        self.segmenter_combo.setFixedHeight(int(btn_h * 0.75))
        self.segmenter_combo.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                           QtWidgets.QSizePolicy.Fixed)
        self.segmenter_combo.setToolTip("Pick the segmenter that turns detector boxes into masks. "
                                        "Locks to (none) for one-shot detectors (YOLOE-seg / SAM3) "
                                        "that already produce their own masks.")
        self.segmenter_combo.currentTextChanged.connect(self._on_segmenter_changed)
        segmenter_col.addWidget(self.segmenter_combo)
        models_row.addLayout(segmenter_col, 1)

        left_layout.addLayout(models_row)

        # Pre-flight: gray out detector items whose checkpoints aren't on disk.
        self._apply_detector_availability()

        # Hidden legacy combo so any code path that still reads
        # self.pipeline_combo doesn't break. Kept in sync via _sync_pipeline_combo.
        self.pipeline_combo = QtWidgets.QComboBox()
        self.pipeline_combo.addItems(self.PIPELINE_PRESETS)
        self.pipeline_combo.hide()

        # Prompt-mode toggle (text vs boxes)
        prompt_mode_label = QtWidgets.QLabel("Prompt:")
        prompt_mode_label.setStyleSheet(lbl_style)
        left_layout.addWidget(prompt_mode_label)

        prompt_mode_row = QtWidgets.QHBoxLayout()
        self.prompt_mode_group = QtWidgets.QButtonGroup(self)
        self.prompt_mode_text_btn = QtWidgets.QRadioButton("Text")
        self.prompt_mode_boxes_btn = QtWidgets.QRadioButton("Boxes")
        self.prompt_mode_text_btn.setStyleSheet(lbl_style)
        self.prompt_mode_boxes_btn.setStyleSheet(lbl_style)
        self.prompt_mode_text_btn.setChecked(True)
        self.prompt_mode_text_btn.setToolTip("Prompt the detector with the typed text prompt below.")
        self.prompt_mode_boxes_btn.setToolTip("Prompt the detector with boxes you draw on the image "
                                              "(only for box-capable detectors).")
        self.prompt_mode_group.addButton(self.prompt_mode_text_btn)
        self.prompt_mode_group.addButton(self.prompt_mode_boxes_btn)
        prompt_mode_row.addWidget(self.prompt_mode_text_btn)
        prompt_mode_row.addWidget(self.prompt_mode_boxes_btn)
        prompt_mode_row.addStretch()
        left_layout.addLayout(prompt_mode_row)
        self.prompt_mode_text_btn.toggled.connect(
            lambda checked: checked and self._on_prompt_mode_changed("text")
        )
        self.prompt_mode_boxes_btn.toggled.connect(
            lambda checked: checked and self._on_prompt_mode_changed("boxes")
        )

        self.prompt_label_widget = QtWidgets.QLabel("Enter Prompt:")
        self.prompt_label_widget.setStyleSheet(lbl_style)
        left_layout.addWidget(self.prompt_label_widget)

        self.prompt_entry = QtWidgets.QLineEdit()
        self.prompt_entry.setStyleSheet(f"font-size: {font}px; color: white; background-color: black;")
        self.prompt_entry.setFixedHeight(int(btn_h * 0.75))
        self.prompt_entry.setToolTip("Text describing what to detect (e.g. 'blueberry'). Used by "
                                     "text-capable detectors; comma-separate multiple classes.")
        # Typing/clearing the prompt changes whether Auto Annotate can run.
        self.prompt_entry.textChanged.connect(lambda _t: self._refresh_auto_annotate_enabled())
        left_layout.addWidget(self.prompt_entry)

        # Display-mode toggles sit directly above the confidence sliders.
        self.checkbox_layout = QtWidgets.QHBoxLayout()
        self.box_checkbox  = QtWidgets.QCheckBox("Bounding Box")
        self.mask_checkbox = QtWidgets.QCheckBox("Segmentation")
        self.box_checkbox.setStyleSheet(lbl_style)
        self.mask_checkbox.setStyleSheet(lbl_style)
        self.box_checkbox.setToolTip("Show bounding boxes on the image and save them to the boxes/ folder.")
        self.mask_checkbox.setToolTip("Show segmentation masks on the image and save them to the segments/ folder.")
        self.checkbox_layout.addWidget(self.box_checkbox)
        self.checkbox_layout.addWidget(self.mask_checkbox)
        self.checkbox_layout.addStretch()
        self.box_checkbox.stateChanged.connect(self._on_box_checked)
        self.mask_checkbox.stateChanged.connect(self._on_mask_checked)
        left_layout.addLayout(self.checkbox_layout)

        # Detector confidence (filters detector-output boxes)
        det_label = QtWidgets.QLabel("Detector confidence: 50")
        det_label.setStyleSheet(lbl_style)
        det_label.setToolTip("Higher = fewer but more confident detections from the detector.\nNote: YOLOE detectors auto-rescale this slider into their practical range (slider 25 ~= 0.05, slider 50 ~= 0.10, slider 100 ~= 0.20). DINO/SAM use the raw value.")
        left_layout.addWidget(det_label)
        self.detection_threshold_label = det_label

        self.detection_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.detection_threshold_slider.setRange(0, 100)
        self.detection_threshold_slider.setValue(50)
        self.detection_threshold_slider.setFixedHeight(30)
        # Force QSS rendering; the native macOS slider drag-bubble can
        # render duplicated or off-position when the widget is height-pinned.
        self.detection_threshold_slider.setStyleSheet(slider_qss())
        left_layout.addWidget(self.detection_threshold_slider)
        self.detection_threshold_slider.valueChanged.connect(self._update_detection_threshold_label)
        self.confidence_slider = self.detection_threshold_slider  # legacy alias

        # Segmenter confidence (stricter mask filter; meaning depends on pipeline)
        mask_label = QtWidgets.QLabel("Segmenter confidence: 30")
        mask_label.setStyleSheet(lbl_style)
        mask_label.setToolTip(
            "Stricter mask filter. For DINO this drives text-class match strictness; "
            "for YOLOE-seg it overrides detection threshold for the segmenter pass; "
            "for plain YOLOE+SAM it currently has no effect (SAM has no conf knob)."
        )
        left_layout.addWidget(mask_label)
        self.mask_threshold_label = mask_label

        self.mask_threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.mask_threshold_slider.setRange(0, 100)
        self.mask_threshold_slider.setValue(30)
        self.mask_threshold_slider.setFixedHeight(30)
        self.mask_threshold_slider.setStyleSheet(slider_qss())
        left_layout.addWidget(self.mask_threshold_slider)
        self.mask_threshold_slider.valueChanged.connect(self._update_mask_threshold_label)
        self.box_threshold_slider = self.mask_threshold_slider  # legacy alias

        # (No mid-panel stretch here: the controls group together and any
        # spare vertical space falls to the bottom of the column instead of
        # leaving a dead gap above the bottom buttons.)

        # Bottom buttons: Regenerate / Next Image / Auto Annotate Remaining /
        # carry toggle. Stacked vertically so the capped-width column fits
        # their full labels without clipping (was a wide horizontal row).
        bottom_layout = QtWidgets.QVBoxLayout()
        bottom_layout.setSpacing(BTN_GAP)

        regen_btn = QtWidgets.QPushButton("Regenerate")
        regen_btn.setStyleSheet(btn_qss(BTN_ORANGE, font))
        regen_btn.setFixedHeight(btn_h)
        regen_btn.setToolTip("Re-run the detector/segmenter on this image and overwrite its saved labels with the fresh model output.")
        regen_btn.clicked.connect(self.display_predictions)
        self.regen_btn = regen_btn
        bottom_layout.addWidget(regen_btn)

        next_btn = QtWidgets.QPushButton("Next Image")
        next_btn.setStyleSheet(btn_qss(BTN_GREEN, font))
        next_btn.setFixedHeight(btn_h)
        next_btn.setToolTip("Save the current image's labels and move to the next image in the folder.")
        next_btn.clicked.connect(lambda: lock_during(next_btn, self.next_image))
        self.next_btn = next_btn
        bottom_layout.addWidget(next_btn)

        auto_annotate_btn = QtWidgets.QPushButton("Auto Annotate Remaining")
        auto_annotate_btn.setStyleSheet(btn_qss(BTN_RED, font))
        auto_annotate_btn.setFixedHeight(btn_h)
        auto_annotate_btn.clicked.connect(self.auto_annotate_remaining)
        self.auto_annotate_btn = auto_annotate_btn
        bottom_layout.addWidget(auto_annotate_btn)

        # Carry Prompts Forward: opt-in toggle. When ON, Auto Annotate
        # Remaining uses THIS image -- its drawn boxes + typed prompt -- as a
        # one-shot reference for every remaining image and auto-runs the model.
        # Replaces the old "Carry boxes forward" checkbox. Same attribute name
        # so existing readers keep working.
        self.carry_forward_checkbox = QtWidgets.QPushButton("Use First Image as Prompt: OFF")
        self.carry_forward_checkbox.setCheckable(True)
        self.carry_forward_checkbox.setChecked(False)
        self.carry_forward_checkbox.setStyleSheet(toggle_qss(BTN_BLUE, font))
        self.carry_forward_checkbox.setFixedHeight(btn_h)
        self.carry_forward_checkbox.toggled.connect(self._on_carry_toggled)
        self.carry_forward_checkbox.setToolTip(
            "Uses the first image (its drawn boxes and typed prompt) as the "
            "prompt for every other image when you run Auto Annotate Remaining.\n"
            "Turn this on for box-mode prompting so the boxes you draw carry "
            "across the whole folder. With it off, the boxes you draw only "
            "apply to the image you are on."
        )
        bottom_layout.addWidget(self.carry_forward_checkbox)
        self._refresh_carry_checkbox_enabled()
        self._refresh_auto_annotate_enabled()

        left_layout.addLayout(bottom_layout)
        left_layout.addStretch()  # trailing space keeps controls top-grouped
        # Pack the controls into a fixed-width container so the column can't
        # expand into the image; right_layout (the image) gets the stretch.
        left_container = QtWidgets.QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(left_panel_w)
        main_layout.addWidget(left_container)

        # Right column: image + checkboxes
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(max(4, BTN_GAP - 2))
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Position indicator: which image of the folder is on screen.
        self.image_index_label = QtWidgets.QLabel("No image folder selected")
        self.image_index_label.setStyleSheet(f"color: white; font-size: {font}px;")
        self.image_index_label.setAlignment(QtCore.Qt.AlignCenter)
        right_layout.addWidget(self.image_index_label)

        self.image_label = AnnotationCanvas()
        right_layout.addWidget(self.image_label, stretch=1)

        # Draw-mode controls sit just above the checkboxes
        draw_row = QtWidgets.QHBoxLayout()
        draw_row.setSpacing(BTN_GAP)
        # Draw Boxes is a split toggle button (matches Edit Boxes / Image Resize):
        # the main body toggles whichever tool is currently selected, and the
        # dropdown picks the tool: "Draw Boxes" or "Semi-Automatic Segmentation".
        # The selected tool's name shows on the button; ON is orange (TGL_DRAW_ON)
        # for both tools. self._draw_tool ("box"|"semiauto") is the active tool.
        self.draw_btn = QtWidgets.QToolButton()
        self.draw_btn.setText("Draw Boxes: OFF")
        self.draw_btn.setStyleSheet(tool_toggle_qss(TGL_DRAW_ON, font))
        self.draw_btn.setFixedHeight(int(btn_h * 0.8))
        self.draw_btn.setCheckable(True)
        self.draw_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.draw_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.draw_btn.setToolTip(TOOLTIP_BOX)  # updated per active tool by _update_draw_btn_label
        self.draw_btn.toggled.connect(self._toggle_draw_btn)
        # Dropdown: exclusive tool picker. "Draw Boxes" is always available;
        # "Semi-Automatic Segmentation" is enabled only when an interactive SAM
        # model is active (SAM2/SAM3 segmenter or SAM3 one-shot), greyed for the
        # YOLOE standalone detectors and segmenter (none). Picking a tool ARMS it
        # (button OFF); press the button to activate. See _select_draw_tool.
        draw_menu = QtWidgets.QMenu()
        draw_menu.setToolTipsVisible(True)
        self.draw_tool_group = QtWidgets.QActionGroup(self)
        self.draw_tool_group.setExclusive(True)
        self.draw_tool_box_action = QtWidgets.QAction("Draw Boxes", self)
        self.draw_tool_box_action.setCheckable(True)
        self.draw_tool_box_action.setChecked(True)
        self.draw_tool_box_action.setToolTip(
            "Draw rectangles. For box-capable detectors they become yellow prompt "
            "boxes; otherwise they save as green manual annotations.")
        self.draw_tool_box_action.triggered.connect(lambda: self._select_draw_tool("box", activate=True))
        self.draw_tool_group.addAction(self.draw_tool_box_action)
        draw_menu.addAction(self.draw_tool_box_action)
        # "Semi-Automatic Point Segmentation": a single foreground point -> SAM
        # masks that one object. (internal tool key: "autodraw")
        self.autodraw_action = QtWidgets.QAction("Semi-Automatic Point Segmentation", self)
        self.autodraw_action.setCheckable(True)
        self.autodraw_action.setEnabled(False)  # gated on SAM model
        self.autodraw_action.setToolTip(TOOLTIP_AUTODRAW)
        self.autodraw_action.triggered.connect(lambda: self._select_draw_tool("autodraw", activate=True))
        self.draw_tool_group.addAction(self.autodraw_action)
        draw_menu.addAction(self.autodraw_action)
        # "Manually Draw Masks": connected foreground points around the object
        # (curve-tool style, closed to segment). (internal tool key: "semiauto")
        self.mask_draw_action = QtWidgets.QAction("Manually Draw Masks", self)
        self.mask_draw_action.setCheckable(True)
        self.mask_draw_action.setEnabled(False)  # gated on SAM model
        self.mask_draw_action.setToolTip(TOOLTIP_SEMIAUTO)
        self.mask_draw_action.triggered.connect(lambda: self._select_draw_tool("semiauto", activate=True))
        self.draw_tool_group.addAction(self.mask_draw_action)
        draw_menu.addAction(self.mask_draw_action)
        self.draw_btn.setMenu(draw_menu)
        self.draw_btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        draw_row.addWidget(self.draw_btn, 1)

        # Image Resize: scroll/pinch zooms toward the cursor, drag pans. The
        # zoom/pan PERSISTS after untoggling so the user can annotate zoomed
        # in; Save & Confirm (and loading a new image) restores fit.
        self.resize_btn = QtWidgets.QToolButton()
        self.resize_btn.setText("Image Resize: OFF")
        self.resize_btn.setStyleSheet(tool_toggle_qss(BTN_BLUE, font))
        self.resize_btn.setFixedHeight(int(btn_h * 0.8))
        self.resize_btn.setCheckable(True)
        self.resize_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.resize_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.resize_btn.setToolTip(
            "Zoom (scroll / pinch) and pan (drag) the image. Your zoom stays "
            "after you turn this off so you can annotate zoomed in; the dropdown's "
            "Original Size (and Save & Confirm) returns the image to its original size.")
        self.resize_btn.toggled.connect(self._toggle_resize_mode)
        resize_menu = QtWidgets.QMenu()
        resize_menu.setToolTipsVisible(True)
        original_size_act = QtWidgets.QAction("Original Size", self)
        original_size_act.setToolTip("Reset zoom/pan to the original (fit) size.")
        original_size_act.triggered.connect(self.image_label.reset_view)
        resize_menu.addAction(original_size_act)
        resize_menu.addSeparator()
        # Tint options (view-only): Normal is the default; Darken dims the
        # image around your detections so they stand out (Roboflow-style).
        # Saved images always use the normal tint regardless of this choice.
        self.tint_group = QtWidgets.QActionGroup(self)
        self.tint_group.setExclusive(True)
        self.normal_tint_act = QtWidgets.QAction("Normal Tint", self)
        self.normal_tint_act.setCheckable(True)
        self.normal_tint_act.setChecked(True)
        self.normal_tint_act.setToolTip("Show the image at full brightness (default).")
        self.normal_tint_act.triggered.connect(
            lambda: self.image_label.set_dark_tint(False))
        self.tint_group.addAction(self.normal_tint_act)
        resize_menu.addAction(self.normal_tint_act)
        self.darken_tint_act = QtWidgets.QAction("Darken Tint", self)
        self.darken_tint_act.setCheckable(True)
        self.darken_tint_act.setToolTip(
            "Dim everything except your detections (boxes, masks, manual and "
            "semi-auto segments) so they stand out. View only; saved images "
            "stay at normal brightness.")
        self.darken_tint_act.triggered.connect(
            lambda: self.image_label.set_dark_tint(True))
        self.tint_group.addAction(self.darken_tint_act)
        resize_menu.addAction(self.darken_tint_act)
        self.resize_btn.setMenu(resize_menu)
        self.resize_btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        draw_row.addWidget(self.resize_btn, 1)

        # Drawing subject is auto-driven by the detector choice
        # (see _on_detector_changed). YOLOE-vis / YOLOE-seg+boxes / SAM3
        # need prompt boxes (yellow); everything else routes drags to
        # the annotation bucket (green). No user-facing toggle.
        right_layout.addLayout(draw_row)

        # Store font size for use in toggle methods
        self._font_px = font

        # Edit-mode row: Edit Boxes (with undo/redo dropdown) + Save & Confirm
        edit_row = QtWidgets.QHBoxLayout()
        edit_row.setSpacing(BTN_GAP)

        self.edit_btn = QtWidgets.QToolButton()
        self.edit_btn.setText("Edit Boxes: OFF")
        self.edit_btn.setStyleSheet(tool_toggle_qss(TGL_EDIT_ON, font))
        self.edit_btn.setFixedHeight(int(btn_h * 0.8))
        self.edit_btn.setCheckable(True)
        self.edit_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.edit_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.edit_btn.setToolTip("Edit existing annotations. The dropdown picks the mode: "
                                 "Edit Boxes (drag/resize/delete) or Edit Drawn Masks "
                                 "(vertex/point editing). Undo/Redo are in the dropdown too.")
        self.edit_btn.toggled.connect(self._toggle_edit_btn)
        edit_menu = QtWidgets.QMenu()
        edit_menu.setToolTipsVisible(True)
        # The dropdown is a mode picker (mirrors the Draw button). Edit Boxes is
        # the default; picking either mode activates it (button turns ON) so the
        # button colour + label always show which edit mode is live.
        self.edit_tool_group = QtWidgets.QActionGroup(self)
        self.edit_tool_group.setExclusive(True)
        self.edit_tool_boxes_action = QtWidgets.QAction("Edit Boxes", self)
        self.edit_tool_boxes_action.setCheckable(True)
        self.edit_tool_boxes_action.setChecked(True)
        self.edit_tool_boxes_action.setToolTip("Drag handles to resize, drag bodies to move, "
                                               "click the X badge to delete.")
        self.edit_tool_boxes_action.triggered.connect(
            lambda: self._select_edit_tool("boxes", activate=True))
        self.edit_tool_group.addAction(self.edit_tool_boxes_action)
        edit_menu.addAction(self.edit_tool_boxes_action)
        # Edit Drawn Masks: greyed unless a SAM model is active AND at least one
        # semi-auto mask exists. Lets the user click a committed mask and re-edit
        # its SAM prompt points or polygon vertices.
        self.semiauto_edit_action = QtWidgets.QAction("Edit Masks", self)
        self.semiauto_edit_action.setCheckable(True)
        self.semiauto_edit_action.setEnabled(False)
        self.semiauto_edit_action.setToolTip(TOOLTIP_SEMIAUTO_EDIT)
        self.semiauto_edit_action.triggered.connect(
            lambda: self._select_edit_tool("masks", activate=True))
        self.edit_tool_group.addAction(self.semiauto_edit_action)
        edit_menu.addAction(self.semiauto_edit_action)
        edit_menu.addSeparator()
        undo_act = QtWidgets.QAction("Undo", self)
        undo_act.triggered.connect(self._undo_annotation)
        redo_act = QtWidgets.QAction("Redo", self)
        redo_act.triggered.connect(self._redo_annotation)
        edit_menu.addAction(undo_act)
        edit_menu.addAction(redo_act)
        self.edit_btn.setMenu(edit_menu)
        self.edit_btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        edit_row.addWidget(self.edit_btn, 1)

        # Select Multiple now lives on the Edit Boxes row (right of it):
        # dragging in empty space inside this mode draws a persistent
        # resizable marquee; every annotation it covers gets a cyan
        # border + its own X delete badge so the user can prune them one
        # at a time. Delete key still bulk-deletes the whole selection.
        # The marquee itself never becomes a saved box. Disabled until
        # an image is loaded (see _refresh_auto_annotate_enabled).
        self.multi_select_btn = QtWidgets.QPushButton("Select Multiple: OFF")
        self.multi_select_btn.setCheckable(True)
        self.multi_select_btn.setFixedHeight(int(btn_h * 0.8))
        self.multi_select_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.multi_select_btn.setStyleSheet(toggle_qss(TGL_EDIT_ON, font))
        self.multi_select_btn.setToolTip(
            "Drag a rectangle to select every box/segment that touches it. "
            "Each selected item shows a red X you can click to delete it; "
            "Delete key removes all selected at once; Esc dismisses the marquee."
        )
        self.multi_select_btn.setEnabled(False)  # gated on image-loaded
        self.multi_select_btn.toggled.connect(self._toggle_multi_select_mode)
        edit_row.addWidget(self.multi_select_btn, 1)
        right_layout.addLayout(edit_row)

        # Save & Confirm now gets its own full-width row beneath the
        # edit/multi-select row; the primary action stands alone so the
        # user can hit it without aiming.
        save_confirm_row = QtWidgets.QHBoxLayout()
        save_confirm_row.setSpacing(6)
        save_confirm_btn = QtWidgets.QPushButton("Save & Confirm")
        save_confirm_btn.setStyleSheet(btn_qss(BTN_GREEN, font))
        save_confirm_btn.setFixedHeight(int(btn_h * 0.8))
        save_confirm_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        save_confirm_btn.setToolTip("Save the current on-screen annotations (including your manual edits) as the final labels for this image - does not re-run the model.")
        save_confirm_btn.clicked.connect(lambda: lock_during(save_confirm_btn, self._save_and_confirm))
        save_confirm_row.addWidget(save_confirm_btn)
        right_layout.addLayout(save_confirm_row)

        # Bounding Box / Segmentation checkboxes are built in the left
        # column, directly above the confidence sliders (see init_ui).

        # Synthetic image generation (Stable Diffusion)
        # Collapsed by default behind one toggle to keep the canvas uncluttered;
        # the expanded/collapsed state persists across images (the panel is
        # built once and display_image never touches it). Prompt + negative
        # live in a popup (Edit Prompts); defaults are domain-agnostic.
        self._sd_prompt = _SD_DEFAULT_PROMPT
        self._sd_neg    = _SD_DEFAULT_NEG

        self.sd_toggle_btn = QtWidgets.QPushButton("Synthetic Images (Diffusion) ▸")
        self.sd_toggle_btn.setCheckable(True)
        self.sd_toggle_btn.setChecked(False)
        self.sd_toggle_btn.setStyleSheet(btn_qss(BTN_PURPLE, font))
        self.sd_toggle_btn.setFixedHeight(int(btn_h * 0.8))
        self.sd_toggle_btn.setToolTip(
            "Show/hide the Stable Diffusion synthetic-image controls. "
            "Stays where you leave it as you flip through images.")
        self.sd_toggle_btn.toggled.connect(self._toggle_sd_panel)
        right_layout.addWidget(self.sd_toggle_btn)

        self.sd_panel = QtWidgets.QWidget()
        sd_layout = QtWidgets.QVBoxLayout(self.sd_panel)
        sd_layout.setContentsMargins(0, 0, 0, 0)
        sd_layout.setSpacing(max(4, BTN_GAP - 2))

        # Prompt/negative popup launcher (text lives in SDPromptDialog).
        self.sd_edit_prompts_btn = QtWidgets.QPushButton("Edit Prompts")
        self.sd_edit_prompts_btn.setStyleSheet(btn_qss(BTN_BLUE, font))
        self.sd_edit_prompts_btn.setFixedHeight(int(btn_h * 0.8))
        self.sd_edit_prompts_btn.setToolTip(
            "Edit the SD prompt and negative prompt in a popup. Each can also "
            "load a .txt of tailored instructions (append or replace).")
        self.sd_edit_prompts_btn.clicked.connect(self._open_sd_prompts)
        sd_layout.addWidget(self.sd_edit_prompts_btn)

        # Generate buttons.
        synth_btn_row = QtWidgets.QHBoxLayout()
        synth_btn_row.setSpacing(BTN_GAP)
        self.gen_variation_btn = QtWidgets.QPushButton("Generate Variation")
        self.gen_variation_btn.setStyleSheet(btn_qss(BTN_PURPLE, font))
        self.gen_variation_btn.setFixedHeight(int(btn_h * 0.8))
        self.gen_variation_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gen_variation_btn.setToolTip(
            "Run Stable Diffusion on the current image, preserving the "
            "annotated regions and regenerating the background. Shows a "
            "side-by-side preview before saving to synthetic images/.")
        self.gen_variation_btn.setEnabled(False)
        self.gen_variation_btn.clicked.connect(self._on_generate_variation)

        self.gen_variation_folder_btn = QtWidgets.QPushButton("Variations for Folder")
        self.gen_variation_folder_btn.setStyleSheet(btn_qss(BTN_RED, font))
        self.gen_variation_folder_btn.setFixedHeight(int(btn_h * 0.8))
        self.gen_variation_folder_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gen_variation_folder_btn.setToolTip(
            "Generate a synthetic variation of every image in the folder "
            "that already has a saved label file. Opens a flipper viewer "
            "when finished so you can prune the bad results.")
        self.gen_variation_folder_btn.setEnabled(False)
        self.gen_variation_folder_btn.clicked.connect(self._on_generate_variations_batch)
        synth_btn_row.addWidget(self.gen_variation_btn)
        synth_btn_row.addWidget(self.gen_variation_folder_btn)
        sd_layout.addLayout(synth_btn_row)

        # Strength label + a deliberately short slider.
        sd_strength_label = QtWidgets.QLabel("Diffusion Strength: 0.20")
        sd_strength_label.setStyleSheet(lbl_style)
        sd_strength_label.setAlignment(QtCore.Qt.AlignCenter)
        sd_strength_label.setToolTip(
            "How much Stable Diffusion regenerates the background. Low = "
            "keep the original scene (most realistic, least varied); high "
            "= fully repaint from the prompt (more varied, risks unnatural "
            "fills). 0.40-0.70 is the realistic sweet spot.")
        sd_layout.addWidget(sd_strength_label)
        self.sd_strength_label = sd_strength_label

        sd_strength_row = QtWidgets.QHBoxLayout()
        sd_strength_row.setContentsMargins(0, 0, 0, 0)
        self.sd_strength_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sd_strength_slider.setRange(0, 100)
        self.sd_strength_slider.setValue(int(_SD_STRENGTH * 100))
        self.sd_strength_slider.setFixedHeight(30)
        self.sd_strength_slider.setMaximumWidth(300)
        self.sd_strength_slider.setStyleSheet(slider_qss())
        self.sd_strength_slider.valueChanged.connect(self._update_sd_strength_label)
        sd_strength_row.addStretch()
        sd_strength_row.addWidget(self.sd_strength_slider)
        sd_strength_row.addStretch()
        sd_layout.addLayout(sd_strength_row)
        self._update_sd_strength_label(self.sd_strength_slider.value())

        self.sd_panel.setVisible(False)
        right_layout.addWidget(self.sd_panel)

        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

        # Apply the detector-dependent radio gating ONCE at startup. The combo's
        # currentTextChanged is connected after addItems(), so _on_detector_changed
        # never fired for the initial DINO(SwinT) selection, and without this the box
        # radio looked enabled until the first detector switch.
        self._apply_prompt_radio_gating()
        self._refresh_prompt_entry_visibility()

    # Prompt-mode + pipeline preset wiring
    def _on_prompt_mode_changed(self, mode):
        """Toggle visibility of the text-prompt UI when switching modes."""
        self.prompt_mode = mode
        self._refresh_prompt_entry_visibility()
        # In Boxes mode, force box-draw on so the user can prompt by drawing.
        # Make sure the Draw button's active tool is "box" first (it may have
        # been switched to semi-auto), then activate it.
        if mode == "boxes":
            self._select_draw_tool("box")
            self._force_btn(self.draw_btn, True)
        self._sync_pipeline_for_prompt_mode()
        self._refresh_carry_checkbox_enabled()
        self._refresh_auto_annotate_enabled()
        # Drawn-box bucket depends on BOTH detector and prompt_mode, so re-evaluate.
        # reclassify=True: a prompt-mode flip is the one case where already-drawn
        # boxes should be re-tagged to the new bucket.
        self._refresh_draw_subject(reclassify=True)

    def _refresh_draw_subject(self, reclassify=False):
        """Recompute which bucket new drags go into. Reads detector_choice and
        prompt_mode directly so it can be called from any handler.

        Rule: drags go to the yellow PROMPT bucket only when the current detector
        actually consumes box prompts (YOLOE-vis always; SAM3/YOLOE-seg only in
        boxes mode). Otherwise drags land in the green ANNOTATION bucket."""
        text = getattr(self, "detector_choice", "")
        is_one_shot = ("YOLOE-seg" in text or "one-shot" in text
                       or text.startswith("SAM3 (") or "YOLOE-vis" in text)
        is_sam3_det = text.startswith("SAM3 (")
        needs_prompt_boxes = (
            ("YOLOE-vis" in text)
            or (is_sam3_det and self.prompt_mode == "boxes")
            or (is_one_shot and "YOLOE-vis" not in text and self.prompt_mode == "boxes")
        )
        if hasattr(self, "image_label"):
            self.image_label.set_draw_subject("prompt" if needs_prompt_boxes else "annotation")
            # Seamless switch ONLY on an explicit Text<->Boxes prompt-mode flip
            # (reclassify=True): boxes drawn before the flip get re-tagged to
            # the new bucket so a box drawn in Text mode becomes a yellow prompt
            # box in Boxes mode (and back) without a redraw.
            # A DETECTOR change must NOT reclassify: that would silently turn
            # saved manual (green) annotations into unsaved prompt (yellow)
            # boxes. Detector changes only update the draw subject above.
            if reclassify:
                self.image_label.reclassify_user_rects(needs_prompt_boxes)

    def _refresh_prompt_entry_visibility(self):
        """Single source of truth for the prompt-text entry's visibility.
        Visible iff prompt_mode == 'text' AND the detector accepts text
        (i.e. not YOLOE-vis, which is visual-prompts only). SAM3 standalone
        supports both modes, so it relies on prompt_mode alone."""
        is_text = (getattr(self, "prompt_mode", "text") == "text")
        is_yoloe_vis = "YOLOE-vis" in getattr(self, "detector_choice", "")
        visible = is_text and not is_yoloe_vis
        if hasattr(self, "prompt_entry"):
            self.prompt_entry.setVisible(visible)
        if hasattr(self, "prompt_label_widget"):
            self.prompt_label_widget.setVisible(visible)

    def _sync_pipeline_for_prompt_mode(self):
        """If current preset is incompatible with the prompt mode, snap to a sensible default."""
        cur = self.pipeline_combo.currentText()
        if self.prompt_mode == "boxes":
            # Box prompts are valid for YOLOE-vis, YOLOE-seg one-shot and
            # SAM3 standalone. Only DINO presets cannot be box-prompted, so
            # only snap away from a DINO preset.
            box_capable = ("YOLOE-vis" in cur or "YOLOE-seg" in cur
                           or "SAM3" in cur)
            if not box_capable:
                for i, p in enumerate(self.PIPELINE_PRESETS):
                    if "YOLOE-vis" in p:
                        self.pipeline_combo.setCurrentIndex(i)
                        break
        else:  # text
            # Any preset is text-compatible *except* YOLOE-vis (which needs box prompts).
            if "YOLOE-vis" in cur:
                self.pipeline_combo.setCurrentIndex(0)

    def _on_pipeline_changed(self, text):
        # Legacy hook, kept so the hidden self.pipeline_combo updates don't
        # crash if anything still calls this. New code uses _on_detector_changed.
        self.current_pipeline = text

    def _apply_detector_availability(self):
        """Gray out items whose checkpoint files aren't on disk.

        Covers both combos:
          - DETECTOR: SwinB needs groundingdino_swinb_cogcoor.pth
          - SEGMENTER: SAM3 needs sam3.pt in 'GUI and Pipeline/'
            (ultralytics 8.4.33 doesn't auto-download SAM3; its asset
            list omits 'sam3.pt' and build_sam3 does a raw open() that
            FileNotFounds. User must fetch sam3.pt manually from Meta's
            gated HF repo: https://huggingface.co/facebook/sam3)

        File-size floor (≥100 MB) catches partial / 'Not Found' stubs from
        bad curl redirects, not just literal missing files."""
        MIN_BYTES = 100 * 1024 * 1024  # 100 MB

        def _size(p):
            try:
                return os.path.getsize(p) if os.path.exists(p) else 0
            except OSError:
                return 0

        # Detector: SwinB
        swinb_path = os.path.join(GROUNDING_DINO_DIR, "weights", "groundingdino_swinb_cogcoor.pth")
        swinb_size = _size(swinb_path)
        swinb_ok = swinb_size >= MIN_BYTES
        for i in range(self.detector_combo.count()):
            text = self.detector_combo.itemText(i)
            item = self.detector_combo.model().item(i)
            if item is None:
                continue
            if "SwinB" in text and not swinb_ok:
                item.setEnabled(False)
                if swinb_size == 0:
                    tip = ("Checkpoint missing: groundingdino_swinb_cogcoor.pth "
                           "not in autoannotate study/GroundingDINO/weights/. "
                           "See README to download.")
                else:
                    tip = (f"Checkpoint looks broken: only {swinb_size:,} bytes "
                           f"(expected ~938 MB). Re-download with `curl -L`.")
                item.setToolTip(tip)

        # Segmenter: SAM3. load_sam resolves 'sam3.pt' against WEIGHTS_DIR,
        # so check the same location the loader will use.
        sam3_path = os.path.join(WEIGHTS_DIR, "sam3.pt")
        sam3_size = _size(sam3_path)
        sam3_ok = sam3_size >= MIN_BYTES
        for i in range(self.segmenter_combo.count()):
            text = self.segmenter_combo.itemText(i)
            item = self.segmenter_combo.model().item(i)
            if item is None:
                continue
            if "SAM3" in text and not sam3_ok:
                item.setEnabled(False)
                if sam3_size == 0:
                    tip = ("sam3.pt not found. Ultralytics 8.4.33 does NOT "
                           "auto-download SAM3, so fetch sam3.pt manually from "
                           "https://huggingface.co/facebook/sam3 (gated, accept "
                           "the license) and drop it in 'GUI and Pipeline/'. "
                           "Use SAM2 (tiny) until then.")
                else:
                    tip = (f"sam3.pt looks broken: only {sam3_size:,} bytes "
                           f"(expected hundreds of MB). Re-download from HF.")
                item.setToolTip(tip)

        # Detector: SAM3 (one-shot), same checkpoint requirement
        for i in range(self.detector_combo.count()):
            text = self.detector_combo.itemText(i)
            item = self.detector_combo.model().item(i)
            if item is None:
                continue
            if text.startswith("SAM3 (") and not sam3_ok:
                item.setEnabled(False)
                if sam3_size == 0:
                    tip = ("sam3.pt not found. Fetch from "
                           "https://huggingface.co/facebook/sam3 (gated) and "
                           "drop it in 'GUI and Pipeline/'.")
                else:
                    tip = (f"sam3.pt looks broken: only {sam3_size:,} bytes. "
                           "Re-download from HF.")
                item.setToolTip(tip)

    def _apply_prompt_radio_gating(self):
        """Enable/disable the Text/Boxes prompt radios for the current detector.
        Single source of truth so the gating is identical at startup and after
        every detector switch (otherwise the initial state, set before the
        combo's signal is connected, disagrees with the post-switch state).
          YOLOE-vis                 -> boxes only (visual prompts; no text)
          DINO (SwinT/SwinB)        -> text only  (text-grounded; no box input)
          SAM3 / YOLOE-seg one-shot -> both"""
        if not hasattr(self, "prompt_mode_boxes_btn"):
            return
        text = getattr(self, "detector_choice", "") or ""
        is_one_shot = self._is_one_shot_detector(text)
        is_yoloe_vis = "YOLOE-vis" in text
        is_dino = not is_one_shot
        text_ok  = not is_yoloe_vis
        boxes_ok = not is_dino
        self.prompt_mode_text_btn.setEnabled(text_ok)
        self.prompt_mode_boxes_btn.setEnabled(boxes_ok)
        if not text_ok and self.prompt_mode != "boxes":
            self.prompt_mode_boxes_btn.setChecked(True)
        elif not boxes_ok and self.prompt_mode != "text":
            self.prompt_mode_text_btn.setChecked(True)

    def _on_detector_changed(self, text):
        # Bulletproof switch: if this change would disable the semi-auto SAM
        # feature while it's in use, prompt Switch/Revert before applying.
        if not self._guard_model_change("detector", text):
            return  # reverted, combo restored, no change applied
        self.detector_choice = text
        # One-shot detectors (YOLOE-vis/-seg, SAM3) default to their OWN masks
        # ("(none)") but the segmenter stays SELECTABLE so the user can add
        # SAM2/SAM3 to re-segment the detected boxes (e.g. YOLOE-vis -> SAM3
        # masks across the folder). DINO always needs a real segmenter.
        is_one_shot = "YOLOE-seg" in text or "one-shot" in text or text.startswith("SAM3 (") or "YOLOE-vis" in text
        self.segmenter_combo.setEnabled(True)
        if is_one_shot:
            idx = self.segmenter_combo.findText("(none)")
            if idx >= 0:
                self.segmenter_combo.blockSignals(True)
                self.segmenter_combo.setCurrentIndex(idx)
                self.segmenter_combo.blockSignals(False)
            self.segmenter_choice = "(none)"
        else:
            if self.segmenter_combo.currentText() == "(none)":
                self.segmenter_combo.setCurrentIndex(0)
            self.segmenter_choice = self.segmenter_combo.currentText()
        # Gate the Text/Boxes radios to what the detector actually consumes.
        self._apply_prompt_radio_gating()
        # Prompt-entry visibility is now driven by a single helper that respects
        # both the detector type and the current prompt mode.
        self._refresh_prompt_entry_visibility()
        # Drags route to the prompt (yellow) bucket only when the detector
        # actually consumes box prompts; otherwise they go to the annotation
        # (green) bucket. SAM3 in text mode does NOT need prompt boxes.
        self._refresh_draw_subject()
        self._sync_pipeline_combo()
        self._refresh_carry_checkbox_enabled()
        self._refresh_auto_annotate_enabled()
        # A detector switch can drop interactive SAM (a YOLOE one-shot forces the
        # segmenter to (none)); grey out the Semi-Auto Points / Manual Masks /
        # Edit Drawn Masks tools and fall the Draw button back to Boxes when SAM
        # is no longer available. (The segmenter handler already does this.)
        self._refresh_mask_draw_enabled()
        # The effective confidence depends on the detector, so refresh the label.
        if hasattr(self, "detection_threshold_slider"):
            self._update_detection_threshold_label(self.detection_threshold_slider.value())
        # Drop any model the new pipeline no longer uses so only the active
        # detector/segmenter stay resident.
        self._offload_unused_models()

    def _on_segmenter_changed(self, text):
        # Bulletproof switch: guard a change that disables semi-auto SAM.
        if not self._guard_model_change("segmenter", text):
            return  # reverted, combo restored, no change applied
        self.segmenter_choice = text
        self._sync_pipeline_combo()
        self._refresh_mask_draw_enabled()
        # Free any model the new detector+segmenter pair no longer uses.
        self._offload_unused_models()

    def _sync_pipeline_combo(self):
        """Map (detector, segmenter) -> legacy pipeline string for back-compat."""
        det = self.detector_choice
        seg = self.segmenter_choice
        if det.startswith("SAM3 ("):
            legacy = "SAM3 standalone"
        elif "YOLOE-seg" in det or "one-shot" in det:
            legacy = "YOLOE-seg standalone"
        elif "YOLOE-vis" in det:
            legacy = "YOLOE-vis standalone"
        else:
            det_short = (
                "DINO (SwinT)" if "SwinT" in det else
                "DINO (SwinB)" if "SwinB" in det else
                "YOLOE-vis"
            )
            seg_short = "SAM3" if "SAM3" in seg else "SAM2"
            legacy = f"{det_short} + {seg_short}"
        self.current_pipeline = legacy
        if hasattr(self, "pipeline_combo"):
            idx = self.pipeline_combo.findText(legacy)
            if idx >= 0:
                self.pipeline_combo.blockSignals(True)
                self.pipeline_combo.setCurrentIndex(idx)
                self.pipeline_combo.blockSignals(False)

    # Approximate resident RAM per model key (GB). Consulted only when a soft
    # budget is set via AUTOANNOTATE_MODEL_BUDGET_GB; otherwise the cache is
    # unbounded (legacy). SAM3 dwarfs the rest, so on an 8GB box a budget keeps
    # it from co-residing with DINO+YOLOE and triggering the silent OOM crash.
    MODEL_FOOTPRINT_GB = {
        "dino_swint": 0.7, "dino_swinb": 0.9, "sam2_t": 0.16,
        "sam3": 3.3, "sam3_det": 3.3, "yoloe_vis": 0.8, "yoloe_seg": 0.8,
    }

    @property
    def DINO(self):
        """Back-compat lazy handle to DINO-SwinT. ManualWindow's own code calls
        _get_model("dino_swint") directly; this exists only for any external
        caller that still reads window.DINO. Loads on first access."""
        return self._get_model("dino_swint")

    def _model_budget_gb(self):
        """Soft resident-RAM budget (GB) for detection/segmentation models, read
        from AUTOANNOTATE_MODEL_BUDGET_GB. 0 / unset = unbounded: never evict
        (exactly the legacy behavior). Set e.g. 4.5 on an 8GB machine to stop
        SAM3 co-residing with other heavy models (see DESIGN_two_stage_yoloe_sam3.md)."""
        try:
            return float(os.environ.get("AUTOANNOTATE_MODEL_BUDGET_GB", "0") or 0)
        except (TypeError, ValueError):
            return 0.0

    def _resident_gb(self, extra_key=None):
        keys = set(self._model_cache)
        if extra_key:
            keys.add(extra_key)
        total = sum(self.MODEL_FOOTPRINT_GB.get(k, 0.5) for k in keys)
        # The SAM3 semantic predictor (run_sam3_text/boxes) is a SEPARATE multi-GB
        # instance living OUTSIDE _model_cache; count it when loaded, else the
        # budget under-counts ~3.3GB while SAM3 is the detector and eviction
        # won't fire when it should.
        if sam_module.sam3_semantic_loaded():
            total += self.MODEL_FOOTPRINT_GB.get("sam3", 3.3)
        return total

    def _evict_for(self, key):
        """Evict least-recently-used cached models until `key` fits the budget.
        Never evicts `key`. No-op when the budget is 0/unset, so default
        behavior is unchanged."""
        budget = self._model_budget_gb()
        if budget <= 0:
            return
        lru = getattr(self, "_model_lru", {})
        while self._resident_gb(key) > budget:
            victims = [k for k in self._model_cache if k != key]
            if not victims:
                break
            victim = min(victims, key=lambda k: lru.get(k, 0))
            if AUTOANNOTATE_DEBUG:
                print(f"[model-cache] evict {victim} to fit {key} "
                      f"(resident {self._resident_gb():.1f}GB > budget {budget:.1f}GB)")
            self._model_cache.pop(victim, None)
            lru.pop(victim, None)
            self._release_inference_memory(force=True)

    def _offload_unused_models(self):
        """After a detector/segmenter switch, free every cached model the NEW
        pipeline does not use, so only the active detector + segmenter (and the
        interactive SAM the semi-auto tool needs) stay resident. Lazy loaders
        reload on demand, so the only cost is a one-time reload if you switch
        back to a model you dropped. Set AUTOANNOTATE_KEEP_MODELS_WARM=1 to keep
        everything cached instead (faster switching, more memory)."""
        if getattr(self, "_busy", False):
            return
        if os.environ.get("AUTOANNOTATE_KEEP_MODELS_WARM", "0") not in ("0", "", "false", "False"):
            return
        try:
            det_key, seg_key, _ = self._detector_keys_for_pipeline()
        except Exception:
            return
        keep = {k for k in (det_key, seg_key, self._active_interactive_sam_key()) if k}
        cache = getattr(self, "_model_cache", None)
        evicted = []
        if isinstance(cache, dict):
            for k in list(cache):
                if k not in keep:
                    cache.pop(k, None)
                    if isinstance(getattr(self, "_model_lru", None), dict):
                        self._model_lru.pop(k, None)
                    evicted.append(k)
        released_sam3 = False
        if det_key != "sam3_det":
            _rel = globals().get("release_sam3_text_predictor")
            if _rel is not None:
                try:
                    released_sam3 = bool(_rel())
                except Exception:
                    pass
        if evicted or released_sam3:
            self._release_inference_memory()
            if AUTOANNOTATE_DEBUG:
                print(f"[model-cache] offloaded {evicted}"
                      f"{' + SAM3-text' if released_sam3 else ''} after switch; "
                      f"resident now {sorted(getattr(self, '_model_cache', {}) or {})}")

    def _get_model(self, key):
        """Lazy model loader with an optional memory budget. Keys: 'dino_swint',
        'dino_swinb', 'sam2_t', 'sam3', 'sam3_det', 'yoloe_vis', 'yoloe_seg'.
        Bumps a per-key LRU timestamp and, when AUTOANNOTATE_MODEL_BUDGET_GB is
        set, evicts least-recently-used models before loading a new one."""
        if not hasattr(self, "_model_lru"):
            self._model_lru = {}
        self._model_lru_tick = getattr(self, "_model_lru_tick", 0) + 1
        if key in self._model_cache:
            self._model_lru[key] = self._model_lru_tick
            return self._model_cache[key]
        # Make room before instantiating the new (possibly heavy) model.
        self._evict_for(key)
        if key == "dino_swint":
            m = load_dino_model('swint')
        elif key == "dino_swinb":
            m = load_dino_model('swinb')
        elif key == "sam2_t":
            m = load_sam("sam2_t")
        elif key == "sam3":
            m = load_sam("sam3")
        elif key == "sam3_det":
            # Same checkpoint as the segmenter; cache separately so that any
            # future per-key state (predictor mode, etc.) won't cross-pollute.
            m = load_sam("sam3")
        elif key == "yoloe_vis":
            if AUTOANNOTATE_DEBUG:
                print("[_get_model] yoloe_vis  -> yoloe-11l-seg.pt")
            m = load_yoloe("yoloe-11l-seg.pt")
        elif key == "yoloe_seg":
            if AUTOANNOTATE_DEBUG:
                print("[_get_model] yoloe_seg  -> yoloe-11l-seg.pt")
            m = load_yoloe("yoloe-11l-seg.pt")
        else:
            raise ValueError(f"Unknown model key: {key}")
        self._model_cache[key] = m
        self._model_lru[key] = self._model_lru_tick
        return m

    def _max_area_frac(self):
        """Fraction of the image area a single detection may cover before it is
        dropped as a spurious whole-/large-image match. Applied UNIFORMLY across
        every detector (DINO, YOLOE, SAM2/SAM3). Default 0.5 -- drops any box
        covering more than half the image (tuned for small-object fruit/berry
        folders). Raise it (e.g. 0.8-0.9) for genuinely large subjects, or lower
        it further (e.g. 0.3) for tiny objects /
        texture folders like fescue, where SAM3 box-exemplar can match a broad
        region whose box still slips under the loose 0.9 cutoff. From
        AUTOANNOTATE_MAX_AREA_FRAC; clamped to (0, 1]."""
        try:
            f = float(os.environ.get("AUTOANNOTATE_MAX_AREA_FRAC", "0.5"))
        except (TypeError, ValueError):
            f = 0.5
        return f if 0.0 < f <= 1.0 else 0.5

    def _yoloe_effective_conf(self, slider_conf):
        """Rescale the user-facing slider value (0..1) into YOLOE's practical
        confidence range. YOLOE-vis/-seg detections sit in roughly 0.0..0.30,
        so the raw slider default of 0.50 filters out everything useful. We
        map slider 0..1 -> 0.0..0.20 linearly (slider 25 -> ~0.05 which is
        the reference baseline; slider 50 -> 0.10; slider 100 -> 0.20).
        DINO and SAM pipelines call this *not* applied -- they use the raw
        slider since their score distributions are in the usual 0..1 range."""
        try:
            v = max(0.0, min(1.0, float(slider_conf)))
        except (TypeError, ValueError):
            v = 0.0
        return v * 0.20

    def _detector_uses_box_exemplars(self):
        """True only for detectors that RE-DETECT from carried example boxes:
        YOLOE-vis (always), and YOLOE-seg / SAM3 only in Boxes mode. DINO is
        text-driven (carry = the typed prompt), and SAM3/YOLOE-seg in Text mode
        carry text, so box-carry is disabled for those cases."""
        det_key, _, _ = self._detector_keys_for_pipeline()
        if det_key == "yoloe_vis":
            return True
        if det_key == "yoloe_seg" and getattr(self, "prompt_mode", "boxes") == "boxes":
            return True
        # SAM3 carries boxes by APPEARANCE (crop-composite) only in Boxes mode;
        # in Text mode it carries the typed prompt, so gate it on the mode just
        # like YOLOE-seg. (An unconditional True here re-ran crop-composite on a
        # stale carry anchor after a Boxes->Text switch, polluting text results.)
        if det_key == "sam3_det":
            return getattr(self, "prompt_mode", "boxes") == "boxes"
        return False

    def _on_carry_toggled(self, on):
        """Carry Prompts Forward toggled: update the label and re-evaluate the
        Auto Annotate Remaining button."""
        self.carry_forward_checkbox.setText(
            "Use First Image as Prompt: ON" if on else "Use First Image as Prompt: OFF")
        self._refresh_auto_annotate_enabled()

    def _refresh_carry_checkbox_enabled(self):
        """The carry toggle drives BOX-exemplar carry (this image's drawn boxes
        -> every remaining image). DINO is text-only with no box path, so its
        typed prompt already applies to every image and the toggle would do
        nothing: disable it for DINO so it does not imply otherwise. Box-capable
        detectors (YOLOE-vis/-seg, SAM3) keep it enabled."""
        if not hasattr(self, "carry_forward_checkbox"):
            return
        det_key, _, _ = self._detector_keys_for_pipeline()
        is_box_capable = det_key in ("yoloe_vis", "yoloe_seg", "sam3_det")
        self.carry_forward_checkbox.setEnabled(is_box_capable)
        self.carry_forward_checkbox.setToolTip(
            "Uses this image's drawn boxes as the box prompt for every other "
            "image when you run Auto Annotate Remaining."
            if is_box_capable else
            "DINO is text-only: your typed prompt already applies to every "
            "image, so there is nothing to carry forward here.")

    def _auto_annotate_available(self):
        """True when Auto Annotate Remaining has enough input to do real
        work. DINO needs prompt text; YOLOE-vis needs a drawn/carried box;
        YOLOE-seg and SAM3 accept either. Without that the run would just
        produce empty labels, so the button is greyed out instead."""
        # Output folder is requested on click (auto_annotate_remaining prompts
        # for it), so it does NOT gate the button -- only having images + the
        # right input for the current detector does.
        if not self.images:
            return False
        det_key, _, _ = self._detector_keys_for_pipeline()
        has_text  = bool(self.prompt_entry.text().strip()) if hasattr(self, "prompt_entry") else False
        has_boxes = bool(self.image_label.get_prompt_boxes_in_image_coords()
                         or getattr(self, "_carry_anchor", None))
        if det_key in ("dino_swint", "dino_swinb"):
            return has_text
        if det_key == "yoloe_vis":
            return has_boxes
        # yoloe_seg / sam3_det are one-shot detectors that can run from
        # either a drawn box exemplar or a text prompt.
        return has_boxes or has_text

    def _refresh_auto_annotate_enabled(self):
        """Grey out Auto Annotate Remaining + Select Multiple + the SD
        variation buttons whenever they could not do useful work for the
        current detector + input state. Select Multiple and the SD
        buttons only need an image to be loaded; Auto Annotate has its
        own _auto_annotate_available() rule."""
        if getattr(self, "_busy", False):
            return
        has_images = bool(getattr(self, "images", None))
        if hasattr(self, "auto_annotate_btn"):
            avail = self._auto_annotate_available()
            self.auto_annotate_btn.setEnabled(avail)
            # Never leave a greyed button a mystery -- say what's missing.
            if avail:
                self.auto_annotate_btn.setToolTip(
                    "Annotate every remaining image using this image's box "
                    "prompts / typed prompt as the reference.")
            elif not has_images:
                self.auto_annotate_btn.setToolTip("Select an image folder first.")
            else:
                _dk, _, _ = self._detector_keys_for_pipeline()
                if _dk in ("dino_swint", "dino_swinb"):
                    self.auto_annotate_btn.setToolTip(
                        "Enter a text prompt to enable Auto Annotate Remaining.")
                else:
                    self.auto_annotate_btn.setToolTip(
                        "Draw at least one box prompt (or enter a text prompt) "
                        "to enable Auto Annotate Remaining.")
        if hasattr(self, "multi_select_btn"):
            self.multi_select_btn.setEnabled(has_images)
            # End-of-folder cleanup: if the user finished the set while
            # multi-select was on, untoggle it so the next folder load
            # doesn't inherit a stuck marquee.
            if not has_images and self.multi_select_btn.isChecked():
                self.multi_select_btn.blockSignals(True)
                self.multi_select_btn.setChecked(False)
                self.multi_select_btn.blockSignals(False)
                self._apply_multi_select_btn_style(False)
                self.image_label.set_multi_select_mode(False)
        if hasattr(self, "gen_variation_btn"):
            self.gen_variation_btn.setEnabled(has_images)
        if hasattr(self, "gen_variation_folder_btn"):
            self.gen_variation_folder_btn.setEnabled(has_images)
        # Draw Mask depends on image-loaded + detector/segmenter choice + view.
        self._refresh_mask_draw_enabled()

    def _detector_keys_for_pipeline(self):
        """Return (detector_key, segmenter_key_or_None, is_standalone) from the
        live Detector + Segmenter dropdown choices. A one-shot detector
        (YOLOE-vis/-seg, SAM3) is STANDALONE only when the segmenter is
        "(none)" -- it then emits its own masks via _oneshot_polys_aligned.
        Picking SAM2/SAM3 turns it into a detect -> segment two-stage pipeline
        (e.g. YOLOE-vis boxes -> SAM3 masks). DINO always needs a segmenter."""
        det = getattr(self, "detector_choice", "") or ""
        seg = getattr(self, "segmenter_choice", "") or ""
        if det.startswith("DINO (SwinT)"):
            det_key = "dino_swint"
        elif det.startswith("DINO (SwinB)"):
            det_key = "dino_swinb"
        elif det.startswith("SAM3 ("):
            det_key = "sam3_det"
        elif "YOLOE-vis" in det:
            det_key = "yoloe_vis"
        elif "YOLOE-seg" in det:
            det_key = "yoloe_seg"
        else:
            det_key = "dino_swint"
        if "SAM3" in seg:
            seg_key = "sam3"
        elif "SAM2" in seg:
            seg_key = "sam2_t"
        else:
            seg_key = None  # "(none)"
        if det_key in ("yoloe_vis", "yoloe_seg", "sam3_det"):
            return det_key, seg_key, (seg_key is None)
        return det_key, (seg_key or "sam2_t"), False

    def _model_tag(self):
        """Filesystem-safe tag for the live pipeline, e.g. 'SwinB_SAM2' (two
        stage) or 'SAM3' (one-shot). Names the annotated_<tag> output folder so
        outputs from different models aren't confused for one another."""
        det_key, seg_key, _ = self._detector_keys_for_pipeline()
        det_name = {"dino_swint": "SwinT", "dino_swinb": "SwinB",
                    "sam3_det": "SAM3", "yoloe_vis": "YOLOEvis",
                    "yoloe_seg": "YOLOEseg"}.get(det_key, det_key)
        seg_name = {"sam2_t": "SAM2", "sam3": "SAM3"}.get(seg_key)
        return det_name + (f"_{seg_name}" if seg_name else "")

    # Detection / segmentation dispatch
    def _run_sam3_crop_composite(self, image_path, ref, conf):
        """SAM3 box-prompt carry by APPEARANCE (crop compositing).

        Paste the carried image-1 box crops into a compact top-left block of
        THIS image, box-prompt SAM3 at those patches, then drop detections that
        land inside the patches. SAM3 then segments the same-LOOKING objects
        ANYWHERE in the image -- no coordinate dependence (verified ~93% berry
        precision vs ~1% for coordinate carry). Returns (boxes_xyxy, polys_norm,
        results); each box hugs its segmentation."""
        crops = (ref or {}).get("crops") or []
        img = self._imread_cached(image_path)
        if img is None or not crops:
            return [], [], None
        ih, iw = img.shape[:2]
        comp = img.copy()
        patches = []
        x = y = row_h = 0
        maxw = max(2, iw // 2)
        for crop in crops:
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            chh, cww = crop.shape[:2]
            if cww < 2 or chh < 2 or cww > maxw or chh > ih:
                continue
            if x + cww > maxw:
                x = 0; y += row_h + 2; row_h = 0
            if y + chh > ih:
                break
            comp[y:y + chh, x:x + cww] = crop
            patches.append([x, y, x + cww, y + chh])
            x += cww + 2
            row_h = max(row_h, chh)
        if not patches:
            return [], [], None
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        cv2.imwrite(tmp, comp)
        try:
            _, results = run_sam3_boxes(tmp, patches, conf=conf, max_area_frac=self._max_area_frac())
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        r = results[0] if results else None
        boxes, polys, keep = [], [], []
        if r is not None and r.masks is not None and r.boxes is not None:
            max_area = ih * iw * self._max_area_frac()
            seg_list = result_clean_polys(r)
            PATCH_MARGIN = 8   # px; also catch detections hugging a patch edge
            def _on_patch(bx):
                # True if box `bx` is a pasted crop OR a detection the crop
                # seeded. The crop region was painted over THIS image, so a box
                # there is never real content. Drop when the box is centered in a
                # patch, lies mostly (>40%) inside a patch, OR covers most (>50%)
                # of a patch even while sprawling much larger -- the oversized
                # "bled" masks the earlier box-fraction-only test let through.
                bx1, by1, bx2, by2 = bx
                ba = max(1.0, (bx2 - bx1) * (by2 - by1))
                bcx = (bx1 + bx2) / 2.0; bcy = (by1 + by2) / 2.0
                for qx1, qy1, qx2, qy2 in patches:
                    px1 = qx1 - PATCH_MARGIN; py1 = qy1 - PATCH_MARGIN
                    px2 = qx2 + PATCH_MARGIN; py2 = qy2 + PATCH_MARGIN
                    if px1 <= bcx <= px2 and py1 <= bcy <= py2:
                        return True
                    ix1 = max(bx1, px1); iy1 = max(by1, py1)
                    ix2 = min(bx2, px2); iy2 = min(by2, py2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if inter <= 0:
                        continue
                    pa = max(1.0, (px2 - px1) * (py2 - py1))
                    if inter / ba > 0.4 or inter / pa > 0.5:
                        return True
                return False
            for i, (box, seg) in enumerate(zip(r.boxes.xyxy.tolist(), seg_list)):
                if (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
                    continue
                if seg is None or len(seg) < 3:
                    continue
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                seg_box = [min(xs) * iw, min(ys) * ih, max(xs) * iw, max(ys) * ih]
                # Drop detections sitting on / seeded by a pasted exemplar crop.
                # Test the raw SAM box AND the saved polygon-derived box.
                if _on_patch(box) or _on_patch(seg_box):
                    continue
                boxes.append(seg_box)
                polys.append(seg)
                keep.append(i)
        # Diagnostic: the patch block is pasted top-left; kept boxes far from it
        # are real look-alike detections, not patch leaks. Helps tell a filter
        # leak apart from a SAM3 false positive when "burned in" boxes appear.
        print(f"[SAM3 carry] patch block (top-left)="
              f"{[[int(v) for v in p] for p in patches]} -> kept {len(keep)} "
              f"box(es) at {[[int(v) for v in b] for b in boxes]}")
        if not keep:
            return [], [], None
        # FILTER the results to kept detections ONLY, so the saved masks/overlay
        # never include the pasted-crop patches (which were "burning in" at the
        # top-left of every segments image).
        try:
            filtered = results[0][keep]
        except Exception:
            filtered = None
        return boxes, polys, ([filtered] if filtered is not None else None)

    def _run_sam3_boxes_partitioned(self, image_path, exemplars_xyxy, conf, text_prompt, supplementary_exemplars=None):
        """SAM3 box-exemplar mode with ROI partitioning.

        A drawn box that covers a large fraction of the image is usually
        the user trying to bound a SEARCH AREA, not give an exemplar.
        Feeding it as a normal exemplar contaminates the similarity
        embedding (one big background patch dominates the others), which
        is what destroyed the demo: a fresh huge box dragged across empty
        space wiped out the prior good detections in favor of garbage
        clustered around the new box.

        So: any exemplar covering > HUGE_FRAC of the image area becomes
        a Region Of Interest. The image is cropped to that box and SAM3
        runs INSIDE the crop using the remaining (normal-sized) exemplars
        clipped to crop coords -- or, if there are none, the current
        text prompt as fallback. Result boxes/polys are translated back
        to image coordinates. Normal-sized exemplars also drive a single
        global pass over the whole image so detections outside any ROI
        still happen.

        Returns (aligned_boxes_xyxy, aligned_polys_norm, results_or_None).
        `results` is whatever the global pass returned (used by callers
        that save raw masks); the polys list is the canonical output.
        """
        import tempfile
        HUGE_FRAC = 0.15  # >15% of image area => treat as ROI, not exemplar
        img = self._imread_cached(image_path)
        if img is None:
            return [], [], None
        ih, iw = img.shape[:2]
        img_area = float(ih * iw) or 1.0
        max_area = img_area * self._max_area_frac()  # drop whole-image detections (same filter as run_sam3_*)
        huge, normal = [], []
        for b in exemplars_xyxy:
            x1, y1, x2, y2 = b
            if (x2 - x1) * (y2 - y1) > img_area * HUGE_FRAC:
                huge.append([float(x1), float(y1), float(x2), float(y2)])
            else:
                normal.append([float(x1), float(y1), float(x2), float(y2)])

        global_boxes, global_polys, global_results = [], [], None
        # Anchor the global exemplar pool to prior detector output too.
        # Without this, a single newly-drawn box dominates the similarity
        # embedding (one bad/noisy draw wipes out the prior good masks).
        # Sampled, not concatenated wholesale, so a busy image doesn't
        # blow up the predictor's input size.
        anchors = list(supplementary_exemplars or [])
        if len(anchors) > 12:
            step = len(anchors) / 12.0
            anchors = [anchors[int(i * step)] for i in range(12)]
        global_pool = list(normal) + anchors
        # Global pass: run whenever there is ANY exemplar in the pool
        # (drawn or prior-detection anchor). With prior anchors present,
        # even a no-fresh-normal regenerate still re-detects the existing
        # objects -- the user's intent on Regenerate isn't to start over.
        if global_pool:
            boxes_list, results = run_sam3_boxes(
                image_path, global_pool, conf=conf, max_area_frac=self._max_area_frac(),
            )
            global_results = results
            r = results[0] if results else None
            if r is not None and r.masks is not None and r.boxes is not None:
                xyxy_list = r.boxes.xyxy.tolist()
                xyn_list = result_clean_polys(r)
                for box, seg in zip(xyxy_list, xyn_list):
                    if (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
                        continue
                    if seg is None or len(seg) < 3:
                        continue
                    global_boxes.append(box)
                    global_polys.append(seg)

        # ROI pass per huge box. Crop, predict, translate back.
        for hb in huge:
            hx1, hy1, hx2, hy2 = [int(round(v)) for v in hb]
            hx1 = max(0, min(iw - 1, hx1)); hx2 = max(0, min(iw, hx2))
            hy1 = max(0, min(ih - 1, hy1)); hy2 = max(0, min(ih, hy2))
            if hx2 - hx1 < 16 or hy2 - hy1 < 16:
                continue
            crop = img[hy1:hy2, hx1:hx2]
            ch, cw = crop.shape[:2]
            # Translate normal exemplars into crop coords, keep those that
            # actually fall inside the ROI (clipped to it).
            crop_exemplars = []
            # Prefer drawn normals as exemplars (the user's most recent
            # intent); fall back to prior-detection anchors that fall
            # inside the ROI when there are no drawn normals.
            for src_box in list(normal) + anchors:
                ax1, ay1, ax2, ay2 = src_box
                cx1 = max(0.0, min(float(cw), ax1 - hx1))
                cy1 = max(0.0, min(float(ch), ay1 - hy1))
                cx2 = max(0.0, min(float(cw), ax2 - hx1))
                cy2 = max(0.0, min(float(ch), ay2 - hy1))
                if cx2 - cx1 >= 4 and cy2 - cy1 >= 4:
                    crop_exemplars.append([cx1, cy1, cx2, cy2])
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            try:
                tmp.close()
                cv2.imwrite(tmp.name, crop)
                if crop_exemplars:
                    _, results = run_sam3_boxes(
                        tmp.name, crop_exemplars, conf=conf, max_area_frac=self._max_area_frac(),
                    )
                elif text_prompt and text_prompt.strip():
                    _, results = run_sam3_text(
                        tmp.name, text_prompt, conf=conf, max_area_frac=self._max_area_frac(),
                    )
                else:
                    # No normal exemplar AND no text prompt -- nothing we
                    # can sensibly search for inside this ROI. Skip rather
                    # than fall back to using the huge box itself.
                    print(f"[SAM3 ROI] skipping huge box {hb}: no exemplar/text to drive a search inside it")
                    continue
            finally:
                try:
                    import os as _os
                    _os.unlink(tmp.name)
                except OSError:
                    pass
            r = results[0] if results else None
            if r is None or r.masks is None or r.boxes is None:
                continue
            crop_max_area = ch * cw * self._max_area_frac()
            xyxy_list = r.boxes.xyxy.tolist()
            xyn_list = result_clean_polys(r)
            for box, seg in zip(xyxy_list, xyn_list):
                if (box[2] - box[0]) * (box[3] - box[1]) >= crop_max_area:
                    continue
                if seg is None or len(seg) < 3:
                    continue
                # Translate box back to image coords.
                gb = [box[0] + hx1, box[1] + hy1, box[2] + hx1, box[3] + hy1]
                if (gb[2] - gb[0]) * (gb[3] - gb[1]) >= max_area:
                    continue
                # Translate polygon (normalized to crop) back to image-norm.
                gseg = []
                for px, py in seg:
                    gx = (px * cw + hx1) / iw
                    gy = (py * ch + hy1) / ih
                    gseg.append([gx, gy])
                global_boxes.append(gb)
                global_polys.append(gseg)

        # Dedupe across global + ROI contributions (same object can be
        # picked up by both a global pass and an ROI pass that overlaps it).
        global_boxes, global_polys = self._nms_dedup(global_boxes, global_polys)
        return global_boxes, global_polys, global_results

    def _run_detector(self, image_path, prompt, det_thresh, mask_thresh, prompt_boxes_img, ref=None):
        """Run the detector portion of the current pipeline.
        det_thresh = detection threshold (0..1); mask_thresh = mask-confidence
        knob (0..1). prompt_boxes_img = absolute-xyxy boxes drawn as PROMPTS
        (yellow), used by YOLOE-vis and YOLOE-seg-with-boxes only.
        ref = optional Carry-Prompts-Forward bundle from _collect_box_prompt_crops
        ({'image_path','boxes_xyxy','crops'}). When present, YOLOE uses it as a
        TRUE one-shot refer_image; SAM3/DINO have no visual-crop path so they
        warn and fall back to the carried text/box prompts.
        Returns (absolute_xyxy_boxes, optional_yoloe_seg_results)."""
        det_key, _, is_standalone = self._detector_keys_for_pipeline()
        self._oneshot_polys_aligned = None
        if det_key == "dino_swint" or det_key == "dino_swinb":
            if ref and ref.get("crops"):
                print(f"[carry] GroundingDINO has no visual-prompt path -- using "
                      f"carried label text only ({len(ref['crops'])} ref crop(s) skipped).")
            model = self._get_model(det_key)
            # mask_thresh feeds DINO text_threshold (class-match strictness).
            text_thr = max(0.05, mask_thresh)
            boxes = run_dino_from_model(
                model, image_path, prompt, det_thresh, text_thr, self._max_area_frac(),
                save_dir=os.path.join(self.output_folder, 'boxes'),
            )
            # Drop duplicate boxes; one caption can match the same object
            # under two phrases ("blueberry" and "blueberry cluster").
            boxes, _ = self._nms_dedup(boxes)
            return boxes, None
        if det_key == "yoloe_vis":
            ref_boxes = ref.get("boxes_xyxy") if ref else None
            if ref_boxes:
                # True one-shot: ref bboxes live in ref['image_path']; YOLOE
                # learns their look and finds similar objects in THIS image.
                model = self._get_model(det_key)
                visual_prompts = dict(
                    bboxes=np.array(ref_boxes, dtype=np.float32),
                    cls=np.zeros(len(ref_boxes), dtype=np.int32),
                )
                try:
                    _, results = run_yoloe_vis(model, image_path, visual_prompts,
                                               conf=self._yoloe_effective_conf(det_thresh),
                                               max_area_frac=self._max_area_frac(), refer_image=ref["image_path"])
                except Exception as _e:
                    # refer_image one-shot can fail under memory pressure / stale
                    # model state; fall back to plain box-coordinate visual prompts
                    # so the run keeps going instead of dying on every image.
                    print(f"[carry] YOLOE refer_image one-shot failed "
                          f"({type(_e).__name__}: {_e}); falling back to box-coordinate prompts")
                    _, results = run_yoloe_vis(model, image_path, visual_prompts,
                                               conf=self._yoloe_effective_conf(det_thresh),
                                               max_area_frac=self._max_area_frac())
            else:
                if not prompt_boxes_img:
                    self._oneshot_polys_aligned = []
                    return [], None
                model = self._get_model(det_key)
                visual_prompts = dict(
                    bboxes=np.array(prompt_boxes_img, dtype=np.float32),
                    cls=np.zeros(len(prompt_boxes_img), dtype=np.int32),
                )
                _, results = run_yoloe_vis(model, image_path, visual_prompts, conf=self._yoloe_effective_conf(det_thresh), max_area_frac=self._max_area_frac())
            # YOLOE-vis is one-shot; extract both boxes and masks from the
            # raw results so segmentation-mask display mode works. Apply the
            # same area filter run_yoloe_vis would have, in lockstep with
            # the masks, so boxes and polys stay aligned.
            aligned_boxes = []
            aligned_polys = []
            r = results[0] if results else None
            if r is not None and r.boxes is not None:
                ih, iw = r.orig_shape[:2]
                max_area = ih * iw * self._max_area_frac()
                xyxy_list = r.boxes.xyxy.tolist()
                xyn_list  = result_clean_polys(r) if r.masks is not None else []
                for idx, box in enumerate(xyxy_list):
                    if (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
                        continue
                    aligned_boxes.append(box)
                    seg = xyn_list[idx] if idx < len(xyn_list) else None
                    if seg is not None and len(seg) >= 3:
                        aligned_polys.append(seg)
                    else:
                        aligned_polys.append(None)
            # If we have a poly hole, drop the parallel box too so alignment
            # holds (display_masks_with_borders treats len(det_polys) as the
            # detector-portion count). Cheap second pass; det count is tiny.
            kept_boxes, kept_polys = [], []
            for b, p in zip(aligned_boxes, aligned_polys):
                if p is None:
                    continue
                kept_boxes.append(b)
                kept_polys.append(p)
            # Drop duplicate detections of the same object (multi-class /
            # multi-concept prompt echoing one object several times).
            kept_boxes, kept_polys = self._nms_dedup(kept_boxes, kept_polys)
            self._oneshot_polys_aligned = kept_polys
            return kept_boxes, results
        if is_standalone and det_key == "sam3_det":
            if ref is not None and ref.get("crops"):
                # Box-prompt carry by APPEARANCE (not coordinates): composite the
                # image-1 box crops into a corner of THIS image, box-prompt SAM3
                # at those patches, and drop detections that land on a patch.
                # SAM3 then finds the same-looking objects ANYWHERE in the image.
                ab, ap, results = self._run_sam3_crop_composite(
                    image_path, ref, det_thresh)
                self._oneshot_polys_aligned = ap
                print(f"[carry] SAM3 crop-composite ({len(ref['crops'])} exemplar"
                      f"(s)) -> {len(ab)} detections")
                return ab, results
            # A drawn/carried box is always a prompt: if boxes are present,
            # box-prompt SAM3 even when the Text radio is selected. Pure
            # text mode applies only when there are no boxes (fixes
            # carry-forward silently doing nothing for SAM3 one-shot).
            if self.prompt_mode == "text" and not prompt_boxes_img:
                # SAM3 open-vocabulary text mode. Goes through SAM3SemanticPredictor
                # (a separate model variant from the interactive box-prompt SAM3).
                if not prompt or not prompt.strip():
                    self._oneshot_polys_aligned = []
                    return [], None
                boxes_list, results = run_sam3_text(
                    image_path, prompt, conf=det_thresh, max_area_frac=self._max_area_frac(),
                )
                r = results[0] if results else None
                aligned_boxes = []
                aligned_polys = []
                if r is not None and r.masks is not None and r.boxes is not None:
                    ih, iw = r.orig_shape[:2]
                    max_area = ih * iw * self._max_area_frac()
                    xyxy_list = r.boxes.xyxy.tolist()
                    xyn_list  = result_clean_polys(r)
                    for box, seg in zip(xyxy_list, xyn_list):
                        if (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
                            continue
                        if seg is None or len(seg) < 3:
                            continue
                        aligned_boxes.append(box)
                        aligned_polys.append(seg)
                aligned_boxes, aligned_polys = self._nms_dedup(aligned_boxes, aligned_polys)
                self._oneshot_polys_aligned = aligned_polys
                return aligned_boxes, results
            # Boxes mode, SAM3 SEMANTIC predictor: the drawn/carried boxes
            # are EXAMPLES; SAM3 finds & segments OTHER similar objects across
            # the image (exemplar re-detection), not the fixed box regions.
            # ROI partition: huge exemplars (> HUGE_FRAC of image) are treated
            # as search regions instead of exemplars so one bad/empty draw
            # cannot contaminate the global similarity search. Prior
            # detector outputs are also fed in as anchor exemplars (see below)
            # so even a regen with no fresh drawn box still re-detects the
            # existing objects. We only bail out if there is literally nothing
            # to drive a search with.
            if not prompt_boxes_img and not self.image_label.get_active_annotations():
                # No drawn box, no prior detections -- nothing to seed
                # the similarity search with.
                self._oneshot_polys_aligned = []
                return [], None
            # Pull prior detector-source annotations (rects + polys) and
            # convert to image-pixel xyxy. These anchor the SAM3 exemplar
            # pool so a single new bad draw cannot dominate the embedding
            # and wipe out the prior good masks (the demo bug).
            prior_anchors = []
            ow = self.image_label._orig_w or 0
            oh = self.image_label._orig_h or 0
            if ow and oh:
                for ann in self.image_label.annotations:
                    if ann.get('deleted') or ann.get('source') != 'detector':
                        continue
                    if ann['type'] == 'rect':
                        cx, cy, w, h = ann['data']
                        x1 = (cx - w / 2) * ow
                        y1 = (cy - h / 2) * oh
                        x2 = (cx + w / 2) * ow
                        y2 = (cy + h / 2) * oh
                    else:  # poly
                        xs = [p[0] for p in ann['data']]
                        ys = [p[1] for p in ann['data']]
                        if not xs or not ys:
                            continue
                        x1 = min(xs) * ow; x2 = max(xs) * ow
                        y1 = min(ys) * oh; y2 = max(ys) * oh
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
                        prior_anchors.append([x1, y1, x2, y2])
            # Image-only prompting: in BOX prompt mode the detector is driven
            # purely by box exemplars, so do NOT leak any leftover prompt-entry
            # text into the SAM3 search. Text only applies in Text prompt mode.
            text_for_sam3 = prompt if self.prompt_mode == "text" else ""
            aligned_boxes, aligned_polys, results = self._run_sam3_boxes_partitioned(
                image_path, prompt_boxes_img, det_thresh, text_for_sam3,
                supplementary_exemplars=prior_anchors,
            )
            # Create the box AROUND each segmentation: a tight bbox of the SAM3
            # mask polygon, so the saved box hugs the segmentation exactly.
            aligned_boxes = self._boxes_from_seg_polys(
                aligned_polys, image_path, fallback=aligned_boxes)
            self._oneshot_polys_aligned = aligned_polys
            return aligned_boxes, results
        if det_key == "yoloe_seg":  # YOLOE-seg detector: standalone one-shot OR two-stage feeding a separate segmenter
            model = self._get_model("yoloe_seg")
            # Standalone one-shot uses mask_thresh as a "make it stricter" knob
            # on the segmenter pass; two-stage (boxes -> SAM2/SAM3) detects at
            # det_thresh like YOLOE-vis, since there mask_thresh is the segmenter knob.
            eff_conf = self._yoloe_effective_conf(
                max(det_thresh, mask_thresh) if is_standalone else det_thresh)
            # Boxes present (drawn or carried) => visual-prompt regardless of
            # the Text/Boxes radio; fall back to text only when no boxes.
            ref_boxes = ref.get("boxes_xyxy") if ref else None
            if ref_boxes:
                # True one-shot from the carried Visual Reference image.
                visual_prompts = dict(
                    bboxes=np.array(ref_boxes, dtype=np.float32),
                    cls=np.zeros(len(ref_boxes), dtype=np.int32),
                )
                try:
                    _, results = run_yoloe_vis(model, image_path, visual_prompts,
                                               conf=eff_conf, max_area_frac=self._max_area_frac(),
                                               refer_image=ref["image_path"])
                except Exception as _e:
                    print(f"[carry] YOLOE-seg refer_image one-shot failed "
                          f"({type(_e).__name__}: {_e}); falling back to box-coordinate prompts")
                    _, results = run_yoloe_vis(model, image_path, visual_prompts,
                                               conf=eff_conf, max_area_frac=self._max_area_frac())
            elif prompt_boxes_img:
                visual_prompts = dict(
                    bboxes=np.array(prompt_boxes_img, dtype=np.float32),
                    cls=np.zeros(len(prompt_boxes_img), dtype=np.int32),
                )
                _, results = run_yoloe_vis(model, image_path, visual_prompts,
                                           conf=eff_conf, max_area_frac=self._max_area_frac())
            else:
                _, results = run_yoloe_text(model, image_path, prompt,
                                            conf=eff_conf, max_area_frac=self._max_area_frac())
            # Re-derive boxes IN LOCKSTEP with masks so live_boxes and
            # live_polys_cache stay index-aligned. The helper run_yoloe_*
            # filters by max_area only and discards masks; here we filter
            # both together by (area < max) AND (mask has >=3 polygon points).
            if not results:
                # Empty prompt (no boxes + no text) -> run_yoloe_text returns
                # None; nothing to detect on this image rather than crash.
                self._oneshot_polys_aligned = []
                return [], None
            r = results[0]
            if r.masks is None or r.boxes is None:
                return [], None
            ih, iw = r.orig_shape[:2]
            max_area = ih * iw * self._max_area_frac()
            xyxy_list = r.boxes.xyxy.tolist()
            xyn_list  = result_clean_polys(r)
            aligned_boxes = []
            aligned_polys = []
            for box, seg in zip(xyxy_list, xyn_list):
                if (box[2] - box[0]) * (box[3] - box[1]) >= max_area:
                    continue
                if seg is None or len(seg) < 3:
                    continue
                aligned_boxes.append(box)
                aligned_polys.append(seg)
            # Drop duplicate detections of the same object before stashing.
            aligned_boxes, aligned_polys = self._nms_dedup(aligned_boxes, aligned_polys)
            # Stash aligned polys on self so the display_* methods can pick
            # them up without re-doing the filter. Cleared on every run.
            # Standalone keeps YOLOE's own masks; in a two-stage run the boxes
            # go to SAM2/SAM3, which produces the masks, so clear these.
            self._oneshot_polys_aligned = aligned_polys if is_standalone else None
            return aligned_boxes, results
        self._oneshot_polys_aligned = None
        return [], None

    @staticmethod
    def _box_iou(a, b):
        """IoU of two absolute or normalized xyxy boxes."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        u = ua + ub - inter
        return inter / u if u > 0 else 0.0

    def _dedup_anns(self, anns, iou_thresh=0.7):
        """Final overlap cleanup on a finished annotation list: drop any
        annotation whose bounding box overlaps an already-kept one by more
        than iou_thresh, so the displayed/saved set does not pile up
        near-duplicate boxes and masks across regenerates.

        Manual (user-drawn) annotations are considered first, so when a manual
        box and a detector box overlap, the user's box is the one kept."""
        ordered = ([a for a in anns if a.get('source') == 'manual']
                   + [a for a in anns if a.get('source') != 'manual'])
        kept, kept_bb = [], []
        for a in ordered:
            bb = self._ann_bbox_norm(a)
            if any(self._box_iou(bb, kb) > iou_thresh for kb in kept_bb):
                continue
            kept.append(a)
            kept_bb.append(bb)
        return kept

    @staticmethod
    def _nms_dedup(boxes, polys=None, iou_thresh=0.7):
        """Drop near-identical detections, the same physical object found
        more than once (e.g. an object that matches two comma-separated
        prompt terms like 'blueberry, blueberry cluster'). Keeps the FIRST
        box of any group whose mutual IoU exceeds iou_thresh; `polys`, if
        given, is shrunk in lockstep so masks stay index-aligned. Distinct
        objects only touch at much lower IoU, so they are kept."""
        def _iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            u = ua + ub - inter
            return inter / u if u > 0 else 0.0
        kept_boxes = []
        kept_polys = []
        for i, b in enumerate(boxes):
            if any(_iou(b, kb) > iou_thresh for kb in kept_boxes):
                continue
            kept_boxes.append(b)
            if polys is not None:
                kept_polys.append(polys[i] if i < len(polys) else None)
        return kept_boxes, (kept_polys if polys is not None else None)

    def _collect_rejected(self):
        """Record the bounding boxes of any currently soft-deleted
        annotations into a per-image reject list, so a later regenerate
        does not re-add an object the user explicitly removed. Reset on
        every new image by display_image."""
        if not hasattr(self, "_rejected_boxes"):
            self._rejected_boxes = []
        for ann in self.image_label.annotations:
            if ann.get('deleted'):
                self._rejected_boxes.append(self._ann_bbox_norm(ann))

    def _drop_rejected(self, boxes, polys=None):
        """Drop detector outputs that RE-COVER a region the user deleted on this
        image, so a regenerate does not re-add an object they removed. `boxes`
        are absolute xyxy; the reject list is normalized. Manual draws never pass
        through here, so they are never dropped.

        SIZE-AWARE: a detection is dropped only when it substantially re-covers a
        deleted box -- i.e. it is the SAME object (high IoU, or the two boxes
        mostly coincide). A SMALLER detection merely NESTED inside a larger
        deleted region (e.g. an individual berry inside a deleted cluster mask)
        is NOT a re-detection and is KEPT. This is the fix for "deleting the big
        cluster mask wipes the smaller model masks within it": the old test
        dropped anything whose CENTER fell inside a deleted box, which nuked
        every small berry sitting inside a big deleted cluster box."""
        rejected = getattr(self, "_rejected_boxes", None)
        if not rejected or not boxes:
            return boxes, polys
        ow = self.image_label._orig_w or 1
        oh = self.image_label._orig_h or 1
        rej_px = [[r[0] * ow, r[1] * oh, r[2] * ow, r[3] * oh] for r in rejected]
        def _redetects(b, rb):
            ix1 = max(b[0], rb[0]); iy1 = max(b[1], rb[1])
            ix2 = min(b[2], rb[2]); iy2 = min(b[3], rb[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return False
            ab  = max(0.0, b[2] - b[0])  * max(0.0, b[3] - b[1])
            arb = max(0.0, rb[2] - rb[0]) * max(0.0, rb[3] - rb[1])
            if ab <= 0 or arb <= 0:
                return False
            iou       = inter / (ab + arb - inter)
            cover_new = inter / ab    # fraction of the NEW box inside the deleted region
            cover_rej = inter / arb   # fraction of the DELETED region the new box covers
            # Same object the user removed: a clean IoU match, OR the two boxes
            # mostly coincide. Requiring cover_rej (not just cover_new) to be high
            # is what spares a small berry nested in a big deleted cluster -- its
            # cover_rej is tiny, so it is not treated as a re-detection. The old
            # center-in-region test had no such size guard.
            return iou > 0.6 or (cover_new > 0.7 and cover_rej > 0.7)
        kept_b, kept_p = [], []
        dropped = 0
        for i, b in enumerate(boxes):
            if any(_redetects(b, rb) for rb in rej_px):
                dropped += 1
                continue
            kept_b.append(b)
            if polys is not None:
                kept_p.append(polys[i] if i < len(polys) else None)
        if dropped:
            print(f"[REJECT] dropped {dropped} re-detection(s) of previously-deleted object(s)")
        return kept_b, (kept_p if polys is not None else None)

    @staticmethod
    def _combine_with_dedup(det_boxes, manual_boxes, iou_thresh=0.5):
        """Return (boxes, sources): detector boxes FIRST, then manual boxes.
        A manual box is appended only if it doesn't duplicate a kept box
        (guards SAM echoing its input boxes back as outputs). Detector
        duplicates of a user-drawn box are removed UPSTREAM by
        _drop_detector_dups_of_manual (manual wins / the drawn box persists),
        so by the time we get here no detector box overlaps a manual one and
        every manual box survives. Detector-first order is required by the
        one-shot seg path, which aligns det polys to boxes[:len(det_polys)]."""
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = ua + ub - inter
            return inter / union if union > 0 else 0.0
        out = list(det_boxes)
        sources = ['detector'] * len(out)
        for mb in manual_boxes:
            if not any(_iou(mb, ob) > iou_thresh for ob in out):
                out.append(list(mb))
                sources.append('manual')
        return out, sources

    @staticmethod
    def _drop_detector_dups_of_manual(det_boxes, manual_boxes, polys=None, iou_thresh=0.5):
        """Drop detector boxes (and their index-aligned polys) that duplicate a
        user-drawn manual box, so the drawn prompt box persists across a
        Regenerate with no overlapping detector box on the same object. Only
        detector entries are removed, so detector-first ordering and the
        det_boxes<->polys alignment are preserved; manual boxes are appended
        afterwards by _combine_with_dedup. Mirrors _drop_rejected."""
        if not manual_boxes:
            return list(det_boxes), (list(polys) if polys is not None else None)
        def _iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            u = ua + ub - inter
            return inter / u if u > 0 else 0.0
        kept_b, kept_p = [], []
        for i, b in enumerate(det_boxes):
            if any(_iou(b, mb) > iou_thresh for mb in manual_boxes):
                continue
            kept_b.append(b)
            if polys is not None:
                kept_p.append(polys[i] if i < len(polys) else None)
        return kept_b, (kept_p if polys is not None else None)

    def _boxes_from_seg_polys(self, polys_norm, image_path, fallback=None):
        """Create a box AROUND each segmentation: the tight pixel-xyxy bbox of
        each normalized mask polygon, so the saved YOLO box hugs the SAM3
        segmentation. Falls back to the matching `fallback` entry for any
        empty/degenerate polygon."""
        fb = list(fallback or [])
        img = self._imread_cached(image_path)
        if img is None:
            return fb
        ih, iw = img.shape[:2]
        out = []
        for i, poly in enumerate(polys_norm or []):
            if poly and len(poly) >= 3:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                out.append([min(xs) * iw, min(ys) * ih, max(xs) * iw, max(ys) * ih])
            elif i < len(fb):
                out.append(fb[i])
        return out

    def _active_interactive_sam_key(self):
        """Model key for interactive point-prompted SAM, or None.

        Available when SAM2/SAM3 is the segmenter, or when SAM3 is the one-shot
        detector (its interactive SAM3 weights). Returns None for YOLOE-seg /
        YOLOE-vis standalone and DINO-bbox-only, so the Draw Mask tool stays
        disabled there."""
        det_key, seg_key, _ = self._detector_keys_for_pipeline()
        if seg_key in ("sam2_t", "sam3"):
            return seg_key
        if det_key == "sam3_det":
            return "sam3"
        return None

    # Robust model / state switching for the semi-auto feature
    @staticmethod
    def _is_one_shot_detector(text):
        t = text or ""
        return ("YOLOE-seg" in t or "one-shot" in t
                or t.startswith("SAM3 (") or "YOLOE-vis" in t)

    def _sam_available_for(self, det, seg):
        """Predict whether interactive SAM would be available for a given
        (detector, segmenter) pair, without committing the choice."""
        save_d, save_s = self.detector_choice, self.segmenter_choice
        self.detector_choice, self.segmenter_choice = det, seg
        try:
            return self._active_interactive_sam_key() is not None
        finally:
            self.detector_choice, self.segmenter_choice = save_d, save_s

    def _semiauto_in_use(self):
        """True when a SAM draw tool is armed/active or its edit mode is on."""
        if getattr(self, "_draw_tool", "box") in ("semiauto", "autodraw"):
            return True
        return bool(getattr(self, "image_label", None) is not None
                    and self.image_label.semiauto_edit_mode)

    def _deactivate_semiauto(self):
        """Cleanly tear down every semi-auto mode + in-progress session. Safe to
        call repeatedly; used before a model switch that removes SAM. The discard
        guard is suppressed throughout: callers reach here only after the loss was
        already authorised (or no SAM tool is in use), so it must not re-prompt."""
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            if getattr(self, "_draw_tool", "box") in ("semiauto", "autodraw"):
                if hasattr(self, "draw_btn") and self.draw_btn.isChecked():
                    self.draw_btn.setChecked(False)
                self._select_draw_tool("box")
            # If drawn-mask editing is the live Edit mode, fall back to Edit Boxes.
            if getattr(self, "_edit_tool", "boxes") == "masks":
                self._select_edit_tool("boxes")
            if hasattr(self, "image_label"):
                self.image_label.set_mask_draw_mode(False)
                self.image_label.set_semiauto_edit_mode(False)
                self.image_label.clear_mask_session()
                self.image_label.clear_semiauto_selection()
        finally:
            self._in_mode_switch = was

    def _styled_message(self, text, title=""):
        """A QMessageBox in the app's consistent look (white background, black
        24px text), the same style as the 'Please draw a prompt box' alerts.
        Caller adds whatever buttons it needs."""
        box = QtWidgets.QMessageBox(self)
        box.setStyleSheet("QLabel { color: black; font-size: 24px; } "
                          "QMessageBox { background-color: white; }")
        if title:
            box.setWindowTitle(title)
        box.setText(text)
        return box

    def _ask_sam_loss(self, has_work):
        """Modal alert for a model change that would drop interactive SAM while
        the semi-auto tool is in use. Semi-Automatic Point Segmentation needs a
        SAM2/SAM3 model, so the default is to keep it (not switch). Returns
        'switch' or 'revert'. Factored out so tests can stub it."""
        msg = ("A SAM2 or SAM3 model must be selected to use Semi-Automatic "
               "Point Segmentation. This model does not support it, so switching "
               "to it turns the tool off.")
        if has_work:
            msg += "\n\nYour in-progress points will be discarded."
        msg += "\n\nKeep the SAM model, or switch anyway?"
        box = self._styled_message(msg, "SAM Model Required")
        revert_btn = box.addButton("Keep SAM model", QtWidgets.QMessageBox.RejectRole)
        switch_btn = box.addButton("Switch anyway", QtWidgets.QMessageBox.AcceptRole)
        box.setDefaultButton(revert_btn)
        box.exec_()
        return "revert" if box.clickedButton() is revert_btn else "switch"

    def _guard_model_change(self, which, new_text):
        """Gate a detector/segmenter change that would disable interactive SAM
        while the feature is in use. Returns True to let the change proceed
        (cleaning up semi-auto first), False if the user reverted (the combo is
        restored to its prior value)."""
        was = self._active_interactive_sam_key() is not None
        if which == "segmenter":
            now = self._sam_available_for(self.detector_choice, new_text)
            prev_text, combo = self.segmenter_choice, getattr(self, "segmenter_combo", None)
        else:
            new_seg = "(none)" if self._is_one_shot_detector(new_text) else self.segmenter_choice
            now = self._sam_available_for(new_text, new_seg)
            prev_text, combo = self.detector_choice, getattr(self, "detector_combo", None)
        il = getattr(self, "image_label", None)
        unfinished = bool(il is not None and il.has_unfinished_semiauto())
        editing = bool(il is not None and il.semiauto_edit_mode
                       and il.get_semiauto_selected_index() is not None)
        has_work = unfinished or editing
        # Switching BETWEEN SAM models keeps the feature available (was and now
        # both True), so it is always allowed. Only a switch to a non-SAM
        # segmenter, or a non-SAM one-shot detector, while the semi-auto tool is
        # in use needs the user to authorise dropping the feature.
        if not (was and not now and (self._semiauto_in_use() or has_work)):
            return True
        if self._ask_sam_loss(has_work) == "revert":
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentText(prev_text)
                combo.blockSignals(False)
            return False
        self._deactivate_semiauto()
        return True

    def _ask_discard_unfinished(self, context):
        """Modal alert for uncommitted points (a draw draft or a mask edit) when
        switching away. Returns True to discard and continue, False to keep them
        and cancel the switch. Factored out for tests."""
        box = self._styled_message(
            "You have uncommitted points that have not been saved.\n\n"
            f"Discard them and continue {context}? Choose Cancel to keep them "
            "and finish first (press Enter to save).",
            "Uncommitted Points")
        discard_btn = box.addButton("Discard", QtWidgets.QMessageBox.AcceptRole)
        cancel_btn  = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec_()
        return box.clickedButton() is discard_btn

    def _force_btn(self, btn, checked):
        """Set a mode button from code WITHOUT tripping the discard guard. The
        guard is only meant for genuine user clicks; every programmatic toggle
        (teardown, save, model switch) must route through here."""
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            btn.setChecked(checked)
        finally:
            self._in_mode_switch = was

    def _guard_tool_switch(self):
        """Before a tool/mode switch throws away in-progress work, make the user
        authorize it. Returns True to proceed (work discarded on confirm), False
        to abort the switch (work kept). Suppressed during the programmatic
        button juggling of a switch that already cleared the guard."""
        if self._in_mode_switch:
            return True
        il = self.image_label
        draft   = il.has_unfinished_semiauto()
        editing = (il.semiauto_edit_mode
                   and il.get_semiauto_selected_index() is not None)
        if not (draft or editing):
            return True
        if not self._ask_discard_unfinished("switching tools"):
            return False
        il.cancel_inprogress()
        return True

    def _confirm_leave_unfinished(self, context):
        """If a semi-auto mask is mid-draw, confirm discarding it. Returns True
        to proceed (clearing the points), False to stay."""
        if not (hasattr(self, "image_label")
                and self.image_label.has_unfinished_semiauto()):
            return True
        if not self._ask_discard_unfinished(context):
            return False
        self.image_label.clear_mask_session()
        return True

    def _run_segmenter(self, image_path, boxes):
        """Run the segmentation portion of the current pipeline. Returns SAM-style results."""
        _, seg_key, is_standalone = self._detector_keys_for_pipeline()
        if is_standalone or seg_key is None:
            return None
        if not boxes:
            return None
        sam = self._get_model(seg_key)
        # segment_with_boxes (not a raw sam() call): it guarantees one mask per
        # prompt box, index-aligned, with each mask clipped near its box. The
        # raw call let ultralytics conf-drop weak masks (shifting every later
        # mask onto the wrong box) and let SAM3 concept-spill giant masks far
        # beyond the boxed object.
        try:
            return segment_with_boxes(sam, image_path, boxes)
        finally:
            # Release after EVERY segmenter run (SAM2/SAM3), regardless of which
            # operation invoked it -- per-image predict, batch, mode switch, or the
            # manual-boxes one-shot -- so no detector/segmenter combination leaks
            # memory across a long folder.
            self._release_inference_memory()

    # Additive regenerate (preserve earlier iterations)
    @staticmethod
    def _copy_ann(a):
        """Deep-ish copy of an annotation dict (data list copied, not shared)."""
        if a['type'] == 'poly':
            data = [list(p) for p in a['data']]
        else:
            data = list(a['data'])
        out = {'type': a['type'], 'data': data,
               'deleted': a.get('deleted', False),
               'source': a.get('source', 'detector')}
        # Carry per-mask metadata (semi-auto flag, SAM prompt points, class id)
        # so additive regenerate doesn't strip a committed semi-auto mask.
        for k in ('semiauto', 'sam_points', 'cls'):
            if k in a:
                out[k] = ([list(p) for p in a[k]] if k == 'sam_points' else a[k])
        return out

    @staticmethod
    def _ann_bbox_norm(ann):
        """Normalized xyxy bounding box of a rect or poly annotation."""
        if ann['type'] == 'rect':
            cx, cy, w, h = ann['data']
            return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        xs = [p[0] for p in ann['data']]
        ys = [p[1] for p in ann['data']]
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _merge_additive_anns(prior_anns, new_anns, iou_thresh=0.5):
        """prior_anns + (new_anns minus any that duplicate a prior ann by
        bbox IoU). Lets Regenerate ADD fresh detections for newly drawn
        boxes while keeping every box/segment from earlier iterations."""
        def _iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            u = ua + ub - inter
            return inter / u if u > 0 else 0.0
        bbs = [ManualWindow._ann_bbox_norm(a) for a in prior_anns]
        merged = list(prior_anns)
        for na in new_anns:
            nb = ManualWindow._ann_bbox_norm(na)
            # Always keep a user-drawn box; only dedup detector output against
            # the prior set. A redundant prior detector box, if any, is removed
            # later by the source-aware _dedup_anns (manual wins).
            if na.get('source') == 'manual' or not any(_iou(nb, pb) > iou_thresh for pb in bbs):
                merged.append(na)
                bbs.append(nb)
        return merged

    def _write_label_files(self, anns, image_path):
        """Overwrite the YOLO box (and, in seg mode, segment) label files for
        `image_path` from the given annotation list."""
        if not self.output_folder:
            return
        img = self._imread_cached(image_path)
        if img is None:
            return
        stem = os.path.splitext(os.path.basename(image_path))[0]
        box_dir = os.path.join(self.output_folder, 'boxes')
        seg_dir = os.path.join(self.output_folder, 'segments')
        os.makedirs(box_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        live = [a for a in anns if not a.get('deleted', False) and a.get('source') != 'prompt']
        polys = [a for a in live if a['type'] == 'poly']
        rects = [a for a in live if a['type'] == 'rect']
        with open(f'{box_dir}/{stem}.txt', 'w') as bf:
            for a in polys:
                xs = [p[0] for p in a['data']]; ys = [p[1] for p in a['data']]
                cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2
                bw = max(xs) - min(xs); bh = max(ys) - min(ys)
                bf.write(f'{int(a.get("cls", 0))} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            for a in rects:
                cx, cy, bw, bh = a['data']
                bf.write(f'{int(a.get("cls", 0))} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        # Only (re)write the segment file when there are polygons to write,
        # so a bbox-mode regenerate doesn't blank an earlier mask file.
        if polys:
            with open(f'{seg_dir}/{stem}.txt', 'w') as sf:
                for a in polys:
                    coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in a['data'])
                    sf.write(f'{int(a.get("cls", 0))} {coords}\n')

    def _finalize_additive(self, prior_anns, image_path):
        """Tail step for an additive Regenerate: prepend the preserved prior
        annotations to the freshly generated set, rebuild live state, re-save
        the label files, and re-bake the overlay so the screen matches."""
        new_anns = [self._copy_ann(a) for a in self.image_label.annotations
                    if not a.get('deleted', False) and a.get('source') != 'prompt']
        merged = self._merge_additive_anns(
            [self._copy_ann(a) for a in prior_anns], new_anns)
        # Final overlap cleanup so accumulated regenerates don't pile up
        # near-duplicate boxes/masks on the same object.
        merged = self._dedup_anns(merged)
        self.image_label.load_annotation_state(merged)
        merged = self.image_label.annotations
        ow = self.image_label._orig_w or 1
        oh = self.image_label._orig_h or 1
        self.live_boxes = []
        self.live_box_sources = []
        for ann in merged:
            if ann.get('source') == 'prompt':
                continue
            b = self._ann_bbox_norm(ann)
            self.live_boxes.append([b[0] * ow, b[1] * oh, b[2] * ow, b[3] * oh])
            self.live_box_sources.append(ann.get('source', 'detector'))
        # Force re-segmentation on the next mode switch rather than trusting a
        # cache that no longer aligns with the merged annotation list.
        self.live_polys_cache = None
        self._write_label_files(merged, image_path)
        self._rebake_overlay()
        if self.current_mode == "bbox":
            self.bbox_anns = list(merged)
        else:
            self.seg_anns = list(merged)

    # Live-boxes sync (state carry-over)
    def _on_canvas_changed(self):
        """Canvas edits happened (deletion, manual draw, manual box removed).
        Rebuild live_boxes from the canvas's current truth and invalidate caches."""
        if self.image_label._orig_w is None:
            return
        # Active rects already reflect deletions in bbox mode.
        # In seg mode (active anns are polygons), fall back to the previous live_boxes
        # minus any deletions and plus manual draws.
        manual = self.image_label.get_boxes_in_image_coords()
        if self.current_mode == "bbox":
            # active_rects already includes manual rect anns under the
            # unified model; `manual` would re-add the same entries.
            rect_pairs = [(b, s) for b, s in self.image_label.get_active_rects_with_sources()
                          if s != 'prompt']
            self.live_boxes        = [b for b, _ in rect_pairs]
            self.live_box_sources  = [s for _, s in rect_pairs]
        elif self.current_mode == "seg":
            # Seg-mode deletions remove from live_boxes index-aligned with seg anns.
            kept = []
            kept_sources = []
            orphaned = 0
            for i, ann in enumerate(self.image_label.annotations):
                if not ann['deleted'] and i < len(self.live_boxes):
                    kept.append(self.live_boxes[i])
                    kept_sources.append(ann.get('source',
                        self.live_box_sources[i] if i < len(self.live_box_sources) else 'detector'))
                elif not ann['deleted'] and ann.get('source') != 'manual':
                    # Index-alignment hazard: a live, non-manual annotation sits
                    # beyond live_boxes, so it has NO index-aligned box to carry.
                    # This is the suspected cause of purple (detector) masks
                    # dropping on a later Regenerate. Count it; warning printed
                    # below (always visible) so a recurrence leaves a trail.
                    orphaned += 1
            if orphaned:
                print(f"[SEG-SYNC] WARNING: {orphaned} non-manual annotation(s) have "
                      f"no index-aligned live_box ({len(self.image_label.annotations)} "
                      f"anns vs {len(self.live_boxes)} boxes); their box is dropped "
                      f"from live_boxes and they may vanish on the next Regenerate. "
                      f"See task #10 / project_regenerate_purple_mask_loss.")
            self.live_boxes       = kept + manual
            self.live_box_sources = kept_sources + (['manual'] * len(manual))
        # Any baked snapshot is now stale.
        self.baked_bbox_cv2 = None
        self.baked_seg_cv2  = None
        self.live_polys_cache = None
        # Re-bake the overlay from the CURRENT annotation set so the non-edit
        # display matches the canvas after a delete/draw. Without this the
        # canvas keeps showing the pre-delete baked pixmap (deleted box/segment
        # reappears, overlapping the live state) once Edit Boxes is toggled off.
        self._rebake_overlay()
        # When Carry Prompts Forward is on, freeze the drawn box prompts into the
        # persistent anchor so they survive a regenerate (which consumes the
        # manual boxes) and keep Auto Annotate Remaining enabled / carried.
        # Delete-means-gone: if the user removed the last manual prompt box,
        # forget the frozen carry anchor so Auto Annotate greys out and the
        # next image doesn't inherit a box that was just deleted. This handler
        # only fires on user canvas edits (draw/delete), never on Regenerate,
        # so a regenerate that consumes the boxes keeps the anchor intact.
        if not self.image_label.get_prompt_boxes_in_image_coords():
            self._carry_anchor = []
        elif self._carry_active():
            self._refresh_and_get_carry_anchor()
        # Drawing/removing a box can change whether Auto Annotate can run.
        self._refresh_auto_annotate_enabled()

    def _render_annotations_overlay(self):
        """Bake every non-deleted annotation onto a fresh copy of the base
        image. Single source of truth for the non-edit-mode display, so what
        the user sees always matches the annotation state."""
        if self.base_cv2_image is None:
            return None
        overlay = self.base_cv2_image.copy()
        h, w = overlay.shape[:2]
        for ann in self.image_label.get_active_annotations():
            if ann.get('source') == 'prompt':
                continue  # input-only prompt boxes are drawn live, never baked/saved
            color = (0, 200, 100) if ann.get('source') == 'manual' else (255, 0, 255)
            if ann['type'] == 'poly':
                pts = np.array([[int(x * w), int(y * h)] for x, y in ann['data']],
                               dtype=np.int32)
                if len(pts) >= 3:
                    # isClosed=True draws one closed ring per polygon, no
                    # stray line linking separate annotations.
                    cv2.polylines(overlay, [pts], True, color, 2)
            elif ann['type'] == 'rect':
                cx, cy, bw, bh = ann['data']
                x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        return overlay

    def _rebake_overlay(self):
        """Refresh the canvas's baked pixmap + the cached baked image for the
        current mode from the live annotation set."""
        overlay = self._render_annotations_overlay()
        if overlay is None:
            return
        self.image_label.set_baked_image(overlay)
        if self.current_mode == "bbox":
            self.baked_bbox_cv2 = overlay.copy()
        elif self.current_mode == "seg":
            self.baked_seg_cv2 = overlay.copy()

    # Annotated-image export
    def _render_overlay_image(self, image_path, boxes=None, polys=None):
        """Return a copy of the image with boxes (absolute xyxy) and/or polys
        (normalized point lists) drawn on it, for saving as a reference image."""
        img = self._imread_cached(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        for p in (polys or []):
            if not p or len(p) < 3:
                continue
            pts = np.array([[int(x * w), int(y * h)] for x, y in p], dtype=np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 255), 2)
        for box in (boxes or []):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
        return img

    def _save_annotated_image(self, image_path, overlay_cv2, kind):
        """Write an annotated overlay image to annotated/<kind>/<stem>.jpg.

        `kind` is 'boxes' or 'masks'. The two reference views used to
        land on the same JPEG, which got crowded once both overlays
        were present; splitting them mirrors the box/segment split
        already in the label folders."""
        if not self.output_folder or overlay_cv2 is None:
            return
        if kind not in ('boxes', 'masks'):
            raise ValueError(f"_save_annotated_image: kind must be 'boxes' or 'masks', got {kind!r}")
        ann_dir = os.path.join(self.output_folder, f'annotated_{self._model_tag()}', kind)
        os.makedirs(ann_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        # Review overlays only (boxes/masks painted on top for visual
        # inspection); no model trains on these, so q85 is a free space win
        # vs OpenCV's default q95. Synthetic TRAINING images keep q95 in
        # _save_variation.
        cv2.imwrite(os.path.join(ann_dir, f'{stem}.jpg'), overlay_cv2,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    def _save_split_overlays(self, image_path, boxes, polys):
        """Helper: render boxes-only and masks-only reference images and
        save each to its subfolder. The masks variant is skipped when
        there are no polygons (bbox-only runs don't need an empty
        polygon overlay). Boxes can be derived from polygon bboxes by
        the caller; if the caller passes an empty box list we skip
        the boxes variant too."""
        if boxes:
            box_only = self._render_overlay_image(image_path, boxes=boxes, polys=None)
            self._save_annotated_image(image_path, box_only, 'boxes')
        if polys:
            mask_only = self._render_overlay_image(image_path, boxes=None, polys=polys)
            self._save_annotated_image(image_path, mask_only, 'masks')

    # Predictions
    def keyPressEvent(self, event):
        # Detect Enter key
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            # In boxes mode the prompt comes from drawn boxes, so don't gate on text.
            if self.prompt_mode == "boxes" or self.prompt_entry.text().strip():
                self.display_predictions()
            else:
                message_box = QtWidgets.QMessageBox()
                message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
                message_box.setText("Please enter a prompt before running the model.")
                message_box.exec_()

    def display_predictions(self):
        if getattr(self, "_busy", False):
            return
        if not self.images:
            message_box = QtWidgets.QMessageBox()
            message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
            message_box.setText("Please select an image folder first.")
            message_box.exec_()
            return
        # Validate inputs based on prompt mode.
        if (self.prompt_mode == "text" and not self.prompt_entry.text().strip()
                and not self.image_label.get_prompt_boxes_in_image_coords()):
            message_box = QtWidgets.QMessageBox()
            message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
            message_box.setText("Please enter a prompt to run the model.")
            message_box.exec_()
            return
        if (self.prompt_mode == "boxes"
                and not self.image_label.get_prompt_boxes_in_image_coords()
                and not getattr(self, "_carry_prompt_img", None)
                and not (self._carry_active() and getattr(self, "_carry_anchor", None))):
            # The detector consumes the yellow prompt-bucket (prompt_boxes), not
            # the green annotation bucket. Drawing-subject auto-snaps to "Prompt"
            # for YOLOE-vis but the user can pick "Annotation" for YOLOE-seg+boxes;
            # validate against what the detector will actually read.
            message_box = QtWidgets.QMessageBox()
            message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
            message_box.setText("Please draw at least one prompt box (yellow) before running the model.")
            message_box.exec_()
            return

        prompt = self.prompt_entry.text()
        confidence  = self.detection_threshold_slider.value() / 100
        mask_thresh = self.mask_threshold_slider.value() / 100
        # `max_area` kept for back-compat with display_* signatures; it's
        # reinterpreted as mask_threshold downstream.
        max_area    = mask_thresh
        if not self.output_folder:
            message_box = QtWidgets.QMessageBox()
            message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
            message_box.setText("Please select an output folder before running predictions.")
            message_box.exec_()
            return
        # Snapshot any boxes/segments the user has deleted into the reject
        # list BEFORE the regenerate replaces the annotation set, so the
        # detector's re-detections of those same objects can be suppressed.
        self._collect_rejected()
        # Busy state for slow paths (especially first SAM3 text load, several
        # hundred MB). processEvents() forces Qt to paint the :disabled style
        # before we block on the model call. Re-enable in finally so a crash
        # never leaves the UI dead.
        busy_btns = [b for b in (getattr(self, "auto_annotate_btn", None),
                                 getattr(self, "regen_btn", None),
                                 getattr(self, "next_btn", None)) if b is not None]
        self._busy = True
        for b in busy_btns:
            b.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        import time
        _t0 = time.perf_counter()
        try:
            _ran = False
            if self.box_checkbox.isChecked():
                self.display_boxes_with_borders(image_path=self.images[self.current_image_index],
                                                prompt=prompt, confidence=confidence,
                                                max_area=max_area, output_path=self.output_folder)
                _ran = True
            elif self.mask_checkbox.isChecked():
                self.display_masks_with_borders(image_path=self.images[self.current_image_index],
                                                prompt=prompt, confidence=confidence,
                                                max_area=max_area, output_path=self.output_folder)
                _ran = True
            else:
                message_box = QtWidgets.QMessageBox()
                message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
                message_box.setText("Please select a display mode (Bounding Box or Segmentation Mask).")
                message_box.exec_()
            if _ran:
                _i = self.current_image_index + 1
                _name = os.path.basename(self.images[self.current_image_index])
                print(f"[auto-annotate] {_name} (image {_i}/{len(self.images)} in "
                      f"folder): {time.perf_counter() - _t0:.1f}s -> "
                      f"{len(self.live_boxes)} boxes")
        except Exception as e:
            import traceback
            error_text = traceback.format_exc()
            print(error_text)
            error_img = QtGui.QImage(self.image_label.width(), self.image_label.height(), QtGui.QImage.Format_RGB888)
            error_img.fill(QtGui.QColor(40, 40, 40))
            painter = QtGui.QPainter(error_img)
            painter.setPen(QtGui.QColor(255, 80, 80))
            painter.setFont(QtGui.QFont("Monospace", 12))
            painter.drawText(error_img.rect(), QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
                             f"PREDICTION ERROR:\n{error_text}")
            painter.end()
            self.image_label._baked_pixmap = QtGui.QPixmap.fromImage(error_img)
            self.image_label.update()
            err_box = QtWidgets.QMessageBox()
            err_box.setStyleSheet("QLabel { color: black; font-size: 18px; } QMessageBox { background-color: white; }")
            err_box.setWindowTitle("Prediction Error")
            err_box.setText(f"Error running prediction:\n{str(e)}")
            err_box.exec_()
        finally:
            self._busy = False
            for b in busy_btns:
                b.setEnabled(True)
            # Re-apply the availability rule (a regenerate may have changed
            # the box/prompt state) instead of leaving it unconditionally on.
            self._refresh_auto_annotate_enabled()
            # Release per-image inference memory so the MPS/CUDA allocator pool
            # and Python temporaries don't creep across a long folder and push
            # an 8GB machine into swap (the cause of the per-image slowdown).
            self._release_inference_memory()

    def _imread_cached(self, path):
        """Decode an image once per prediction and hand out cheap COPIES. A
        single prediction reads the same file 3-5x (detector, segmenter, overlay,
        box-from-poly); the decode dominates, so we cache the decoded array
        (single most-recent path -> bounded memory) and return a copy each call
        so callers can draw on it without corrupting the cache. Path-keyed, so a
        different image just re-decodes (never stale data). Returns None for an
        unreadable file and caches that too, so a bad decode isn't retried every
        call. Falls back to a plain decode if anything unexpected happens."""
        try:
            cached = getattr(self, "_imread_cache", None)
            if cached is not None and cached[0] == path:
                arr = cached[1]
            else:
                arr = cv2.imread(path)
                self._imread_cache = (path, arr)
            return arr.copy() if arr is not None else None
        except Exception:
            return cv2.imread(path)

    def _release_every(self):
        """Inference calls between full memory releases. From
        AUTOANNOTATE_RELEASE_EVERY (default 1 = release every time = the original
        behavior). Raise it (e.g. 5) on a roomy CUDA / large-RAM machine to cut
        the per-image gc + empty_cache overhead; keep it at 1 on an 8GB box."""
        try:
            n = int(os.environ.get("AUTOANNOTATE_RELEASE_EVERY", "1"))
            return n if n > 0 else 1
        except (TypeError, ValueError):
            return 1

    def _release_inference_memory(self, force=False):
        """Free per-image inference memory (gc + torch allocator empty_cache) so
        the pool + Python temporaries don't accumulate over a long folder and
        thrash a low-RAM (8GB MPS) machine into swap. Honors
        AUTOANNOTATE_RELEASE_EVERY (skip all but every Nth call) UNLESS
        force=True (used right after a model eviction, where freeing immediately
        is the whole point)."""
        self._release_tick = getattr(self, "_release_tick", 0) + 1
        every = self._release_every()
        if not force and every > 1 and (self._release_tick % every) != 0:
            return
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

    def display_boxes_with_borders(self, image_path, prompt, confidence, max_area, output_path, DINO=None):
        """Run detection and display boxes. Updates `self.live_boxes` as truth source.

        Manual additions (from the green annotation_boxes bucket and from any
        prior-iteration manual rects already in `self.image_label.annotations`)
        are promoted to durable manual rect annotations and the bucket is
        cleared. This means re-running the detector does NOT lose previous
        manual draws and does NOT duplicate them in the saved YOLO file."""
        img = self._imread_cached(image_path)
        boxes_dir = os.path.join(output_path, 'boxes')
        os.makedirs(boxes_dir, exist_ok=True)

        # Additive regenerate: for YOLOE box detectors, snapshot the prior
        # iteration's non-deleted annotations so this run ADDS to them rather
        # than replacing them (see _finalize_additive).
        det_key0, _, _ = self._detector_keys_for_pipeline()
        # Both YOLOE one-shot detectors get additive regenerate. (No
        # prompt_mode gate: YOLOE-seg visual-prompts from drawn boxes
        # regardless of the Text/Boxes radio, so it must preserve too.)
        # SAM3 (one-shot) box-exemplar mode also joins: drawing a new
        # exemplar should ADD to prior detections, not replace them, so
        # the ROI-partition fix doesn't lose the good results either.
        additive = ((det_key0 in ("yoloe_vis", "yoloe_seg")
                     or (det_key0 == "sam3_det" and self.prompt_mode == "boxes"))
                    and bool(self.image_label.get_active_annotations()))
        # Preserve prior DETECTOR/segment output across the regenerate, but
        # NOT the manually drawn rect prompt boxes (yellow). Those are this
        # run's input: the detector consumes them and turns them into real
        # detections, which arrive via the normal run's output. Keeping them
        # in `prior_anns` too would (a) leave the yellow prompt box on screen
        # after regenerate and (b) double-count it against its own detection.
        prior_anns = ([self._copy_ann(a) for a in self.image_label.get_active_annotations()
                       if a.get('source') != 'prompt'
                       and not (a['type'] == 'rect' and a.get('source') == 'manual')]
                      if additive else [])

        prompt_boxes_img     = self.image_label.get_prompt_boxes_in_image_coords()
        # Carry Prompts Forward: reuse the frozen anchor when the manual boxes
        # were consumed by a prior regenerate, so they need not be redrawn.
        if not prompt_boxes_img and self._carry_active():
            prompt_boxes_img = self._carry_anchor_boxes_img()
        # Carried exemplar (from Next Image) primes the DETECTOR only; it is
        # consumed once and never added to the saved/segmented manual set.
        carry_prompt         = list(getattr(self, "_carry_prompt_img", None) or [])
        self._carry_prompt_img = []
        # Carry-Prompts-Forward appearance bundle (crops of the previous image's
        # boxes), set by next_image. Consumed once, like carry_prompt. Lets
        # YOLOE use a TRUE refer_image one-shot and SAM3 use crop-composite for
        # single-step Next Image carry instead of raw coordinates that would
        # land on background in the new image.
        carry_ref            = getattr(self, "_carry_ref_bundle", None)
        self._carry_ref_bundle = None
        # Drawn boxes serve both as detector prompts and as saved annotations
        # under the unified model. `manual_boxes_img` covers rect-form draws
        # (current state) and poly-form manuals (SAM masks of earlier draws
        # carried forward through seg mode) via the type-agnostic helper.
        manual_boxes_img     = list(self.image_label.get_manual_anns_as_boxes_in_image_coords())

        det_boxes, yoloe_seg_results = self._run_detector(image_path, prompt, confidence, max_area, prompt_boxes_img + carry_prompt, ref=carry_ref)
        # Suppress detections on regions the user deleted earlier on this
        # image so a regenerate doesn't bring the same objects straight back.
        det_boxes, self._oneshot_polys_aligned = self._drop_rejected(
            det_boxes, self._oneshot_polys_aligned)
        det_key, _, is_standalone = self._detector_keys_for_pipeline()
        # Manual wins: drop detector boxes (and their aligned one-shot polys)
        # that duplicate a user-drawn box, so the drawn prompt box persists
        # across Regenerate with no overlapping detector box on the same object.
        det_boxes, self._oneshot_polys_aligned = self._drop_detector_dups_of_manual(
            det_boxes, manual_boxes_img, self._oneshot_polys_aligned)
        # Final on-screen + on-disk box set = detector output (first) + manual,
        # IoU-deduped so SAM's pass-through (it echoes prompts as outputs)
        # doesn't double-count drawn boxes.
        absolute_boxes, sources = self._combine_with_dedup(list(det_boxes), manual_boxes_img, 0.5)

        # Persist post-edit truth: overwrite the YOLO file with the actual displayed set.
        save_boxes_yolo(absolute_boxes, image_path, boxes_dir)

        # Color the baked overlay so manual boxes are green even outside edit mode.
        colors = [(0, 200, 100) if s == 'manual' else (255, 0, 255) for s in sources]
        img_with_boxes = draw_boxes_on_image(img, absolute_boxes, colors=colors)

        # Update live truth.
        self.live_boxes = list(absolute_boxes)
        self.live_box_sources = list(sources)
        self.live_polys_cache = None
        if is_standalone and self._oneshot_polys_aligned is not None:
            # Index-aligned with det_boxes (filtered by max_area + >=3 pts in
            # _run_detector). live_boxes = det_boxes + manual; cache only
            # covers the det portion. _switch_to_mode handles the rest.
            self.live_polys_cache = list(self._oneshot_polys_aligned)

        # Cache for mode switching
        self.base_cv2_image = img.copy()
        self.baked_bbox_cv2 = img_with_boxes.copy()

        # Build normalised rect annotations so Edit Boxes works in bbox mode
        h, w = img.shape[:2]
        rects_norm = []
        for x1, y1, x2, y2 in absolute_boxes:
            rects_norm.append([
                (x1 + x2) / 2 / w,
                (y1 + y2) / 2 / h,
                (x2 - x1) / w,
                (y2 - y1) / h,
            ])

        self.image_label.set_clean_image(img)
        self.image_label.set_baked_image(img_with_boxes)
        self.image_label.set_annotations(rects=rects_norm, rect_sources=sources)
        # Bucket boxes are now durable rect annotations; clear the bucket so
        # they aren't double-counted on the next regenerate or save.
        self.image_label.annotation_boxes = []
        self.image_label.update()
        self.bbox_anns    = list(self.image_label.annotations)
        self.current_mode = "bbox"

        # Additive regenerate: fold the prior iteration's boxes back in so
        # earlier detections survive a regenerate after new boxes are drawn.
        if additive:
            self._finalize_additive(prior_anns, image_path)
            self.image_label.update()

    def display_masks_with_borders(self, image_path, prompt, confidence, max_area, output_path, DINO=None):
        """Run detection + segmentation. Saves masks for the final live box set.

        Same promotion logic as display_boxes_with_borders: existing manual
        rects + bucket draws are carried forward, tagged 'manual', and the
        bucket is cleared. SAM result polys inherit the source of the box
        that produced them (first len(det_boxes) = detector, rest = manual)."""
        img = self._imread_cached(image_path)
        boxes_dir = os.path.join(output_path, 'boxes')
        seg_dir = os.path.join(output_path, 'segments')
        os.makedirs(boxes_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        # Additive regenerate: snapshot the prior iteration's annotations for
        # YOLOE box detectors so this run adds to them (see _finalize_additive).
        det_key0, _, _ = self._detector_keys_for_pipeline()
        # Both YOLOE one-shot detectors get additive regenerate. (No
        # prompt_mode gate: YOLOE-seg visual-prompts from drawn boxes
        # regardless of the Text/Boxes radio, so it must preserve too.)
        # SAM3 (one-shot) box-exemplar mode also joins: drawing a new
        # exemplar should ADD to prior detections, not replace them, so
        # the ROI-partition fix doesn't lose the good results either.
        additive = ((det_key0 in ("yoloe_vis", "yoloe_seg")
                     or (det_key0 == "sam3_det" and self.prompt_mode == "boxes"))
                    and bool(self.image_label.get_active_annotations()))
        # Preserve prior DETECTOR/segment output across the regenerate, but
        # NOT the manually drawn rect prompt boxes (yellow). Those are this
        # run's input: the detector consumes them and turns them into real
        # detections, which arrive via the normal run's output. Keeping them
        # in `prior_anns` too would (a) leave the yellow prompt box on screen
        # after regenerate and (b) double-count it against its own detection.
        prior_anns = ([self._copy_ann(a) for a in self.image_label.get_active_annotations()
                       if a.get('source') != 'prompt'
                       and not (a['type'] == 'rect' and a.get('source') == 'manual')]
                      if additive else [])

        prompt_boxes_img     = self.image_label.get_prompt_boxes_in_image_coords()
        # Carry Prompts Forward: reuse the frozen anchor when the manual boxes
        # were consumed by a prior regenerate, so they need not be redrawn.
        if not prompt_boxes_img and self._carry_active():
            prompt_boxes_img = self._carry_anchor_boxes_img()
        # Carried exemplar (from Next Image) primes the DETECTOR only; it is
        # consumed once and never added to the saved/segmented manual set.
        carry_prompt         = list(getattr(self, "_carry_prompt_img", None) or [])
        self._carry_prompt_img = []
        # Carry-Prompts-Forward appearance bundle (crops of the previous image's
        # boxes), set by next_image. Consumed once, like carry_prompt. Lets
        # YOLOE use a TRUE refer_image one-shot and SAM3 use crop-composite for
        # single-step Next Image carry instead of raw coordinates that would
        # land on background in the new image.
        carry_ref            = getattr(self, "_carry_ref_bundle", None)
        self._carry_ref_bundle = None
        # Drawn boxes serve both as prompts and as saved annotations under the
        # unified model. Type-agnostic helper picks up rect-form draws plus any
        # poly-form manuals carried over from a previous seg-mode pass.
        manual_boxes_img     = list(self.image_label.get_manual_anns_as_boxes_in_image_coords())

        det_boxes, yoloe_seg_results = self._run_detector(image_path, prompt, confidence, max_area, prompt_boxes_img + carry_prompt, ref=carry_ref)
        # Suppress detections on regions the user deleted earlier on this
        # image so a regenerate doesn't bring the same objects straight back.
        det_boxes, self._oneshot_polys_aligned = self._drop_rejected(
            det_boxes, self._oneshot_polys_aligned)
        det_key, _, is_standalone = self._detector_keys_for_pipeline()
        # Manual wins: drop detector boxes (and their aligned one-shot polys)
        # that duplicate a user-drawn box, keeping det_boxes<->polys aligned and
        # detector-first ordering so the standalone seg slice below holds.
        det_boxes, self._oneshot_polys_aligned = self._drop_detector_dups_of_manual(
            det_boxes, manual_boxes_img, self._oneshot_polys_aligned)
        # IoU-dedup so SAM's pass-through doesn't double-count drawn boxes.
        all_boxes, sources = self._combine_with_dedup(list(det_boxes), manual_boxes_img, 0.5)

        save_boxes_yolo(all_boxes, image_path, boxes_dir)

        if not all_boxes:
            if additive and prior_anns:
                # Nothing new detected, so keep the prior iteration's
                # annotations rather than blanking the canvas.
                self.base_cv2_image = img.copy()
                self._finalize_additive(prior_anns, image_path)
                self.image_label.update()
                return
            self.image_label.set_clean_image(img)
            self.image_label.annotation_boxes = []
            self.image_label.update()
            self.live_boxes = []
            self.live_box_sources = []
            return

        if is_standalone:
            # One-shot: use the aligned polys from _run_detector for the
            # detector portion. SAM2 fills in manual additions so the user
            # still sees masks for boxes they drew. Aligned polys may be
            # empty (no detections), in which case fall back to SAM on all
            # manual boxes if any exist.
            det_polys = list(self._oneshot_polys_aligned or [])
            manual_subset = all_boxes[len(det_polys):]
            sam_for_manual = None
            if manual_subset:
                try:
                    sam = self._get_model("sam2_t")
                    sam_for_manual = segment_with_boxes(sam, image_path, manual_subset)
                except Exception as e:
                    print(f"[one-shot] SAM2 fallback for manual boxes failed: {e}")
            sam_results = None  # signal to use det_polys + sam_for_manual below
        else:
            sam_results = self._run_segmenter(image_path, all_boxes)
        if is_standalone:
            # Compose ann_polys = det polys (aligned) + SAM polys for manual.
            det_polys = list(self._oneshot_polys_aligned or [])
            # The det portion of all_boxes is the first len(det_polys) entries
            # (set up in display_*_with_borders: detector first, manual after).
            ann_polys = list(det_polys)
            kept_indices = list(range(len(det_polys)))
            # Append manual polys from SAM2 fallback, keeping index alignment.
            if sam_for_manual is not None and sam_for_manual[0].masks is not None:
                for j, seg in enumerate(result_clean_polys(sam_for_manual[0])):
                    if seg is not None and len(seg) >= 3:
                        ann_polys.append(seg)
                        kept_indices.append(len(det_polys) + j)
            poly_sources = [sources[i] if i < len(sources) else 'detector' for i in kept_indices]
            self.live_boxes       = [all_boxes[i] for i in kept_indices]
            self.live_box_sources = [sources[i]   for i in kept_indices]
            self.live_polys_cache = ann_polys
            if not ann_polys:
                if additive and prior_anns:
                    self.base_cv2_image = img.copy()
                    self._finalize_additive(prior_anns, image_path)
                    self.image_label.update()
                    return
                self.image_label.set_clean_image(img)
                return
            # Render polygons manually (no sam_results to feed adjust_masks).
            image_with_borders = np.copy(img)
            h_img, w_img = img.shape[:2]
            for poly in ann_polys:
                pts = np.array([[int(x * w_img), int(y * h_img)] for x, y in poly], dtype=np.int32)
                cv2.drawContours(image_with_borders, [pts], -1, (255, 0, 255), 2)
            # Save masks: write YOLO-style polygon segments. save_masks expects
            # a results object; for one-shot we hand-roll it.
            try:
                stem = os.path.splitext(os.path.basename(image_path))[0]
                with open(os.path.join(seg_dir, f"{stem}.txt"), "w") as f:
                    for poly in ann_polys:
                        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                        f.write(f"0 {coords}\n")
            except Exception as e:
                print(f"[one-shot] saving polygon segments failed: {e}")
        else:
            if sam_results is None or sam_results[0].masks is None:
                self.image_label.set_clean_image(img)
                return

            save_masks(sam_results, seg_dir, image_path)

            masks = adjust_masks(sam_results)
            image_with_borders = np.copy(img)
            for mask_i in masks:
                image_with_borders = overlay_with_borders(image_with_borders, mask_i, color=(255, 0, 255), thickness=2)

            ann_polys = []
            poly_sources = []
            kept_indices = []
            for i, seg in enumerate(result_clean_polys(sam_results[0])):
                if seg is not None and len(seg) >= 3:
                    ann_polys.append(seg)
                    poly_sources.append(sources[i] if i < len(sources) else 'detector')
                    kept_indices.append(i)
            self.live_boxes       = [all_boxes[i] for i in kept_indices]
            self.live_box_sources = [sources[i]   for i in kept_indices]
            self.live_polys_cache = ann_polys

        # Cache for mode switching
        self.base_cv2_image = img.copy()

        self.image_label.set_clean_image(img)
        kept_new = self.image_label.set_annotations(polys=ann_polys, poly_sources=poly_sources)
        # set_annotations may drop detector/manual polys that duplicate a sticky
        # hand-drawn mask; shrink live_boxes/sources/polys in lockstep so they
        # stay index-aligned with the canvas annotations for _switch_to_mode
        # (prevents the purple-mask-loss drift).
        if kept_new is not None and len(kept_new) == len(ann_polys):
            if (self.live_polys_cache is not None
                    and len(self.live_polys_cache) == len(kept_new)):
                self.live_polys_cache = [p for p, k in zip(self.live_polys_cache, kept_new) if k]
            if len(self.live_boxes) == len(kept_new):
                self.live_box_sources = [s for s, k in zip(self.live_box_sources, kept_new) if k]
                self.live_boxes = [b for b, k in zip(self.live_boxes, kept_new) if k]
        # Bake the non-edit overlay from the SAME annotation set the canvas
        # edits, never from the raw SAM masks. The segmenter can emit masks
        # that have no clean polygon (result_clean_polys -> None for empty /
        # degenerate masks), and those were drawn into image_with_borders but
        # NOT stored as annotations. Toggling Edit Boxes (which renders from
        # annotations, then re-bakes via _render_annotations_overlay) then
        # silently dropped them for good. Routing the bake through the single
        # source of truth keeps every detector + segmenter segment intact
        # across the Edit toggle.
        baked = self._render_annotations_overlay()
        if baked is None:
            baked = image_with_borders
        self.baked_seg_cv2  = baked.copy()
        self.image_label.set_baked_image(baked)
        # Bucket promoted to durable annotations, so clear it.
        self.image_label.annotation_boxes = []
        self.image_label.update()
        self.seg_anns     = list(self.image_label.annotations)
        self.current_mode = "seg"

        # Additive regenerate: fold prior-iteration segments back in so they
        # survive a regenerate triggered after drawing new boxes.
        if additive:
            self._finalize_additive(prior_anns, image_path)
            self.image_label.update()

    def show_result_image(self, cv2_image):
        """Convenience wrapper used by error handling paths."""
        self.image_label.set_baked_image(cv2_image)

    def _toggle_resize_mode(self, checked):
        # Image Resize PARKS whatever mode is active without clearing it: an
        # in-progress draw draft OR a mask edit (points, cyan preview, selection,
        # vertex handles) is preserved so the user can zoom/pan and keep going.
        # All canvas input is gated off while resizing (mouse/keys check
        # _resize_mode), so nothing is drawn, committed, or cleared; everything
        # resumes when resize turns off. The zoom/pan itself persists afterward.
        self.image_label.set_resize_mode(checked)
        self.resize_btn.setText("Image Resize: ON" if checked else "Image Resize: OFF")

    # Draw Boxes split button: tool select + toggle
    # Label + tooltip per active tool. Tooltip carries the hover instructions
    # that used to live in the on-canvas banner.
    # Display labels are decoupled from the internal tool keys: "autodraw" is the
    # single-point tool shown as "Semi-Auto Points"; "semiauto" is the multi-point
    # curve tool shown as "Manual Masks".
    _DRAW_TOOL_LABEL = {"box": "Draw Boxes", "autodraw": "Semi-Auto Points",
                        "semiauto": "Manual Masks"}
    _DRAW_TOOL_TIP   = {"box": TOOLTIP_BOX, "autodraw": TOOLTIP_AUTODRAW,
                        "semiauto": TOOLTIP_SEMIAUTO}

    def _update_draw_btn_label(self):
        """Button text + hover tooltip reflect the active tool + on/off.
        Orange-on is handled by tool_toggle_qss via the :checked state."""
        on = self.draw_btn.isChecked()
        name = self._DRAW_TOOL_LABEL.get(self._draw_tool, "Draw Boxes")
        self.draw_btn.setText(f"{name}: {'ON' if on else 'OFF'}")
        self.draw_btn.setToolTip(self._DRAW_TOOL_TIP.get(self._draw_tool, TOOLTIP_BOX))

    def _select_draw_tool(self, tool, activate=False):
        """Dropdown picked a tool for the Draw button. Switching tools first
        turns the button off so the previously active tool is deactivated.
        activate=True (the user picked it from the menu) turns the button ON so
        the tool engages immediately; internal fallbacks pass activate=False to
        re-arm without drawing."""
        if tool not in ("box", "autodraw", "semiauto"):
            return
        if activate and tool != self._draw_tool and not self._guard_tool_switch():
            # Aborted: leave the menu pointing at the current tool.
            self.draw_tool_box_action.setChecked(self._draw_tool == "box")
            if hasattr(self, "autodraw_action"):
                self.autodraw_action.setChecked(self._draw_tool == "autodraw")
            self.mask_draw_action.setChecked(self._draw_tool == "semiauto")
            return
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            if self.draw_btn.isChecked():
                self.draw_btn.setChecked(False)  # deactivates the old tool
            self._draw_tool = tool
            # Keep the menu checkmarks authoritative (QActionGroup is exclusive,
            # but set explicitly so programmatic switches stay in sync).
            self.draw_tool_box_action.setChecked(tool == "box")
            if hasattr(self, "autodraw_action"):
                self.autodraw_action.setChecked(tool == "autodraw")
            self.mask_draw_action.setChecked(tool == "semiauto")
            if activate:
                self.draw_btn.setChecked(True)   # engage the selected tool now
        finally:
            self._in_mode_switch = was
        self._update_draw_btn_label()

    def _toggle_draw_btn(self, checked):
        """Main Draw button toggled; activates the currently selected tool."""
        if not self._guard_tool_switch():
            self.draw_btn.blockSignals(True)
            self.draw_btn.setChecked(not checked)
            self.draw_btn.blockSignals(False)
            self._update_draw_btn_label()
            return
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            # Mutually exclusive with the other canvas modes.
            if checked:
                # Image Resize is deliberately NOT in this list: it has
                # priority and PARKS the draw tool rather than cancelling it,
                # so the user can zoom/pan with Draw still armed and resume
                # drawing the instant they turn Resize off (no re-press needed).
                # All draw input is gated on _resize_mode, so a parked draw
                # never actually fires.
                for n in ("edit_btn", "multi_select_btn"):
                    b = getattr(self, n, None)
                    if b is not None and b.isChecked():
                        b.setChecked(False)
            if self._draw_tool in ("semiauto", "autodraw"):
                self.image_label.set_draw_mode(False)  # ensure box-draw off
                # SAM masks belong in the Segmentation view; flip to it when
                # activated from the Bounding Box view so the committed polygon
                # survives seg<->bbox round trips.
                if checked and self.current_mode != "seg" and hasattr(self, "mask_checkbox"):
                    self.mask_checkbox.setChecked(True)
                self.image_label.set_mask_draw_mode(checked, kind=self._draw_tool)
                if not checked:
                    self.image_label.clear_mask_session()
            else:  # box tool
                self.image_label.set_mask_draw_mode(False)  # ensure SAM draw off
                self.image_label.set_draw_mode(checked)
        finally:
            self._in_mode_switch = was
        self._update_draw_btn_label()

    def _refresh_mask_draw_enabled(self):
        """Enable the Semi-Automatic Segmentation tool whenever an interactive
        SAM model is active (SAM2/SAM3 segmenter, or SAM3 one-shot detector).
        Greyed for the YOLOE standalone detectors and segmenter (none). If
        semi-auto is the active tool but the model stops being SAM, fall the
        button back to Draw Boxes."""
        if not hasattr(self, "mask_draw_action"):
            return
        has_images = bool(getattr(self, "images", None))
        sam_active = self._active_interactive_sam_key() is not None
        enabled = has_images and sam_active
        self.mask_draw_action.setEnabled(enabled)
        if hasattr(self, "autodraw_action"):
            self.autodraw_action.setEnabled(enabled)
        if not enabled and self._draw_tool in ("semiauto", "autodraw"):
            self._select_draw_tool("box")
        # Edit Masks works on ANY polygon, model-generated masks included,
        # so it only needs images + at least one editable mask, NOT a SAM
        # model. Vertex editing is always available; the SAM "points" re-run is
        # gated separately (settings dialog) on a live SAM model.
        if hasattr(self, "semiauto_edit_action"):
            edit_ok = (has_images and self.image_label.has_editable_masks())
            self.semiauto_edit_action.setEnabled(edit_ok)
            # If mask-editing is the live mode but no longer available, fall the
            # Edit button back to Boxes (without force-activating it).
            if not edit_ok and getattr(self, "_edit_tool", "boxes") == "masks":
                self._select_edit_tool("boxes")

    def _update_edit_btn_label(self):
        """Button text reflects the selected edit mode + on/off. The purple
        :checked colour (tool_toggle_qss) shows which mode is live."""
        on = self.edit_btn.isChecked()
        name = "Edit Masks" if self._edit_tool == "masks" else "Edit Boxes"
        self.edit_btn.setText(f"{name}: {'ON' if on else 'OFF'}")

    def _select_edit_tool(self, tool, activate=False):
        """Edit dropdown picked a mode ("boxes" | "masks"). Switching while a
        mode is live drops the old one first; activate=True (the user picked it)
        turns the button ON so the chosen mode engages immediately."""
        if tool not in ("boxes", "masks"):
            return
        if activate and tool != self._edit_tool and not self._guard_tool_switch():
            self.edit_tool_boxes_action.setChecked(self._edit_tool == "boxes")
            if hasattr(self, "semiauto_edit_action"):
                self.semiauto_edit_action.setChecked(self._edit_tool == "masks")
            return
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            if self.edit_btn.isChecked():
                self.edit_btn.setChecked(False)   # deactivates the old mode
            self._edit_tool = tool
            self.edit_tool_boxes_action.setChecked(tool == "boxes")
            if hasattr(self, "semiauto_edit_action"):
                self.semiauto_edit_action.setChecked(tool == "masks")
            if activate:
                self.edit_btn.setChecked(True)    # fires _toggle_edit_btn
        finally:
            self._in_mode_switch = was
        self._update_edit_btn_label()

    def _toggle_edit_btn(self, checked):
        """Edit button toggled; engage the currently selected edit mode."""
        if not self._guard_tool_switch():
            self.edit_btn.blockSignals(True)
            self.edit_btn.setChecked(not checked)
            self.edit_btn.blockSignals(False)
            self._update_edit_btn_label()
            return
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            if self._edit_tool == "masks":
                self._set_edit_masks(checked)
            else:
                self._set_edit_boxes(checked)
        finally:
            self._in_mode_switch = was
        self._update_edit_btn_label()

    def _set_edit_masks(self, checked):
        """Engage/disengage drawn-mask (semi-auto) editing. Mutually exclusive
        with the draw tools / resize / multi-select; needs the Segmentation view."""
        if checked:
            for n in ("draw_btn", "resize_btn", "multi_select_btn"):
                b = getattr(self, n, None)
                if b is not None and b.isChecked():
                    b.setChecked(False)
            if self.current_mode != "seg" and hasattr(self, "mask_checkbox"):
                self.mask_checkbox.setChecked(True)
            # Default to VERTEX editing: clicking a mask immediately shows its
            # existing outline vertices as draggable square handles to reshape,
            # instead of dropping the user into the SAM-points tool (which
            # starts empty and makes them click brand-new points). The
            # SAM-points re-run is still reachable per-mask via the settings
            # popup (press S) when a SAM2/SAM3 model is active.
            self.image_label.set_semiauto_edit_target("vertices")
        self.image_label.set_semiauto_edit_mode(checked)

    def _apply_semiauto_edit(self):
        """Apply the in-progress edit to the selected semi-auto mask, persist,
        and deselect. In "points" mode the polygon comes from the SAM preview
        and sam_points are refreshed; in "vertices" mode ann['data'] was edited
        in place, so we just keep it."""
        idx = self.image_label.get_semiauto_selected_index()
        if idx is None:
            return
        anns = self.image_label.annotations
        if idx >= len(anns):
            return
        ann = anns[idx]
        self.image_label.push_undo_semiauto_edit()
        target = self.image_label.get_semiauto_edit_target()
        ow = self.image_label._orig_w or 0
        oh = self.image_label._orig_h or 0
        if target == "points":
            poly = self.image_label.get_mask_preview()
            if not poly or len(poly) < 3:
                return
            ann['data'] = [list(p) for p in poly]
            if ow and oh:
                ann['sam_points'] = [[x / ow, y / oh, lab] for (x, y), lab
                                     in self.image_label.get_mask_points_image_coords()]
        else:  # vertices: ann['data'] is already the edited polygon
            if len(ann['data']) < 3:
                return
        # Once the user hand-edits a mask it becomes THEIR mask: tag it sticky
        # so set_annotations carries it untouched across a later Regenerate
        # instead of replacing it with fresh detector output (manual work wins).
        ann['semiauto'] = True
        ann.setdefault('sam_points', [])
        # Sticky SAM masks aren't in live_boxes (carried via set_annotations), so
        # there's nothing to realign; the edited ann persists as-is.
        self.image_label.clear_semiauto_selection()
        self.image_label.update()
        self.image_label.boxes_changed.emit()
        self._rebake_overlay()
        self._persist_annotations(silent=True)

    def _delete_semiauto_mask(self):
        """Delete the whole selected drawn mask (X badge / Delete key), persist,
        and deselect; mirrors _apply_semiauto_edit's teardown."""
        idx = self.image_label.get_semiauto_selected_index()
        if idx is None:
            return
        anns = self.image_label.annotations
        if idx >= len(anns):
            return
        self.image_label._push_undo()
        anns[idx]['deleted'] = True
        self.image_label.clear_semiauto_selection()
        self.image_label.update()
        self.image_label.boxes_changed.emit()
        self._rebake_overlay()
        self._persist_annotations(silent=True)
        # May have removed the last mask, so refresh the edit gate (and fall the
        # Edit button back to Boxes if nothing is left to edit).
        self._refresh_mask_draw_enabled()

    @staticmethod
    def _simplify_poly(points, eps):
        """Ramer-Douglas-Peucker simplification of a closed polygon given as
        normalized [[x,y],...]. eps is the distance tolerance (0-1 normalized).
        Pure-python so it needs no cv2. Always returns >= 3 points."""
        if eps <= 0 or len(points) <= 3:
            return [list(p) for p in points]

        def _rdp(pts):
            if len(pts) < 3:
                return pts
            ax, ay = pts[0]; bx, by = pts[-1]
            dx, dy = bx - ax, by - ay
            denom = (dx * dx + dy * dy) ** 0.5
            dmax, idx = -1.0, 0
            for i in range(1, len(pts) - 1):
                px, py = pts[i]
                if denom == 0:
                    d = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
                else:
                    d = abs(dx * (ay - py) - (ax - px) * dy) / denom
                if d > dmax:
                    dmax, idx = d, i
            if dmax > eps:
                left = _rdp(pts[:idx + 1])
                right = _rdp(pts[idx:])
                return left[:-1] + right
            return [pts[0], pts[-1]]

        simplified = _rdp([list(p) for p in points])
        # _rdp drops the closing duplicate; guarantee a valid polygon.
        if len(simplified) < 3:
            return [list(p) for p in points]
        return simplified

    def _on_min_vertex_delete(self):
        """User tried to remove a vertex from a 3-point mask. A polygon can't
        go below 3 points, so offer to delete the whole mask instead."""
        if self.image_label.get_semiauto_selected_index() is None:
            return
        box = QtWidgets.QMessageBox(self)
        box.setStyleSheet("QLabel { color: black; font-size: 18px; } "
                          "QMessageBox { background-color: white; }")
        box.setWindowTitle("Delete mask?")
        box.setText("A mask can't have fewer than 3 points.\n"
                    "Delete the whole mask instead?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)
        if box.exec_() == QtWidgets.QMessageBox.Yes:
            self._delete_semiauto_mask()

    def _on_mask_selected(self):
        """Vertex-edit selection: auto-thin a dense model contour (SAM3 /
        YOLOE-seg masks can carry hundreds of points) down to a workable
        number of draggable handles. The full-detail original stays in the
        Esc-revert snapshot, so nothing is lost; press Esc to get it back."""
        if self.image_label.get_semiauto_edit_target() != "vertices":
            return
        idx = self.image_label.get_semiauto_selected_index()
        if idx is None or idx >= len(self.image_label.annotations):
            return
        ann = self.image_label.annotations[idx]
        data = ann.get('data') or []
        target = 30                      # aim for ~this many handles
        if len(data) <= target:
            return                       # already sparse, leave it alone
        # Ramp the RDP tolerance until the handle count is manageable. eps is a
        # normalized (fraction-of-image) distance, so it's resolution-agnostic.
        eps, simplified = 0.003, list(data)
        for _ in range(7):
            simplified = self._simplify_poly(data, eps)
            if len(simplified) <= target:
                break
            eps *= 1.7
        if 3 <= len(simplified) < len(data):
            ann['data'] = simplified
            self.image_label.set_mask_preview([list(p) for p in simplified])
            self.image_label.update()
            print(f"[edit] auto-simplified mask {idx}: "
                  f"{len(data)} -> {len(simplified)} vertices")

    def _open_semiauto_settings(self):
        """Per-mask settings dialog for the selected semi-auto mask: edit target
        (SAM points vs polygon vertices), class id, and polygon simplification."""
        idx = self.image_label.get_semiauto_selected_index()
        if idx is None or idx >= len(self.image_label.annotations):
            return
        ann = self.image_label.annotations[idx]
        dlg = SemiAutoSettingsDialog(
            self,
            target=self.image_label.get_semiauto_edit_target(),
            cls=int(ann.get('cls', 0)),
            points_enabled=self._active_interactive_sam_key() is not None,
        )

        def _apply_simplify():
            eps = dlg.simplify_eps()
            new_poly = self._simplify_poly(ann['data'], eps)
            ann['data'] = new_poly
            self.image_label.set_mask_preview([list(p) for p in new_poly])
            self.image_label.update()
        dlg.simplify_now.clicked.connect(_apply_simplify)

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            ann['cls'] = dlg.cls()
            self.image_label.set_semiauto_edit_target(dlg.target())
            # Re-seed the view for the chosen target.
            if dlg.target() == "points":
                self._on_mask_point_added()
            else:
                self.image_label.set_mask_preview([list(p) for p in ann['data']])
            self.image_label.update()

    def _sam_segment_points(self, image_path, pts_img, labels=None):
        """Segment ONE object from prompt points. `pts_img` = [(x, y), ...] in
        image pixels; `labels` = per-point 1=foreground / 0=background (defaults
        to all foreground). Returns an image-normalized polygon or None.

        Returns a LIST of image-normalized polygons (one per connected blob,
        largest first); [] if nothing. This is the single SEAM for every
        interactive SAM call. It routes through _segment_region (ROI crop)."""
        return self._segment_region(image_path, pts_img, labels)

    def _roi_image(self, image_path):
        """Pixels for the image we're drawing on. Reuses the already-decoded
        current image (base_cv2_image) when possible, else reads from disk."""
        base = getattr(self, "base_cv2_image", None)
        imgs = getattr(self, "images", None)
        idx = getattr(self, "current_image_index", 0)
        if (base is not None and imgs and idx < len(imgs)
                and imgs[idx] == image_path):
            return base
        return self._imread_cached(image_path)

    def _sam_points_call(self, key, source, pts_local, labels=None):
        """Raw SAM call for ONE object from prompt points (1=fore, 0=back).

        CRITICAL: ultralytics treats a flat (N, 2) points array as N separate
        one-point objects (predict.py `_prepare_prompts`: points[:, None, :]).
        To prompt a single object with N points we must nest one level deeper:
        points (1, N, 2) and labels (1, N), otherwise multi-point prompts
        produce N disjoint masks "nowhere near the points". `source` is a file
        path OR a cropped BGR ndarray. Returns the crop-normalized polygon."""
        points = [[[float(x), float(y)] for x, y in pts_local]]  # (1, N, 2) -> ONE object
        if labels is None:
            labels = [1] * len(pts_local)
        labels = [[int(l) for l in labels]]                      # (1, N) per-point labels
        try:
            res = self._get_model(key)(source, points=points, labels=labels,
                                       verbose=False, save=False)
        except Exception as e:
            print(f"[mask-draw] SAM point predict failed: {e}")
            return None
        if res is None or res[0].masks is None:
            return []
        # One polygon per connected blob (no bridge); disconnected pieces stay
        # separate masks; the caller commits each as its own annotation.
        return _mask_to_polys(res[0])

    def _segment_region(self, image_path, pts_img, labels=None, max_grows=2):
        """ROI-cropped interactive SAM: run the model only on the WINDOW around
        the clicked points, not the whole image, so bounding compute + peak memory
        to the area the user is working in (a top-left click never touches the
        bottom-right). Grows the window and retries if the mask reaches a crop
        edge (object bigger than the window). Returns an image-normalized polygon.

        Mirrors the crop -> predict -> translate-back pattern already used by
        _run_sam3_boxes_partitioned. FUTURE: this seam is also where a full-image
        embedding cache (A) or per-tile embedding cache (C) would slot in here; the
        callers and the coordinate contract stay the same."""
        key = self._active_interactive_sam_key()
        if key is None or not pts_img:
            return None
        img = self._roi_image(image_path)
        if img is None:
            return None
        H, W = img.shape[:2]
        xs = [p[0] for p in pts_img]; ys = [p[1] for p in pts_img]
        # Start with a generous window around the points; grow-and-retry covers
        # the rest. Single-point (auto draw) has zero spread, so the fixed floor
        # (12% of the short side) sets the initial window.
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        margin = max(spread * 0.6, 0.12 * min(W, H))
        for attempt in range(max_grows + 1):
            x1 = max(0, int(min(xs) - margin)); y1 = max(0, int(min(ys) - margin))
            x2 = min(W, int(max(xs) + margin)); y2 = min(H, int(max(ys) + margin))
            cw, ch = x2 - x1, y2 - y1
            if cw < 4 or ch < 4:
                return None
            full = (x1 <= 0 and y1 <= 0 and x2 >= W and y2 >= H)
            crop = img[y1:y2, x1:x2]
            local_pts = [(px - x1, py - y1) for px, py in pts_img]
            polys_crop = self._sam_points_call(key, crop, local_pts, labels)
            if not polys_crop:
                if full or attempt == max_grows:
                    return []
                margin *= 1.8
                continue
            # Grow if ANY blob hugs a crop edge that is NOT the image edge
            # (object likely extends past the window).
            eps = 0.01
            touches = any((cx <= eps and x1 > 0) or (cx >= 1 - eps and x2 < W)
                          or (cy <= eps and y1 > 0) or (cy >= 1 - eps and y2 < H)
                          for poly in polys_crop for cx, cy in poly)
            if touches and not full and attempt < max_grows:
                margin *= 1.8
                continue
            # Translate each crop-normalized blob -> image-normalized.
            return [[[(x1 + cx * cw) / W, (y1 + cy * ch) / H] for cx, cy in poly]
                    for poly in polys_crop]
        return []

    def _on_mask_point_added(self):
        """Re-run SAM and show the resulting mask(s) as a live preview. Fired on
        every point click / edit, honoring per-point labels (1=fore, 0=back).

        DRAW (Semi-Auto Points): each positive click is segmented as its OWN
        single-point object, the reliable "what's under the click" prompt,
        because pooling many spread-out points into one object makes SAM grab
        random stuff between them and destabilises earlier masks. Negatives
        refine the object they fall in; overlapping results are then unioned so
        touching objects merge and separated ones stay separate.

        EDIT (points target): keeps the single-object prompt + clip-to-outline so
        a dragged point reshapes one mask within its boundary."""
        pts = self.image_label.get_mask_points_image_coords()
        positives = [xy for xy, lab in pts if lab == 1]
        negatives = [xy for xy, lab in pts if lab == 0]
        if not self.images or not positives:
            self.image_label.set_mask_preview(None)
            return
        image_path = self.images[self.current_image_index]
        # Editing ONE existing mask: all points as a single prompt, largest blob,
        # clipped to the outline.
        if (self.image_label.semiauto_edit_mode
                and self.image_label.get_semiauto_edit_target() == "points"):
            pts_img = [xy for xy, _ in pts]
            labels  = [lab for _, lab in pts]
            polys = self._sam_segment_points(image_path, pts_img, labels)
            if not polys:
                self.image_label.set_mask_preview(None)
                return
            sam_poly = polys[0]
            if len(pts_img) >= 3:
                ow = self.image_label._orig_w or 1
                oh = self.image_label._orig_h or 1
                outline = [[x / ow, y / oh] for x, y in pts_img]
                clipped = self._clip_poly_to_outline(sam_poly, outline)
                if clipped and len(clipped) >= 3:
                    sam_poly = clipped
            self.image_label.set_mask_preview(sam_poly)
            return
        # Drawing: one reliable mask per positive click, then union overlaps.
        objs = self._segment_objects(image_path, positives, negatives)
        polys = self._union_polys(objs)
        if not polys:
            self.image_label.set_mask_preview(None)
            return
        self.image_label.set_mask_preview(polys[0])
        self.image_label.set_mask_preview_extra(polys[1:])

    def _segment_objects(self, image_path, positives, negatives):
        """Segment EACH positive click as its own single-point object (reliable),
        refining with any negative points that fall INSIDE that object's mask.
        Returns a list of image-normalized polygons (one+ per object). Stateless:
        adding a new click never perturbs the already-computed objects."""
        ow = self.image_label._orig_w or 1
        oh = self.image_label._orig_h or 1
        out = []
        for p in positives:
            base = self._sam_segment_points(image_path, [p], [1]) or []
            if not base:
                continue
            negs = [n for n in negatives if self._point_in_polys(n, base, ow, oh)]
            if negs:
                pts_img = [p] + negs
                labels  = [1] + [0] * len(negs)
                refined = self._sam_segment_points(image_path, pts_img, labels)
                out.extend(refined if refined else base)
            else:
                out.extend(base)
        return out

    @staticmethod
    def _point_in_polys(pt_img, polys_norm, ow, oh):
        """True if image-pixel point pt_img=(x,y) lies inside any normalized poly."""
        try:
            from shapely.geometry import Point as _Pt, Polygon as _Poly
            sp = _Pt(pt_img[0] / ow, pt_img[1] / oh)
            for poly in polys_norm:
                if poly and len(poly) >= 3:
                    pg = _Poly(poly)
                    if not pg.is_valid:
                        pg = pg.buffer(0)
                    if pg.contains(sp):
                        return True
            return False
        except Exception:
            return False

    @staticmethod
    def _union_polys(polys_norm):
        """Union overlapping/touching normalized polygons into separate connected
        pieces (largest first). Non-overlapping inputs come back separate. Falls
        back to the raw list if shapely is unavailable."""
        try:
            from shapely.geometry import Polygon as _Poly
            from shapely.ops import unary_union
            geoms = []
            for poly in polys_norm:
                if poly and len(poly) >= 3:
                    pg = _Poly(poly)
                    if not pg.is_valid:
                        pg = pg.buffer(0)
                    if pg.area > 0:
                        geoms.append(pg)
            if not geoms:
                return []
            u = unary_union(geoms)
            gs = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
            pieces = []
            for g in gs:
                if g.geom_type == "Polygon" and g.area > 0:
                    pieces.append((g.area, [[float(x), float(y)]
                                            for x, y in g.exterior.coords]))
            pieces.sort(key=lambda ap: ap[0], reverse=True)
            return [p for _a, p in pieces]
        except Exception:
            return [p for p in polys_norm if p and len(p) >= 3]

    @staticmethod
    def _clip_poly_to_outline(sam_poly, outline):
        """Clip a SAM polygon to the user's outline (both normalized). The mask
        follows SAM's segmentation but can NEVER extend past the drawn outline,
        so a foreground point that landed on leaf can't drag the whole leaf in.
        Returns the largest clipped piece, or None if nothing falls inside.
        Falls back to the unclipped SAM polygon if shapely is unavailable."""
        try:
            from shapely.geometry import Polygon as _P
            sp = _P(sam_poly); op = _P(outline)
            if not sp.is_valid: sp = sp.buffer(0)
            if not op.is_valid: op = op.buffer(0)
            inter = sp.intersection(op)
            if inter.is_empty:
                return None
            if inter.geom_type == "MultiPolygon":
                inter = max(inter.geoms, key=lambda g: g.area)
            if inter.geom_type != "Polygon" or inter.area <= 0:
                return None
            return [[float(x), float(y)] for x, y in inter.exterior.coords]
        except Exception:
            return sam_poly

    @staticmethod
    def _clean_polygon(poly_norm):
        """Return a valid simple polygon from possibly self-intersecting points
        (shapely buffer(0); largest piece). Falls back to the raw points if
        shapely is unavailable or the repair fails."""
        if not poly_norm or len(poly_norm) < 3:
            return None
        try:
            from shapely.geometry import Polygon as _P
            p = _P(poly_norm)
            if not p.is_valid:
                p = p.buffer(0)
            if p.is_empty:
                return [list(pt) for pt in poly_norm]
            if p.geom_type == "MultiPolygon":
                p = max(p.geoms, key=lambda g: g.area)
            if p.geom_type != "Polygon" or p.area <= 0:
                return [list(pt) for pt in poly_norm]
            return [[float(x), float(y)] for x, y in p.exterior.coords]
        except Exception:
            return [list(pt) for pt in poly_norm]

    def _close_mask_object(self):
        """Manual Masks: the user closed the outline. Run SAM on the connected
        points and CLIP the result to the outline. If SAM can't segment inside
        the outline (common on big or concave clusters, where a hand-drawn
        many-point outline often self-intersects), fall back to the polygon the
        user actually drew, so closing an outline ALWAYS yields a mask."""
        pts = self.image_label.get_mask_points_image_coords()
        if len(pts) < 3 or not self.images:
            return
        image_path = self.images[self.current_image_index]
        pts_img = [(x, y) for (x, y), _lab in pts]
        polys = self._sam_segment_points(image_path, pts_img)
        sam_poly = polys[0] if polys else None   # outline tool = one closed shape
        ow = self.image_label._orig_w or 1
        oh = self.image_label._orig_h or 1
        outline = [[x / ow, y / oh] for x, y in pts_img]
        poly = self._clip_poly_to_outline(sam_poly, outline) if sam_poly else None
        if not poly or len(poly) < 3:
            # SAM gave nothing usable inside the outline -> use the drawn shape
            # itself (repaired of any self-intersections). No dead-end alert.
            poly = self._clean_polygon(outline)
        if not poly or len(poly) < 3:
            self._styled_message(
                "Couldn't make a mask from that outline.\n\n"
                "Draw at least three points around the object and close it again.",
                "Manually Draw Masks").exec_()
            return
        self.image_label.set_mask_preview(poly)
        self._commit_mask_object()   # reads the preview, stores sam_points, clears session

    def _commit_mask_object(self):
        """Finalize the in-progress SAM mask(s) as saved manual segments, then
        start a fresh object. Triggered by closing a semi-auto outline, or by
        Enter in auto-draw mode. Disconnected blobs (primary + extras) each
        become their OWN separate mask, no bridge."""
        primary = self.image_label.get_mask_preview()
        commit_polys = [p for p in [primary] + self.image_label.get_mask_preview_extra()
                        if p and len(p) >= 3]
        if not commit_polys:
            return
        ow = self.image_label._orig_w or 0
        oh = self.image_label._orig_h or 0
        # SAM prompt points (NORMALIZED [x, y, label]) are stored on each piece so
        # "Edit Drawn Masks" can restore + re-run them. (Re-running points on a
        # split piece may regenerate all pieces; edit vertices to avoid that.)
        sam_points = []
        if ow and oh:
            for (x, y), lab in self.image_label.get_mask_points_image_coords():
                sam_points.append([x / ow, y / oh, lab])
        self.image_label._push_undo()
        for poly in commit_polys:
            ann = {
                'type': 'poly',
                'data': [list(p) for p in poly],
                'deleted': False,
                'source': 'manual',
                'semiauto': True,
                'sam_points': sam_points,
            }
            self.image_label.annotations.append(ann)
            # Hand-drawn masks WIN immediately: soft-delete any DETECTOR
            # annotation this piece duplicates (grid-limited to nearby detector
            # anns; mask IoU keeps clustered neighbours intact).
            cand_grid = SpatialGrid.build(
                [a for a in self.image_label.annotations
                 if a is not ann and not a.get('deleted') and a.get('source') == 'detector'],
                self.image_label._ann_bbox_norm_xyxy)
            for a in cand_grid.query_bbox(self.image_label._ann_bbox_norm_xyxy(ann)):
                if self.image_label._is_duplicate_of(a, ann):
                    a['deleted'] = True
        self.image_label.clear_mask_session()
        self.image_label.update()
        self.image_label.boxes_changed.emit()
        self._rebake_overlay()
        self._persist_annotations(silent=True)
        # Interactive semi-auto runs SAM repeatedly during a session (live
        # preview per click); release on commit so a heavy edit on one image
        # can't accumulate before the user advances.
        self._release_inference_memory()

    def _clear_manual_boxes(self):
        self.image_label.clear_boxes()

    def run_with_manual_boxes(self):
        """Skip the detector; feed manually drawn boxes straight into the current segmenter."""
        boxes = self.image_label.get_boxes_in_image_coords()
        if not boxes:
            return
        if not self.output_folder:
            msg = QtWidgets.QMessageBox()
            msg.setText("Please select an output folder first.")
            msg.exec_()
            return
        image_path = self.images[self.current_image_index]
        try:
            sam_results = self._run_segmenter(image_path, boxes)
            if sam_results is None:
                return
            save_masks(sam_results, self.output_folder + '/segments', image_path)

            img = self._imread_cached(image_path)
            masks = adjust_masks(sam_results)
            image_with_borders = np.copy(img)
            for mask_i in masks:
                image_with_borders = overlay_with_borders(image_with_borders, mask_i, color=(0, 255, 128), thickness=2)
            self.show_result_image(image_with_borders)
        except Exception as e:
            import traceback
            print(traceback.format_exc())


    def _set_edit_boxes(self, checked):
        # Entering edit mode turns off Image Resize so left-drag edits the
        # selection (handles) instead of panning.
        if checked and hasattr(self, 'resize_btn') and self.resize_btn.isChecked():
            self.resize_btn.setChecked(False)
        # Edit and the SAM draw tools both consume canvas clicks, so turn the SAM
        # draw tool off when entering edit (box-draw may coexist with edit).
        if checked and getattr(self, '_draw_tool', 'box') in ('semiauto', 'autodraw') \
                and self.draw_btn.isChecked():
            self.draw_btn.setChecked(False)
        # Leaving edit mode: re-bake the overlay so the non-edit display
        # reflects any deletions/edits made while editing (WYSIWYG).
        if not checked:
            self._rebake_overlay()
            # Multi-select rides on top of edit mode -- untoggle the
            # button so the on-screen state matches the underlying mode.
            if hasattr(self, 'multi_select_btn') and self.multi_select_btn.isChecked():
                # Block the signal so we don't recursively re-disable
                # edit mode through the multi-select handler.
                self.multi_select_btn.blockSignals(True)
                self.multi_select_btn.setChecked(False)
                self.multi_select_btn.blockSignals(False)
                self._apply_multi_select_btn_style(False)
                self.image_label.set_multi_select_mode(False)
        self.image_label.set_edit_mode(checked)

    # Synthetic image generation
    def _annotations_to_image_boxes(self):
        """Return active annotations as image-pixel xyxy BBOXES.

        Polygons are reduced to their bounding box -- callers that want
        the actual polygon shape should use _annotations_to_preserve_regions
        instead. Kept for callers that only need rectangles."""
        ow = self.image_label._orig_w or 0
        oh = self.image_label._orig_h or 0
        if not ow or not oh:
            return []
        boxes = []
        for ann in self.image_label.get_active_annotations():
            if ann["type"] == "rect":
                cx, cy, w_, h_ = ann["data"]
                x1 = (cx - w_ / 2) * ow
                y1 = (cy - h_ / 2) * oh
                x2 = (cx + w_ / 2) * ow
                y2 = (cy + h_ / 2) * oh
            else:  # poly -- collapse to bbox
                xs = [p[0] for p in ann["data"]]
                ys = [p[1] for p in ann["data"]]
                if not xs or not ys:
                    continue
                x1 = min(xs) * ow; x2 = max(xs) * ow
                y1 = min(ys) * oh; y2 = max(ys) * oh
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
        return boxes

    def _annotations_to_preserve_regions(self):
        """Return (boxes_pixel, polys_pixel) for the live annotations,
        keeping polygons at full shape fidelity (not reduced to bboxes).

        Used by the single-image variation flow so segmentation
        annotations get pixel-accurate preservation in the SD mask --
        otherwise SD would re-inpaint the parts inside the bbox but
        outside the actual segment, producing visible rectangular
        cutouts around organic shapes."""
        ow = self.image_label._orig_w or 0
        oh = self.image_label._orig_h or 0
        if not ow or not oh:
            return [], []
        boxes, polys = [], []
        for ann in self.image_label.get_active_annotations():
            if ann["type"] == "rect":
                cx, cy, w_, h_ = ann["data"]
                x1 = (cx - w_ / 2) * ow
                y1 = (cy - h_ / 2) * oh
                x2 = (cx + w_ / 2) * ow
                y2 = (cy + h_ / 2) * oh
                if x2 - x1 >= 1 and y2 - y1 >= 1:
                    boxes.append([x1, y1, x2, y2])
            elif ann["type"] == "poly":
                pts = [[p[0] * ow, p[1] * oh] for p in ann["data"]]
                if len(pts) >= 3:
                    polys.append(pts)
        return boxes, polys

    def _synth_dirs(self):
        """Ensure `synthetic images/{images,labels,segments}/` exists under
        the output folder and return all three paths. Segments folder
        only gets populated when polygon annotations exist (mirrors how
        the main output works -- `segments/` stays empty on bbox-only
        runs)."""
        synth_dir = os.path.join(self.output_folder, "synthetic images")
        img_dir   = os.path.join(synth_dir, "images")
        lbl_dir   = os.path.join(synth_dir, "labels")
        seg_dir   = os.path.join(synth_dir, "segments")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        return img_dir, lbl_dir, seg_dir

    def _save_variation(self, image_path, variation_pil,
                        boxes_pixel=None, polys_pixel=None,
                        copy_box_label=None, copy_seg_label=None):
        """Write the variation image + label file(s) under synthetic images/.

        Two modes:
          (a) DERIVE FROM LIVE: pass `boxes_pixel` and/or `polys_pixel`
              (image-pixel coords). They get written as normalized YOLO
              labels into labels/ (always) and segments/ (only if polys
              are present). Used by the single-image flow.
          (b) COPY FROM DISK: pass `copy_box_label` and/or
              `copy_seg_label` -- absolute paths to existing YOLO files
              that get verbatim-copied. Used by the batch flow so the
              synthetic dataset inherits the source's saved labels.

        Returns the absolute path of the saved .jpg."""
        img_dir, lbl_dir, seg_dir = self._synth_dirs()
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_img = os.path.join(img_dir, f"variation_{stem}.jpg")
        out_lbl = os.path.join(lbl_dir, f"variation_{stem}.txt")
        out_seg = os.path.join(seg_dir, f"variation_{stem}.txt")
        variation_pil.convert("RGB").save(out_img, quality=95)
        iw, ih = variation_pil.size
        # COPY-FROM-DISK mode: shutil.copyfile preserves the source's
        # normalized coords. Variation is the same dimensions as the
        # source, so normalized labels stay valid as-is.
        if copy_box_label is not None and os.path.exists(copy_box_label):
            import shutil
            shutil.copyfile(copy_box_label, out_lbl)
        if copy_seg_label is not None and os.path.exists(copy_seg_label):
            import shutil
            shutil.copyfile(copy_seg_label, out_seg)
        # DERIVE-FROM-LIVE mode: write boxes (always) and segments
        # (only if polys exist) from image-pixel coords. Box labels
        # come from BOTH rect-source boxes AND the bbox of each poly,
        # so a segmentation-mode session still produces a usable bbox
        # label for downstream training.
        if boxes_pixel is not None or polys_pixel is not None:
            with open(out_lbl, "w") as f:
                for x1, y1, x2, y2 in (boxes_pixel or []):
                    cx = (x1 + x2) / 2 / iw
                    cy = (y1 + y2) / 2 / ih
                    bw = (x2 - x1) / iw
                    bh = (y2 - y1) / ih
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                # Also write a bbox row for each poly (bbox of the poly)
                # so the YOLO box file is complete.
                for poly in (polys_pixel or []):
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    cx = (x1 + x2) / 2 / iw
                    cy = (y1 + y2) / 2 / ih
                    bw = (x2 - x1) / iw
                    bh = (y2 - y1) / ih
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            if polys_pixel:
                with open(out_seg, "w") as f:
                    for poly in polys_pixel:
                        coords = " ".join(
                            f"{p[0] / iw:.6f} {p[1] / ih:.6f}" for p in poly
                        )
                        f.write(f"0 {coords}\n")
        return out_img

    def _on_generate_variation(self):
        """Single-image flow: generate one variation of the current
        image using the live annotations as the preserve mask, show a
        side-by-side preview, save to synthetic images/ if accepted."""
        if getattr(self, "_busy", False):
            return
        if not self.images:
            return
        if not self.output_folder:
            self.select_output_folder()
            if not self.output_folder:
                return
        boxes, polys = self._annotations_to_preserve_regions()
        if not boxes and not polys:
            msg = QtWidgets.QMessageBox(self)
            msg.setStyleSheet("QLabel { color: black; font-size: 18px; } "
                              "QMessageBox { background-color: white; }")
            msg.setWindowTitle("Nothing to preserve")
            msg.setText("Draw or run the model first; the variation "
                        "needs at least one annotated region to "
                        "preserve.")
            msg.exec_()
            return
        image_path = self.images[self.current_image_index]
        prompt = (self._sd_prompt or "").strip() or _SD_DEFAULT_PROMPT
        neg_prompt = (self._sd_neg or "").strip() or None

        self._busy = True
        self.gen_variation_btn.setEnabled(False)
        self.gen_variation_folder_btn.setEnabled(False)
        self.gen_variation_btn.setText("Generating...")
        QtWidgets.QApplication.processEvents()
        # Hand the window's model cache to generate_variation so the
        # SD loader can release YOLOE / DINO / SAM3 from memory before
        # pulling SD-1.5 in from disk. Without this, multi-GB dead model
        # weights stay parked in MPS's caching allocator and the SD
        # load swap-thrashes.
        generate_variation._extra_caches = [self._model_cache]
        try:
            variation, original = generate_variation(
                image_path,
                boxes_xyxy=boxes,
                polys_xyxy_pixel=polys,
                prompt=prompt,
                negative_prompt=neg_prompt,
                strength=self.sd_strength_slider.value() / 100.0,
            )
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            err = QtWidgets.QMessageBox(self)
            err.setStyleSheet("QLabel { color: black; font-size: 16px; } "
                              "QMessageBox { background-color: white; }")
            err.setWindowTitle("Variation failed")
            err.setText(f"Could not generate variation:\n{e}")
            err.exec_()
            return
        finally:
            self._busy = False
            self.gen_variation_btn.setText("Generate Variation")
            self._refresh_auto_annotate_enabled()

        def _regen():
            try:
                new_var, _ = generate_variation(
                    image_path,
                    boxes_xyxy=boxes,
                    polys_xyxy_pixel=polys,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    strength=self.sd_strength_slider.value() / 100.0,
                )
                return new_var
            except Exception as e:
                print(f"[generate_variation regen] {e}")
                return None

        dlg = VariationPreviewDialog(original, variation, parent=self, regenerate_cb=_regen)
        dlg.exec_()
        if dlg.accepted_save:
            try:
                out_path = self._save_variation(
                    image_path, dlg.variation,
                    boxes_pixel=boxes, polys_pixel=polys,
                )
                done = QtWidgets.QMessageBox(self)
                done.setStyleSheet("QLabel { color: black; font-size: 16px; } "
                                   "QMessageBox { background-color: white; }")
                done.setWindowTitle("Variation saved")
                done.setText(f"Saved to:\n{out_path}")
                done.exec_()
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                err = QtWidgets.QMessageBox(self)
                err.setWindowTitle("Save failed")
                err.setText(f"Could not save variation:\n{e}")
                err.exec_()

    def _on_generate_variations_batch(self):
        """Batch flow: generate one variation per image in the folder
        that already has a saved label file (boxes/<stem>.txt), write
        them all to synthetic images/, then open BatchVariationViewer
        so the user can flip through and prune."""
        if getattr(self, "_busy", False):
            return
        if not self.images:
            return
        if not self.output_folder:
            self.select_output_folder()
            if not self.output_folder:
                return
        boxes_dir = os.path.join(self.output_folder, "boxes")
        if not os.path.isdir(boxes_dir):
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("No labels found")
            msg.setText("No 'boxes/' subfolder in the output. Run the "
                        "model or Auto Annotate Remaining first so the "
                        "synthetic variations have labels to inherit.")
            msg.exec_()
            return

        # Build the work list: every image with a corresponding label.
        def _label_for(img):
            stem = os.path.splitext(os.path.basename(img))[0]
            p = os.path.join(boxes_dir, f"{stem}.txt")
            return p if os.path.exists(p) else None
        targets = [(img, _label_for(img)) for img in self.images]
        targets = [(img, lbl) for img, lbl in targets if lbl is not None]
        if not targets:
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("No labels found")
            msg.setText("None of the images in the folder have saved "
                        "labels yet. Run the model or Auto Annotate "
                        "Remaining first.")
            msg.exec_()
            return

        prompt = (self._sd_prompt or "").strip() or _SD_DEFAULT_PROMPT
        neg_prompt = (self._sd_neg or "").strip() or None

        self._busy = True
        for btn in (self.gen_variation_btn, self.gen_variation_folder_btn,
                    getattr(self, "auto_annotate_btn", None),
                    getattr(self, "regen_btn", None),
                    getattr(self, "next_btn", None)):
            if btn is not None:
                btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        progress = QtWidgets.QProgressDialog(
            "Generating variations...", "Cancel", 0, len(targets), self)
        progress.setWindowTitle("Variations for Folder")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setStyleSheet("QLabel { color: black; } QProgressDialog { background-color: white; }")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        # Release detector / segmenter models before SD loads -- see the
        # comment in _on_generate_variation for the rationale.
        generate_variation._extra_caches = [self._model_cache]
        saved_paths = []
        failed = []
        canceled = False
        for idx, (image_path, label_path) in enumerate(targets):
            if progress.wasCanceled():
                canceled = True
                break
            progress.setLabelText(
                f"Image {idx + 1} of {len(targets)}: {os.path.basename(image_path)}"
            )
            progress.setValue(idx)
            QtWidgets.QApplication.processEvents()
            try:
                # Read the SAVED labels (not live annotations) so each
                # variation matches what was actually written to disk
                # for that image. Polygon labels live in segments/ and
                # are preferred when present -- they preserve the
                # actual object shape in the SD mask. Box labels in
                # boxes/ are used either as the only source (bbox-only
                # session) or as a complement.
                img_pil = Image.open(image_path).convert("RGB")
                iw, ih = img_pil.size
                stem = os.path.splitext(os.path.basename(image_path))[0]
                seg_path = os.path.join(self.output_folder, "segments", f"{stem}.txt")
                boxes, polys = [], []
                # Polys from segments/<stem>.txt if present.
                if os.path.exists(seg_path):
                    with open(seg_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 7:  # cls + at least 3 (x,y) pairs
                                continue
                            try:
                                coords = [float(x) for x in parts[1:]]
                            except ValueError:
                                continue
                            if len(coords) % 2 != 0:
                                continue
                            poly = [[coords[i] * iw, coords[i + 1] * ih]
                                    for i in range(0, len(coords), 2)]
                            if len(poly) >= 3:
                                polys.append(poly)
                # Boxes from label_path (the boxes/<stem>.txt the
                # outer loop already located).
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        except ValueError:
                            continue
                        x1 = (cx - bw / 2) * iw
                        y1 = (cy - bh / 2) * ih
                        x2 = (cx + bw / 2) * iw
                        y2 = (cy + bh / 2) * ih
                        boxes.append([x1, y1, x2, y2])
                if not boxes and not polys:
                    failed.append((image_path, "empty label file"))
                    continue
                variation, _ = generate_variation(
                    image_path,
                    boxes_xyxy=boxes,
                    polys_xyxy_pixel=polys,
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    strength=self.sd_strength_slider.value() / 100.0,
                )
                out_path = self._save_variation(
                    image_path, variation,
                    copy_box_label=label_path,
                    copy_seg_label=seg_path if os.path.exists(seg_path) else None,
                )
                saved_paths.append(out_path)
            except Exception as e:
                failed.append((image_path, str(e)[:80]))
                print(f"[variations-batch] {image_path}: FAILED {e}")

        progress.setValue(len(targets))
        progress.close()

        parts = [f"Generated: {len(saved_paths)}"]
        if canceled:
            parts.insert(0, "Run canceled before finishing.")
        if failed:
            parts.append(f"Failed: {len(failed)}")
            for path, err in failed[:5]:
                parts.append(f"  - {os.path.basename(path)}: {err}")
            if len(failed) > 5:
                parts.append(f"  ...and {len(failed) - 5} more")
        summary = QtWidgets.QMessageBox(self)
        summary.setStyleSheet("QLabel { color: white; font-size: 16px; } "
                              "QMessageBox { background-color: black; }")
        summary.setWindowTitle("Variations for Folder")
        summary.setText("\n".join(parts))
        summary.exec_()

        self._busy = False
        self._refresh_auto_annotate_enabled()

        if saved_paths:
            viewer = BatchVariationViewer(saved_paths, parent=self)
            viewer.exec_()

    def _apply_multi_select_btn_style(self, checked):
        self.multi_select_btn.setText("Select Multiple: ON" if checked else "Select Multiple: OFF")

    def _toggle_multi_select_mode(self, checked):
        if checked and not self._guard_tool_switch():
            self.multi_select_btn.blockSignals(True)
            self.multi_select_btn.setChecked(False)
            self.multi_select_btn.blockSignals(False)
            self._apply_multi_select_btn_style(False)
            return
        was = self._in_mode_switch
        self._in_mode_switch = True
        try:
            self._do_toggle_multi_select_mode(checked)
        finally:
            self._in_mode_switch = was

    def _do_toggle_multi_select_mode(self, checked):
        # Mutually exclusive with the Draw tool (box OR semi-auto). Untoggle
        # draw FIRST so the side effects below (edit auto-enable, style apply)
        # run with the canvas committed to selection-only behavior. Unchecking
        # draw_btn deactivates whichever draw tool is active.
        if checked and self.draw_btn.isChecked():
            self.draw_btn.setChecked(False)
        if checked and hasattr(self, 'resize_btn') and self.resize_btn.isChecked():
            self.resize_btn.setChecked(False)
        # Multi-select needs Edit Boxes on (it operates on the editable
        # annotation set + per-ann X badges only render in edit mode).
        # If the user toggled it on without edit, flip edit on too --
        # blockSignals around the programmatic check so _toggle_edit_btn
        # does not in turn untoggle multi-select (the linkage is one-way:
        # turning edit OFF disables multi-select, but turning multi
        # ON enables edit).
        if checked and not self.image_label.edit_mode:
            # Multi-select operates on the box/handle editor, so force the Edit
            # button into Boxes mode (not Masks) before enabling it.
            self._edit_tool = "boxes"
            self.edit_tool_boxes_action.setChecked(True)
            if hasattr(self, "semiauto_edit_action"):
                self.semiauto_edit_action.setChecked(False)
            self.edit_btn.blockSignals(True)
            self.edit_btn.setChecked(True)
            self.edit_btn.blockSignals(False)
            # Apply edit-mode side effects manually since we suppressed
            # the toggled signal.
            self.image_label.set_edit_mode(True)
            self._update_edit_btn_label()
        self._apply_multi_select_btn_style(checked)
        self.image_label.set_multi_select_mode(checked)

    def _undo_annotation(self):
        self.image_label.undo()
        self._rebake_overlay()
        self._persist_annotations(silent=True)

    def _redo_annotation(self):
        self.image_label.redo()
        self._rebake_overlay()
        self._persist_annotations(silent=True)

    def _save_and_confirm(self):
        """Save & Confirm button handler; persists annotations, shows a popup."""
        self._persist_annotations(silent=False)
        # Done editing this image: untoggle Image Resize (if on) and snap the
        # view back to its original (fit) size.
        if hasattr(self, 'resize_btn') and self.resize_btn.isChecked():
            self.resize_btn.setChecked(False)
        self.image_label.reset_view()

    def _persist_annotations(self, silent=False):
        """Write the current image's on-screen annotations to the box/segment
        label files. With silent=True the confirmation popup is skipped, used
        by Next Image so manual edits/new boxes are saved even when the user
        didn't press Save & Confirm first."""
        if not self.images or not self.output_folder:
            return
        active = self.image_label.get_saveable_annotations()
        image_path = self.images[self.current_image_index]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        img = self._imread_cached(image_path)
        h, w = img.shape[:2]

        seg_dir = self.output_folder + '/segments'
        box_dir = self.output_folder + '/boxes'
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(box_dir, exist_ok=True)

        # Preserve cross-mode model output: if the user is currently in
        # bbox display mode but a previous seg-mode run already produced
        # masks for this image, write them too. Truncating seg.txt to
        # empty would silently delete the model's segmentations. Symmetric
        # for rects/bbox_anns when in seg mode.
        polys = [a for a in active if a['type'] == 'poly']
        if not polys:
            polys = [a for a in (self.seg_anns or [])
                     if a['type'] == 'poly' and not a.get('deleted', False)
                     and a.get('source') != 'prompt']
        rects = [a for a in active if a['type'] == 'rect']
        if not rects:
            rects = [a for a in (self.bbox_anns or [])
                     if a['type'] == 'rect' and not a.get('deleted', False)
                     and a.get('source') != 'prompt']

        # Single-pass overwrite of both label files. The "in-flight
        # manual" boxes are bucket draws the user made AFTER the last
        # regenerate that haven't yet been promoted to rect anns; they
        # need to be saved too, so we collect them up front and write
        # everything in one 'w' block. No append phase -- this is the
        # only write to each label file per save.
        in_flight_manual = self.image_label.get_boxes_in_image_coords()
        with open(f'{seg_dir}/{stem}.txt', 'w') as sf, \
             open(f'{box_dir}/{stem}.txt', 'w') as bf:
            for ann in polys:
                poly = ann['data']
                cid = int(ann.get('cls', 0))
                coords = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly)
                sf.write(f'{cid} {coords}\n')
                xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                cx = (min(xs) + max(xs)) / 2; cy = (min(ys) + max(ys)) / 2
                bw = max(xs) - min(xs); bh = max(ys) - min(ys)
                bf.write(f'{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            for ann in rects:
                cx2, cy2, bw2, bh2 = ann['data']
                bf.write(f'{int(ann.get("cls", 0))} {cx2:.6f} {cy2:.6f} {bw2:.6f} {bh2:.6f}\n')
            for x1, y1, x2, y2 in in_flight_manual:
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                bf.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')

        # Round-trip sanity check: confirm what's on disk matches what was
        # actually written (the polys+rects collection above), not just the
        # canvas's current display.
        expected = []
        for ann in polys:
            xs = [p[0] for p in ann['data']]; ys = [p[1] for p in ann['data']]
            expected.append((min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h))
        for ann in rects:
            cx2, cy2, bw2, bh2 = ann['data']
            expected.append((
                (cx2 - bw2/2) * w, (cy2 - bh2/2) * h,
                (cx2 + bw2/2) * w, (cy2 + bh2/2) * h,
            ))
        expected.extend(in_flight_manual)
        ok, max_err = verify_boxes_round_trip(expected, image_path, box_dir, tol_px=1.5)
        if not ok:
            print(f"[ROUND-TRIP CHECK] FAILED for {stem}: max error {max_err:.3f}px")
        elif AUTOANNOTATE_DEBUG:
            print(f"[ROUND-TRIP CHECK] OK for {stem} (max error {max_err:.3f}px)")

        # Re-render overlay with only the remaining annotations
        overlay = img.copy()
        for ann in active:
            if ann['type'] == 'poly':
                pts = np.array([[int(x * w), int(y * h)] for x, y in ann['data']], dtype=np.int32)
                cv2.drawContours(overlay, [pts], -1, (255, 0, 255), 2)
            elif ann['type'] == 'rect':
                cx2, cy2, bw2, bh2 = ann['data']
                rx1 = int((cx2 - bw2 / 2) * w); ry1 = int((cy2 - bh2 / 2) * h)
                rx2 = int((cx2 + bw2 / 2) * w); ry2 = int((cy2 + bh2 / 2) * h)
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)

        # Update cached baked image for current mode
        if self.current_mode == "bbox":
            self.baked_bbox_cv2 = overlay.copy()
            self.bbox_anns = list(active)
        elif self.current_mode == "seg":
            self.baked_seg_cv2 = overlay.copy()
            self.seg_anns = list(active)

        self.image_label.set_baked_image(overlay)
        self._force_btn(self.edit_btn, False)

        # Save reference images split by kind: annotated/boxes/ gets a
        # bbox-only render, annotated/masks/ gets a polygon-only render.
        # Boxes are computed from poly bboxes + rect annotations + any
        # still-in-flight manual draws so the saved reference matches
        # what was written to the label files.
        ref_boxes = []
        for ann in polys:
            xs = [p[0] for p in ann['data']]; ys = [p[1] for p in ann['data']]
            ref_boxes.append([min(xs) * w, min(ys) * h,
                              max(xs) * w, max(ys) * h])
        for ann in rects:
            cx2, cy2, bw2, bh2 = ann['data']
            ref_boxes.append([(cx2 - bw2 / 2) * w, (cy2 - bh2 / 2) * h,
                              (cx2 + bw2 / 2) * w, (cy2 + bh2 / 2) * h])
        ref_boxes.extend(in_flight_manual)
        ref_polys = [a['data'] for a in polys]
        self._save_split_overlays(image_path, ref_boxes, ref_polys)

        if not silent:
            msg = QtWidgets.QMessageBox()
            msg.setStyleSheet("QLabel { color: black; } QMessageBox { background-color: white; }")
            msg.setText(f"Saved {len(active)} annotation(s) for {stem}.")
            msg.exec_()

    def _on_box_checked(self, state):
        if state == QtCore.Qt.Checked:
            # Leaving the Segmentation view would drop an in-progress mask.
            if not self._confirm_leave_unfinished("to the Bounding Box view"):
                self.box_checkbox.blockSignals(True)
                self.box_checkbox.setChecked(False)
                self.box_checkbox.blockSignals(False)
                return
            self.mask_checkbox.blockSignals(True)
            self.mask_checkbox.setChecked(False)
            self.mask_checkbox.blockSignals(False)
            self._switch_to_mode("bbox")

    def _on_mask_checked(self, state):
        if state == QtCore.Qt.Checked:
            self.box_checkbox.blockSignals(True)
            self.box_checkbox.setChecked(False)
            self.box_checkbox.blockSignals(False)
            self._switch_to_mode("seg")

    def _switch_to_mode(self, mode):
        """Switch display mode while preserving user edits.

        - bbox->seg: re-run segmenter on current `live_boxes` (so deletions and
          newly drawn boxes propagate as masks).
        - seg->bbox: render boxes from `live_boxes` directly (no detector re-run).
        """
        if mode == self.current_mode:
            return

        # Snapshot current state into live_boxes before we change anything.
        if self.image_label._orig_w is not None:
            if self.current_mode == "bbox":
                # Active rects already reflect deletions; manual draws are widget-only.
                rect_pairs = self.image_label.get_active_rects_with_sources()
                rects = [b for b, _ in rect_pairs]
                rect_sources = [s for _, s in rect_pairs]
                manual = self.image_label.get_boxes_in_image_coords()
                self.live_boxes = rects + manual
                self.live_box_sources = rect_sources + (['manual'] * len(manual))
            elif self.current_mode == "seg":
                # Cull live_boxes by the currently-not-deleted seg anns (index-aligned).
                kept = []
                kept_sources = []
                for i, ann in enumerate(self.image_label.annotations):
                    if not ann['deleted'] and i < len(self.live_boxes):
                        kept.append(self.live_boxes[i])
                        # Prefer the ann's source (authoritative) over the parallel
                        # array; they should agree, but the ann is what the user sees.
                        kept_sources.append(ann.get('source',
                            self.live_box_sources[i] if i < len(self.live_box_sources) else 'detector'))
                manual = self.image_label.get_boxes_in_image_coords()
                self.live_boxes = kept + manual
                self.live_box_sources = kept_sources + (['manual'] * len(manual))
            else:
                # No prior model run (fresh boot): capture manually drawn
                # boxes so switching display mode doesn't wipe them. They
                # live in self.annotations as source=='manual' rects.
                rect_pairs = self.image_label.get_active_rects_with_sources()
                self.live_boxes       = [b for b, _ in rect_pairs]
                self.live_box_sources = [s for _, s in rect_pairs]

        if self.base_cv2_image is None:
            return

        img = self.base_cv2_image
        h, w = img.shape[:2]

        if mode == "bbox":
            # Render bbox view from live_boxes, no model call needed.
            sources = list(self.live_box_sources) if self.live_box_sources else \
                      ['detector'] * len(self.live_boxes)
            colors = [(0, 200, 100) if s == 'manual' else (255, 0, 255) for s in sources]
            img_with_boxes = draw_boxes_on_image(img.copy(), self.live_boxes, colors=colors)
            rects_norm = []
            for x1, y1, x2, y2 in self.live_boxes:
                rects_norm.append([
                    (x1 + x2) / 2 / w,
                    (y1 + y2) / 2 / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                ])
            self.image_label.set_clean_image(img)
            # Only annotation_boxes get promoted (folded into live_boxes as rects)
            # prompt_boxes are user prompts that should survive mode switches
            # until the next regenerate consumes them. Direct assignment (not
            # clear_boxes) so we don't fire the boxes_changed signal mid-transition.
            self.image_label.annotation_boxes = []
            self.image_label.set_annotations(rects=rects_norm, rect_sources=sources)
            # Re-bake from the full annotation set (set_annotations carried the
            # sticky SAM masks) so they show as green outlines in box view too,
            # instead of vanishing. Detector rects render as magenta boxes.
            baked = self._render_annotations_overlay()
            if baked is None:
                baked = img_with_boxes
            self.image_label.set_baked_image(baked)
            self.baked_bbox_cv2 = baked.copy()
            self.current_mode = "bbox"
            self._refresh_mask_draw_enabled()
            return

        if mode == "seg":
            if not self.live_boxes:
                # No detector boxes to segment, but the user may have hand-drawn
                # sticky SAM masks, so render those instead of bailing (otherwise
                # they'd seem to vanish when there's no detector output).
                if self.image_label.has_semiauto_masks():
                    self.image_label.set_clean_image(img)
                    baked = self._render_annotations_overlay()
                    if baked is not None:
                        self.image_label.set_baked_image(baked)
                        self.baked_seg_cv2 = baked.copy()
                    self.current_mode = "seg"
                    self._refresh_mask_draw_enabled()
                return
            image_path = self.images[self.current_image_index]
            try:
                # YOLOE-seg standalone: prefer cached polygons if we still have them
                # AND the box count matches (no edits since last run).
                _, _, is_standalone = self._detector_keys_for_pipeline()
                ann_polys = None
                # Snapshot the input box list/sources so we can shrink them in
                # lockstep with any SAM-dropped masks (same alignment fix as
                # display_masks_with_borders).
                input_boxes   = list(self.live_boxes)
                input_sources = list(self.live_box_sources) if self.live_box_sources else \
                                ['detector'] * len(input_boxes)
                if is_standalone:
                    # Cache covers the detector portion of live_boxes (first
                    # N entries); manual additions need SAM2 to get masks.
                    det_polys = list(self.live_polys_cache or [])
                    n_det = len(det_polys)
                    manual_boxes_only = input_boxes[n_det:]
                    sam_results = None  # not used for assembly below
                    ann_polys = list(det_polys)
                    poly_sources = list(input_sources[:n_det])
                    if manual_boxes_only:
                        try:
                            sam = self._get_model("sam2_t")
                            sam_r = segment_with_boxes(sam, image_path, manual_boxes_only)
                            if sam_r is not None and sam_r[0].masks is not None:
                                kept_manual_idx = []
                                for j, seg in enumerate(result_clean_polys(sam_r[0])):
                                    if seg is not None and len(seg) >= 3:
                                        ann_polys.append(seg)
                                        kept_manual_idx.append(n_det + j)
                                poly_sources += [input_sources[i] for i in kept_manual_idx
                                                 if i < len(input_sources)]
                                self.live_boxes       = input_boxes[:n_det] + [input_boxes[i] for i in kept_manual_idx]
                                self.live_box_sources = input_sources[:n_det] + [input_sources[i] for i in kept_manual_idx
                                                                                 if i < len(input_sources)]
                        except Exception as e:
                            print(f"[one-shot switch] SAM2 fallback failed: {e}")
                            # Fall back: draw manual rects as 4-point polygons.
                            for i, (x1, y1, x2, y2) in enumerate(manual_boxes_only):
                                img_h, img_w = img.shape[:2]
                                rect_poly = [
                                    [x1 / img_w, y1 / img_h],
                                    [x2 / img_w, y1 / img_h],
                                    [x2 / img_w, y2 / img_h],
                                    [x1 / img_w, y2 / img_h],
                                ]
                                ann_polys.append(rect_poly)
                                if n_det + i < len(input_sources):
                                    poly_sources.append(input_sources[n_det + i])
                else:
                    sam_results = self._run_segmenter(image_path, self.live_boxes)
                    if sam_results is None or sam_results[0].masks is None:
                        return
                    ann_polys = []
                    poly_sources = []
                    kept_indices = []
                    for i, seg in enumerate(result_clean_polys(sam_results[0])):
                        if seg is not None and len(seg) >= 3:
                            ann_polys.append(seg)
                            poly_sources.append(input_sources[i]
                                                if i < len(input_sources) else 'detector')
                            kept_indices.append(i)
                    # Shrink live_boxes/sources to only the kept ones so the
                    # next regenerate sees the right manual set.
                    self.live_boxes       = [input_boxes[i]   for i in kept_indices]
                    self.live_box_sources = [input_sources[i] for i in kept_indices]
                self.live_polys_cache = ann_polys

                masks_overlay = img.copy()
                if sam_results is not None:
                    masks = adjust_masks(sam_results)
                    for mask_i in masks:
                        masks_overlay = overlay_with_borders(masks_overlay, mask_i,
                                                             color=(255, 0, 255), thickness=2)
                else:
                    # Render polygons by hand from cache.
                    for poly in ann_polys:
                        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], dtype=np.int32)
                        cv2.drawContours(masks_overlay, [pts], -1, (255, 0, 255), 2)

                # poly_sources was built above (in lockstep with kept SAM masks).
                self.image_label.set_clean_image(img)
                # Only annotation_boxes get promoted; prompt_boxes (yellow)
                # are user prompts that persist across mode switches.
                self.image_label.annotation_boxes = []
                self.image_label.set_annotations(polys=ann_polys, poly_sources=poly_sources)
                # Re-bake from the full annotation set (set_annotations carried the
                # sticky SAM masks) so the user's hand-authored masks render here
                # too, not just the freshly-segmented detector masks.
                baked = self._render_annotations_overlay()
                if baked is None:
                    baked = masks_overlay
                self.image_label.set_baked_image(baked)
                self.baked_seg_cv2 = baked.copy()
                self.current_mode = "seg"
                self._refresh_mask_draw_enabled()
                # Persist mask file for the current edited box set.
                if self.output_folder and sam_results is not None:
                    seg_dir = os.path.join(self.output_folder, 'segments')
                    os.makedirs(seg_dir, exist_ok=True)
                    save_masks(sam_results, seg_dir, image_path)
            except Exception as e:
                import traceback
                print(traceback.format_exc())

    def go_back(self):
        from .splash import MainWindow
        self.main_window = MainWindow(self.model, self.processor)
        self.main_window.show()
        self.close()

    def select_folder(self):
        options = QtWidgets.QFileDialog.Options()
        dialog = QtWidgets.QFileDialog(self, "Select Image Folder", CUMULATIVE_DIR, options=options)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            folder = dialog.selectedFiles()[0]
            if folder:
                # Load images from the selected folder.
                self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if self.images:
                    self.images.sort()
                    self.current_image_index = 0
                    # Display the first image.
                    self.display_image(self.images[self.current_image_index])
                    self._update_image_indicator()

                    # Check if a prompt is already entered
                    if self.prompt_mode == "text" and self.prompt_entry.text().strip():
                        self.display_predictions()
                else:
                    # Notify the user if the folder is empty.
                    message_box = QtWidgets.QMessageBox()
                    message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
                    message_box.setText("The selected folder does not contain any images.")
                    message_box.exec_()

    def select_output_folder(self):
        options = QtWidgets.QFileDialog.Options()
        dialog = QtWidgets.QFileDialog(self, "Select Output Folder", options=options)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.output_folder = dialog.selectedFiles()[0]
            if self.output_folder:
                self._refresh_auto_annotate_enabled()
                message_box = QtWidgets.QMessageBox()
                message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
                message_box.setText(f"Output folder selected: {self.output_folder}")
                message_box.exec_()

    def display_image(self, image_path):
        img = self._imread_cached(image_path)
        if img is None:
            return
        self.baked_bbox_cv2 = None
        self.baked_seg_cv2  = None
        self.bbox_anns      = []
        self.seg_anns       = []
        self.current_mode   = None
        self.live_boxes     = []
        self.live_box_sources = []
        self.live_polys_cache = None
        # Per-image reject list; deletions on the previous image must not
        # suppress detections on this one.
        self._rejected_boxes = []
        self.base_cv2_image = img.copy()
        self.image_label.clear_all()
        # A new image always starts at Normal Tint; Darken Tint is a
        # per-view aid, not a sticky preference.
        if hasattr(self, "normal_tint_act"):
            self.normal_tint_act.setChecked(True)
        self.image_label.set_dark_tint(False)
        self.image_label.set_clean_image(img)
        self._refresh_auto_annotate_enabled()

    def _update_image_indicator(self):
        """Refresh the 'Image X of N' label above the canvas."""
        if not hasattr(self, 'image_index_label'):
            return
        if not self.images:
            self.image_index_label.setText("No image folder selected")
            return
        n = len(self.images)
        i = self.current_image_index + 1
        name = os.path.basename(self.images[self.current_image_index])
        self.image_index_label.setText(f"Image {i} of {n} -> {name}")

    def _carry_active(self):
        """True when the Carry Prompts Forward toggle is on."""
        return (hasattr(self, "carry_forward_checkbox")
                and self.carry_forward_checkbox.isChecked())

    def _carry_anchor_boxes_img(self):
        """The frozen carry anchor converted to CURRENT image pixel xyxy, or []."""
        anchor = getattr(self, "_carry_anchor", None) or []
        ow = self.image_label._orig_w or 0
        oh = self.image_label._orig_h or 0
        if not anchor or not ow or not oh:
            return []
        return [[(cx - w / 2) * ow, (cy - h / 2) * oh,
                 (cx + w / 2) * ow, (cy + h / 2) * oh]
                for cx, cy, w, h in anchor]

    def _refresh_and_get_carry_anchor(self):
        """Frozen carry-forward exemplar set as normalized [cx, cy, w, h].

        Refreshes the anchor ONLY from the current image's manual *rect*
        draws (what mouseReleaseEvent creates). Manual polygons are NOT
        considered: in seg mode the carried boxes get re-segmented into
        manual polys, and feeding their bounding boxes back would drift the
        anchor every image (quality degradation). Rect-only + the frozen
        self._carry_anchor fallback keeps it stable yet still re-anchors on a
        genuine new user draw, and survives a Regenerate that re-tags the
        boxes as detector output."""
        ow = self.image_label._orig_w
        oh = self.image_label._orig_h
        manual = []
        if ow and oh:
            for x1, y1, x2, y2 in self.image_label.get_prompt_boxes_in_image_coords():
                manual.append([
                    ((x1 + x2) / 2) / ow,
                    ((y1 + y2) / 2) / oh,
                    abs(x2 - x1) / ow,
                    abs(y2 - y1) / oh,
                ])
        if manual:
            self._carry_anchor = manual
            return list(manual)
        return list(getattr(self, "_carry_anchor", []) or [])

    def _finish_folder(self):
        """Called after the last image is passed: tell the user the folder is
        done and deselect the input + output folders so a stale selection
        can't bleed into the next session."""
        msg = QtWidgets.QMessageBox()
        msg.setStyleSheet("QLabel { color: black; font-size: 22px; } QMessageBox { background-color: white; }")
        msg.setWindowTitle("Folder complete")
        msg.setText("You have gone through all of the images in the folder.")
        msg.exec_()
        # Deselect input + output folders and reset the canvas/indicator.
        self.images = []
        self.current_image_index = 0
        self.output_folder = None
        self._carry_prompt_img = []
        self._carry_ref_bundle = None
        self._carry_anchor = []
        self.image_label.clear_all()
        self.image_label.set_clean_image(np.zeros((10, 10, 3), dtype=np.uint8))
        self._update_image_indicator()
        self._refresh_auto_annotate_enabled()

    def next_image(self):
        if getattr(self, "_busy", False):
            return
        if not self.images:
            message_box = QtWidgets.QMessageBox()
            message_box.setStyleSheet("QLabel { color: black; font-size: 24px; } QMessageBox { background-color: white; }")
            message_box.setText("No images loaded.")
            message_box.exec_()
            return
        # Advancing would drop an in-progress semi-auto mask, so confirm first.
        if not self._confirm_leave_unfinished("to the next image"):
            return
        # Auto-save the current image first; Next Image does the equivalent
        # of Save & Confirm (minus the popup) so manual edits and newly drawn
        # boxes are never dropped just because Save & Confirm wasn't pressed.
        self._persist_annotations(silent=True)
        # Last image: no wraparound; the folder is done.
        if self.current_image_index >= len(self.images) - 1:
            self._finish_folder()
            return
        # Carry the current image's drawn boxes forward as prompts for the
        # next image (same opt-in checkbox as Auto Annotate Remaining).
        # Captured BEFORE display_image clears the canvas; stored as
        # normalized fractions so a differently-sized next image keeps the
        # same relative box positions. Box-carry only for detectors that
        # RE-DETECT from example boxes (YOLOE-vis / YOLOE-seg). DINO/SAM3 ignore it.
        use_carry = ((not hasattr(self, 'carry_forward_checkbox')
                      or self.carry_forward_checkbox.isChecked())
                     and self._detector_uses_box_exemplars())
        carried_norm = self._refresh_and_get_carry_anchor() if use_carry else []

        # Appearance bundle for the next image's detector run: the SAME crop
        # one-shot that Auto Annotate Remaining uses, captured from THIS image
        # before we advance/clear the canvas. Gated on the carry checkbox AND a
        # box-exemplar detector (same gate as use_carry above): text-only
        # detectors (DINO) can not consume a visual ref bundle, so building one
        # only produced a per-image "no visual-prompt path" console line.
        # Consumed once by the next display_predictions so SAM3 carries by
        # APPEARANCE (crop-composite) and YOLOE by refer_image. None -> as before.
        self._carry_ref_bundle = (
            self._collect_box_prompt_crops()
            if (hasattr(self, "carry_forward_checkbox")
                and self.carry_forward_checkbox.isChecked()
                and self._detector_uses_box_exemplars())
            else None
        )

        # Advance to the next image (no modulo; wraparound removed).
        self.current_image_index = self.current_image_index + 1
        self.display_image(self.images[self.current_image_index])
        self._update_image_indicator()

        # Carried boxes are the detector's VISUAL EXEMPLAR only - never
        # installed as annotations, never saved/segmented at fixed coords.
        # Converted to the new image's pixel space and consumed once by the
        # next display_predictions; only the detector's fresh per-image
        # output is saved.
        self._carry_prompt_img = []
        if carried_norm:
            ow = self.image_label._orig_w or 1
            oh = self.image_label._orig_h or 1
            self._carry_prompt_img = [
                [(cx - w / 2) * ow, (cy - h / 2) * oh,
                 (cx + w / 2) * ow, (cy + h / 2) * oh]
                for cx, cy, w, h in carried_norm
            ]

        # Run predictions for the new image.
        self.display_predictions()

    def _collect_box_prompt_crops(self):
        """Bundle the CURRENT image's drawn boxes as the carry-forward prompt:
        the image path, the boxes' xyxy coords in that image, and the cropped
        pixels. The drawn boxes become the appearance prompt for every remaining
        image. Falls back to the frozen carry anchor when a prior regenerate
        consumed the on-screen boxes. Returns None when there are no boxes."""
        if not getattr(self, "images", None):
            return None
        boxes = (self.image_label.get_prompt_boxes_in_image_coords()
                 or self._carry_anchor_boxes_img())
        if not boxes:
            return None
        img_path = self.images[self.current_image_index]
        img = self._imread_cached(img_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        crops, clean_boxes = [], []
        for x1, y1, x2, y2 in boxes:
            ix1 = max(0, int(round(min(x1, x2))))
            iy1 = max(0, int(round(min(y1, y2))))
            ix2 = min(w, int(round(max(x1, x2))))
            iy2 = min(h, int(round(max(y1, y2))))
            if ix2 - ix1 < 2 or iy2 - iy1 < 2:
                continue
            crops.append(img[iy1:iy2, ix1:ix2].copy())
            clean_boxes.append([float(ix1), float(iy1), float(ix2), float(iy2)])
        if not crops:
            return None
        print(f"[carry] using {len(crops)} box crop(s) from "
              f"{os.path.basename(img_path)} as the one-shot reference")
        return {"image_path": img_path, "boxes_xyxy": clean_boxes, "crops": crops}

    def _batch_chunk_size(self):
        """How many images to DETECT before switching to the SEGMENT pass in a
        two-stage (detect -> separate segmenter) batch run. Chunking keeps just
        one heavy model resident per pass, so under a memory budget YOLOE/DINO
        and SAM3 load once PER CHUNK instead of reloading every image. From
        AUTOANNOTATE_BATCH_CHUNK (default 8); <=0 / invalid -> 8."""
        try:
            n = int(os.environ.get("AUTOANNOTATE_BATCH_CHUNK", "8"))
            return n if n > 0 else 8
        except (TypeError, ValueError):
            return 8

    def _batch_start_index(self):
        """First image index Auto Annotate Remaining processes. With "Use First
        Image as Prompt" (carry) ON, the current image is the prompt source and
        the user wants it annotated too, so start AT it -> N output files, not
        N-1. Otherwise start at the NEXT image (the current one is annotated via
        the interactive Auto Annotate button first; do not clobber its edits)."""
        carry_on = (hasattr(self, "carry_forward_checkbox")
                    and self.carry_forward_checkbox.isChecked())
        return self.current_image_index if carry_on else self.current_image_index + 1

    def _clear_segment_file(self, image_path, seg_dir):
        """Overwrite this image's segments label with an EMPTY file. Called in a
        produce-masks run when an image yields no mask, so boxes/ and segments/
        stay in lockstep and a stale mask from a prior interactive save cannot
        linger and make that image look inconsistent. Matches save_masks naming
        (basename stem) so it targets the same file save_masks would write."""
        stem = os.path.splitext(os.path.basename(image_path))[0]
        os.makedirs(seg_dir, exist_ok=True)
        open(os.path.join(seg_dir, f"{stem}.txt"), "w").close()

    def _retry_factor(self):
        """Multiplier for the ONE lower-threshold retry on an empty detection in
        Auto Annotate Remaining (default 0.5 = half). From AUTOANNOTATE_RETRY_FACTOR;
        clamped to (0,1) so the retry is always LOWER and bounded -- a single
        retry, never a loop."""
        try:
            f = float(os.environ.get("AUTOANNOTATE_RETRY_FACTOR", "0.5"))
        except (TypeError, ValueError):
            f = 0.5
        return f if 0.0 < f < 1.0 else 0.5

    def _detect_with_retry(self, image_path, prompt, det_thresh, mask_thresh, carried, ref):
        """Run the detector; if it returns ZERO boxes, retry ONCE at a lower
        threshold (det+mask x _retry_factor, floored at 0.02). Returns
        (det_boxes, yoloe_seg_results, used_det, retried). Bounded: the detector
        runs at most twice per image, so it can never spin into a loop."""
        det_boxes, yres = self._run_detector(
            image_path, prompt, det_thresh, mask_thresh, carried, ref=ref)
        if det_boxes:
            return det_boxes, yres, det_thresh, False
        factor = self._retry_factor()
        retry_det  = round(max(0.02, det_thresh * factor), 4)
        retry_mask = round(max(0.02, mask_thresh * factor), 4)
        if retry_det >= det_thresh:   # already at the floor -> retry can't help
            return det_boxes, yres, det_thresh, False
        det_boxes, yres = self._run_detector(
            image_path, prompt, retry_det, retry_mask, carried, ref=ref)
        return det_boxes, yres, retry_det, True

    def _reset_review_dir(self, output_folder, model_tag):
        """Delete a previous run's review folder for THIS model so the review
        set always reflects the LATEST run of this pipeline (no stale image
        copies / report accumulate), while leaving OTHER models' review folders
        intact. SAFETY: only ever removes <output>/_review/<model_tag> -- a dir
        whose parent is literally named "_review" sitting directly under the
        output folder; never the input folder, never output/boxes|segments, and
        never the whole _review tree."""
        if not output_folder or not model_tag:
            return
        model_dir = os.path.normpath(os.path.join(output_folder, "_review", model_tag))
        # Paranoia guards: exact model basename, parent named "_review", that
        # "_review" sits directly under the output folder, and it is a real dir.
        if os.path.basename(model_dir) != model_tag:
            return
        parent = os.path.dirname(model_dir)
        if os.path.basename(parent) != "_review":
            return
        if os.path.dirname(parent) != os.path.normpath(output_folder):
            return
        if not os.path.isdir(model_dir):
            return
        import shutil
        try:
            shutil.rmtree(model_dir)
        except Exception as _e:
            print(f"[review] could not clear old review folder {model_dir}: {_e}")
        # Drop the now-empty _review parent so a clean run leaves no clutter.
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass

    def _finalize_review(self, review, output_folder):
        """Write the review folder for images that produced nothing (after the
        retry) or failed. Layout (per-model subfolder so different pipelines'
        problem images stay separate, e.g. _review/SwinB_SAM2, _review/YOLOEseg):
            <output>/_review/<model>/boxes/      detection-stage problem images
            <output>/_review/<model>/segments/   segmentation-stage problem images
            <output>/_review/<model>/review_report.csv   one row per problem image
        Source images are COPIED (never moved), so the input folder is untouched.
        Returns the review dir (or None when there is nothing to review)."""
        model_tag = self._model_tag()
        # Wipe THIS model's previous review folder first so it never accumulates
        # stale copies; other models' review folders are left intact.
        self._reset_review_dir(output_folder, model_tag)
        if not review:
            return None
        import csv as _csv, shutil
        review_dir = os.path.join(output_folder, "_review", model_tag)
        boxes_rev  = os.path.join(review_dir, "boxes")
        seg_rev    = os.path.join(review_dir, "segments")
        os.makedirs(boxes_rev, exist_ok=True)
        os.makedirs(seg_rev, exist_ok=True)
        report = os.path.join(review_dir, "review_report.csv")
        try:
            with open(report, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["image", "stage", "status", "reason", "detector",
                            "prompt", "orig_threshold", "retry_threshold"])
                for r in review:
                    dest = seg_rev if r.get("stage") == "segments" else boxes_rev
                    try:
                        shutil.copy2(r["image"], os.path.join(dest, os.path.basename(r["image"])))
                    except Exception as _e:
                        print(f"[review] could not copy {r['image']}: {_e}")
                    w.writerow([os.path.basename(r["image"]), r.get("stage",""),
                                r.get("status",""), r.get("reason",""),
                                r.get("detector",""), r.get("prompt",""),
                                r.get("orig",""), r.get("retry","")])
            print(f"[review] {len(review)} image(s) need review -> {review_dir}")
        except Exception as _e:
            print(f"[review] failed to write review report: {_e}")
        return review_dir

    def auto_annotate_remaining(self):
        """Apply the current image's drawn boxes as carry-forward prompts to
        every remaining image. Uses the selected detector pipeline (YOLOE-vis,
        SAM2/SAM3 box-prompted, DINO text-prompted, etc.) and skips images
        whose YOLO label already exists in the output folder."""
        if getattr(self, "_busy", False):
            return
        if not self.images:
            return
        if not self.output_folder:
            self.select_output_folder()
            if not self.output_folder:
                return

        # Lock the three bottom-left buttons + set the busy flag so a
        # second click on Regenerate / Next / Auto Annotate is dropped
        # while this run is in progress.
        _aar_busy_btns = [b for b in (getattr(self, "auto_annotate_btn", None),
                                      getattr(self, "regen_btn", None),
                                      getattr(self, "next_btn", None)) if b is not None]
        for _b in _aar_busy_btns:
            _b.setEnabled(False)
        QtWidgets.QApplication.processEvents()
        self._busy = True

        boxes_dir = os.path.join(self.output_folder, 'boxes')
        seg_dir   = os.path.join(self.output_folder, 'segments')
        os.makedirs(boxes_dir, exist_ok=True)
        os.makedirs(seg_dir,   exist_ok=True)

        # Static prompt set: the user's drawn boxes from the CURRENT image
        # are reused on every remaining image. Predictable, no compounding
        # drift if one detection is off.
        use_carry   = ((not hasattr(self, 'carry_forward_checkbox')
                        or self.carry_forward_checkbox.isChecked())
                       and self._detector_uses_box_exemplars())
        # Same frozen anchor as Next Image (user-curated boxes only),
        # converted to absolute xyxy for this run's images.
        anchor_norm = self._refresh_and_get_carry_anchor() if use_carry else []
        ow = self.image_label._orig_w or 1
        oh = self.image_label._orig_h or 1
        carried     = [[(cx - w / 2) * ow, (cy - h / 2) * oh,
                        (cx + w / 2) * ow, (cy + h / 2) * oh]
                       for cx, cy, w, h in anchor_norm]
        prompt      = self.prompt_entry.text() if hasattr(self, 'prompt_entry') else ''
        det_thresh  = self.confidence_slider.value()  / 100
        mask_thresh = self.box_threshold_slider.value() / 100
        _, seg_key, is_standalone = self._detector_keys_for_pipeline()
        produce_masks = bool(is_standalone or seg_key)

        # Carry Prompts Forward: when ON, collect this image's Visual Reference
        # crops once so every remaining image gets the same one-shot visual
        # prompt. The typed text prompt already carries (reused per image below).
        # None -> no crops, Auto Annotate Remaining behaves exactly as before.
        # Gate on a box-exemplar detector too (matches use_carry above and the
        # next_image path): text-only detectors can not consume a visual ref
        # bundle, so building one only wasted crops + printed a per-image warning.
        ref_bundle = (self._collect_box_prompt_crops()
                      if (hasattr(self, "carry_forward_checkbox")
                          and self.carry_forward_checkbox.isChecked()
                          and self._detector_uses_box_exemplars())
                      else None)

        # "Use First Image as Prompt" (carry) ON => include the CURRENT
        # (prompt) image so the output folder gets N files, not N-1. Logic in
        # _batch_start_index (headless-tested, T46); model-agnostic: the
        # per-image loop below runs whatever pipeline is selected on it too.
        targets   = self.images[self._batch_start_index():]
        processed = 0
        failed    = []
        review    = []   # images that came back empty (after retry) or failed
        canceled  = False

        # Auto Annotate Remaining always overwrites: each per-image
        # save below opens its label/segment files with 'w', so a
        # re-run cleanly replaces whatever was there. No skip, no
        # prompt -- the user explicitly asked for this so re-tuning
        # parameters doesn't require deleting the output folder.

        # Modal progress dialog keeps the Qt event loop alive during the
        # (otherwise blocking) per-image loop, so the window repaints and
        # doesn't show "Application Not Responding" when tabbed away. It
        # also gives the user a Cancel button and a live image counter.
        progress = QtWidgets.QProgressDialog(
            "Auto-annotating remaining images...", "Cancel", 0, len(targets), self)
        progress.setWindowTitle("Auto Annotate Remaining")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setStyleSheet("QLabel { color: black; } QProgressDialog { background-color: white; }")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        # Two-stage runs (detector -> a SEPARATE segmenter, e.g. YOLOE/DINO ->
        # SAM3) detect and segment in separate CHUNKED passes so only one heavy
        # model is resident per pass. Interleaving detect+segment per image
        # would thrash a tight model budget (reload YOLOE and SAM3 every image);
        # chunking loads each heavy model once per chunk. Output is identical to
        # the per-image order. Standalone one-shot (masks from the detector) and
        # bbox-only runs keep the simple per-image loop below.
        two_stage = bool(produce_masks and (not is_standalone) and seg_key)
        _chunk = self._batch_chunk_size() if two_stage else len(targets)
        import time  # per-image timing in the batch logs (to spot slowdown creep)
        n_folder = len(self.images)   # total images in the loaded folder
        n_run    = len(targets)       # images this run will annotate (the remaining ones)

        if two_stage:
            done = 0
            for _base in range(0, len(targets), _chunk):
                if canceled:
                    break
                group = targets[_base:_base + _chunk]
                detected = {}
                # --- DETECT pass: detector resident (segmenter may be evicted) ---
                for _j, image_path in enumerate(group):
                    if progress.wasCanceled():
                        canceled = True
                        break
                    progress.setLabelText(
                        f"Detecting {done + len(detected) + 1} of {len(targets)}: "
                        f"{os.path.basename(image_path)}")
                    QtWidgets.QApplication.processEvents()
                    _t0 = time.perf_counter()
                    try:
                        det_boxes, _yoloe, used_det, retried = self._detect_with_retry(
                            image_path, prompt, det_thresh, mask_thresh, carried, ref_bundle)
                        absolute_boxes, _ = self._combine_with_dedup(list(det_boxes), [], 0.5)
                        save_boxes_yolo(absolute_boxes, image_path, boxes_dir)
                        detected[image_path] = absolute_boxes
                        _rt = f" (retried @ {used_det:.2f})" if retried else ""
                        print(f"[auto-annotate] {os.path.basename(image_path)} "
                              f"({_base + _j + 1}/{n_run} run, {n_folder} in folder): "
                              f"detect {time.perf_counter() - _t0:.1f}s -> "
                              f"{len(absolute_boxes)} boxes{_rt}")
                        if not absolute_boxes:
                            review.append({"image": image_path, "stage": "boxes",
                                "status": "empty",
                                "reason": f"0 detections after retry @ {used_det:.2f}",
                                "detector": self.detector_choice, "prompt": prompt,
                                "orig": f"{det_thresh:.2f}", "retry": f"{used_det:.2f}"})
                    except Exception as e:
                        import traceback
                        detected[image_path] = None
                        failed.append((image_path, str(e)))
                        review.append({"image": image_path, "stage": "boxes",
                            "status": "failed", "reason": str(e)[:200],
                            "detector": self.detector_choice, "prompt": prompt,
                            "orig": f"{det_thresh:.2f}", "retry": ""})
                        print(f"[auto-annotate] {image_path}: DETECT FAILED {e}")
                        traceback.print_exc()
                    self._release_inference_memory()
                if canceled:
                    break
                # --- SEGMENT pass: segmenter resident (detector may be evicted) ---
                for _j, image_path in enumerate(group):
                    if progress.wasCanceled():
                        canceled = True
                        break
                    absolute_boxes = detected.get(image_path)
                    if absolute_boxes is None:  # detect failed -> already logged
                        done += 1
                        progress.setValue(min(done, len(targets)))
                        continue
                    stem = os.path.splitext(os.path.basename(image_path))[0]
                    progress.setLabelText(
                        f"Segmenting {done + 1} of {len(targets)}: "
                        f"{os.path.basename(image_path)}")
                    QtWidgets.QApplication.processEvents()
                    _t0 = time.perf_counter()
                    try:
                        overlay_polys = None
                        if absolute_boxes:
                            sam_results = self._run_segmenter(image_path, absolute_boxes)
                            if sam_results is not None:
                                save_masks(sam_results, seg_dir, image_path)
                                overlay_polys = result_clean_polys(sam_results[0])
                        # No mask this run (0 detections, or segmenter empty)?
                        # Clear any stale segments file so boxes/ and segments/
                        # stay in lockstep and this image is not left looking
                        # inconsistent (esp. the carried first/prompt image).
                        if not overlay_polys:
                            self._clear_segment_file(image_path, seg_dir)
                        self._save_split_overlays(image_path, absolute_boxes, overlay_polys)
                        processed += 1
                        print(f"[auto-annotate] {stem} ({processed}/{n_run} done, "
                              f"{n_folder} in folder): segment "
                              f"{time.perf_counter() - _t0:.1f}s conf={det_thresh:.2f} "
                              f"mask={mask_thresh:.2f} -> {len(absolute_boxes)} boxes")
                        if absolute_boxes and not overlay_polys:
                            review.append({"image": image_path, "stage": "segments",
                                "status": "empty",
                                "reason": "detector found boxes but segmenter produced no masks",
                                "detector": self.detector_choice, "prompt": prompt,
                                "orig": f"{mask_thresh:.2f}", "retry": ""})
                    except Exception as e:
                        import traceback
                        failed.append((image_path, str(e)))
                        review.append({"image": image_path, "stage": "segments",
                            "status": "failed", "reason": str(e)[:200],
                            "detector": self.detector_choice, "prompt": prompt,
                            "orig": f"{mask_thresh:.2f}", "retry": ""})
                        print(f"[auto-annotate] {image_path}: SEGMENT FAILED {e}")
                        traceback.print_exc()
                    self._release_inference_memory()
                    done += 1
                    progress.setValue(min(done, len(targets)))
        else:
            for _idx, image_path in enumerate(targets):
                if progress.wasCanceled():
                    canceled = True
                    break
                stem = os.path.splitext(os.path.basename(image_path))[0]
                progress.setLabelText(f"Image {_idx + 1} of {len(targets)}: {os.path.basename(image_path)}")
                progress.setValue(_idx)
                # Pump the event loop so the dialog paints and stays responsive
                # while the (blocking) detector/segmenter call runs below.
                QtWidgets.QApplication.processEvents()
                _t0 = time.perf_counter()
                try:
                    det_boxes, yoloe_seg_results, used_det, retried = self._detect_with_retry(
                        image_path, prompt, det_thresh, mask_thresh, carried, ref_bundle)
                    absolute_boxes, _ = self._combine_with_dedup(
                        list(det_boxes), [], 0.5)
                    save_boxes_yolo(absolute_boxes, image_path, boxes_dir)
                    overlay_polys = None
                    if produce_masks and absolute_boxes:
                        if is_standalone:
                            # One-shot detectors (YOLOE-seg / SAM3) already filtered their
                            # masks by max_area + NMS into _oneshot_polys_aligned, index-
                            # aligned with absolute_boxes. Save THOSE, not the raw results,
                            # so segments/ matches boxes/ (no giant leaf masks or dropped-
                            # duplicate detections re-appearing only in the seg view).
                            overlay_polys = list(self._oneshot_polys_aligned or [])
                            save_polys_yolo(overlay_polys, seg_dir, image_path)
                        elif not is_standalone:
                            sam_results = self._run_segmenter(image_path, absolute_boxes)
                            if sam_results is not None:
                                save_masks(sam_results, seg_dir, image_path)
                                overlay_polys = result_clean_polys(sam_results[0])
                    # Mask run but no mask produced? Clear any stale segments
                    # file so boxes/ and segments/ stay in lockstep (bbox-only
                    # runs do not touch segments/, so gate on produce_masks).
                    if produce_masks and not overlay_polys:
                        self._clear_segment_file(image_path, seg_dir)
                    # Save reference images split by kind. The boxes view
                    # always renders (bbox-only runs still produce a useful
                    # reference); the masks view only renders when polys
                    # exist.
                    self._save_split_overlays(image_path, absolute_boxes, overlay_polys)
                    processed += 1
                    _rt = f" (retried @ {used_det:.2f})" if retried else ""
                    print(f"[auto-annotate] {stem} ({processed}/{n_run} done, "
                          f"{n_folder} in folder): {time.perf_counter() - _t0:.1f}s "
                          f"conf={det_thresh:.2f} mask={mask_thresh:.2f} -> "
                          f"{len(absolute_boxes)} boxes{_rt}")
                    if not absolute_boxes:
                        review.append({"image": image_path, "stage": "boxes",
                            "status": "empty",
                            "reason": f"0 detections after retry @ {used_det:.2f}",
                            "detector": self.detector_choice, "prompt": prompt,
                            "orig": f"{det_thresh:.2f}", "retry": f"{used_det:.2f}"})
                    elif produce_masks and not overlay_polys:
                        review.append({"image": image_path, "stage": "segments",
                            "status": "empty",
                            "reason": "detector found boxes but segmenter produced no masks",
                            "detector": self.detector_choice, "prompt": prompt,
                            "orig": f"{mask_thresh:.2f}", "retry": ""})
                except Exception as e:
                    import traceback
                    failed.append((image_path, str(e)))
                    review.append({"image": image_path, "stage": "boxes",
                        "status": "failed", "reason": str(e)[:200],
                        "detector": self.detector_choice, "prompt": prompt,
                        "orig": f"{det_thresh:.2f}", "retry": ""})
                    print(f"[auto-annotate] {image_path}: FAILED {e}")
                    traceback.print_exc()
                # Free per-image memory inside the batch loop too, same reason as
                # the interactive path: stop the allocator pool creeping into swap.
                self._release_inference_memory()

        progress.setValue(len(targets))
        progress.close()

        review_dir = self._finalize_review(review, self.output_folder)
        empties = [r for r in review if r.get("status") == "empty"]
        parts = [f"Processed: {processed}"]
        if canceled:
            parts.insert(0, "Run canceled before finishing.")
        if failed:
            parts.append(f"Failed: {len(failed)}")
            for path, err in failed[:5]:
                parts.append(f"  - {os.path.basename(path)}: {err[:60]}")
            if len(failed) > 5:
                parts.append(f"  ...and {len(failed) - 5} more")
        if empties:
            parts.append(f"Empty (needs review): {len(empties)}")
            for r in empties[:5]:
                parts.append(f"  - {os.path.basename(r['image'])}: {r['reason'][:50]}")
            if len(empties) > 5:
                parts.append(f"  ...and {len(empties) - 5} more")
        if review_dir:
            parts.append(f"Review folder: {review_dir}")
        msg = QtWidgets.QMessageBox()
        msg.setStyleSheet("QLabel { color: white; font-size: 18px; } QMessageBox { background-color: black; }")
        msg.setWindowTitle("Auto Annotate Remaining")
        msg.setText("\n".join(parts))
        msg.exec_()

        self._busy = False
        for _b in _aar_busy_btns:
            _b.setEnabled(True)
        self._refresh_auto_annotate_enabled()

        # A completed (not canceled) run has annotated every remaining image,
        # so the folder is done, the same closure as reaching the last image via
        # Next Image: the all-done alert plus deselecting the folders.
        if not canceled:
            self._finish_folder()

    def _update_detection_threshold_label(self, value):
        # Show the EFFECTIVE confidence the current detector actually uses:
        # YOLOE squashes the slider (x0.20); DINO/SAM3 use it raw. So the same
        # slider position means different strictness depending on the model.
        det_key, _, _ = self._detector_keys_for_pipeline()
        if det_key in ("yoloe_vis", "yoloe_seg"):
            eff = f"YOLOE {self._yoloe_effective_conf(value / 100):.2f}"
        elif det_key == "sam3_det":
            eff = f"SAM3 {value / 100:.2f}"
        elif det_key in ("dino_swint", "dino_swinb"):
            eff = f"DINO {value / 100:.2f}"
        else:
            eff = f"{value / 100:.2f}"
        self.detection_threshold_label.setText(f"Detector confidence: {value}  ({eff})")

    def _update_mask_threshold_label(self, value):
        self.mask_threshold_label.setText(f"Segmenter confidence: {value}")

    def _toggle_sd_panel(self, checked):
        """Show/hide the Stable Diffusion controls. State persists across
        images because the panel is built once and display_image never
        resets it."""
        self.sd_panel.setVisible(checked)
        self.sd_toggle_btn.setText(
            "Synthetic Images (Diffusion) \u25be" if checked
            else "Synthetic Images (Diffusion) \u25b8")

    def _open_sd_prompts(self):
        """Edit the SD prompt + negative prompt in a popup; store them as
        single-line strings (the generation read sites use them verbatim)."""
        dlg = SDPromptDialog(self, prompt=self._sd_prompt, negative=self._sd_neg)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._sd_prompt = " ".join(dlg.prompt().split())
            self._sd_neg    = " ".join(dlg.negative().split())

    def _update_sd_strength_label(self, value):
        self.sd_strength_label.setText(f"Diffusion Strength: {value/100:.2f}")

    # Legacy aliases, some methods still reference these.
    def update_confidence_value(self, value):
        self._update_detection_threshold_label(value)

    def update_box_threshold_value(self, value):
        self._update_mask_threshold_label(value)

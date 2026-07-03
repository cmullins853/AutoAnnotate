"""Small dialogs and widgets: info badge, SD prompts, semi-auto settings, variation viewers."""
import os

import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from .style import BTN_GAP, BTN_GREEN, BTN_RED, btn_qss

class InfoBadge(QtWidgets.QLabel):
    """Small 'i' badge that pops up a legend while the cursor is over it
    (on hover or click) and hides it the moment the cursor leaves the badge."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._info_text = ""
        self._popup = None

    def set_info_text(self, text):
        self._info_text = text

    def _ensure_popup(self):
        if self._popup is None:
            self._popup = QtWidgets.QLabel(self.window())
            self._popup.setWindowFlags(QtCore.Qt.ToolTip | QtCore.Qt.FramelessWindowHint)
            self._popup.setStyleSheet(
                "QLabel { background-color: #2b2b2b; color: white; "
                "border: 1px solid #777777; border-radius: 4px; padding: 8px; "
                "font-size: 12px; font-weight: normal; }")
        return self._popup

    def _show_popup(self):
        if not self._info_text:
            return
        pop = self._ensure_popup()
        pop.setText(self._info_text)
        pop.adjustSize()
        # Sit it just to the right of the badge so the cursor stays on the 'i';
        # moving off the badge fires leaveEvent and hides it.
        pop.move(self.mapToGlobal(QtCore.QPoint(self.width() + 6, 0)))
        pop.show()

    def _hide_popup(self):
        if self._popup is not None:
            self._popup.hide()

    def enterEvent(self, e):
        self._show_popup()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hide_popup()
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._show_popup()
        super().mousePressEvent(e)

    def hideEvent(self, e):
        self._hide_popup()
        super().hideEvent(e)


class SDPromptDialog(QtWidgets.QDialog):
    """Popup editor for the Stable Diffusion prompt + negative prompt. Each
    field can also load a .txt of tailored instructions, asking whether to
    append it to or replace the current text."""
    def __init__(self, parent, prompt="", negative=""):
        super().__init__(parent)
        self.setWindowTitle("Stable Diffusion Prompts")
        self.setStyleSheet(
            "QDialog { background-color: #3a3a3a; }"
            "QLabel { color: white; }"
            "QPlainTextEdit { background-color: #444; color: white; }")
        self.resize(560, 380)
        lay = QtWidgets.QVBoxLayout(self)

        lay.addWidget(QtWidgets.QLabel("SD prompt -> describes the regenerated background:"))
        self.prompt_edit = QtWidgets.QPlainTextEdit(prompt)
        lay.addWidget(self.prompt_edit, 1)
        p_btn = QtWidgets.QPushButton("Attach .txt\u2026")
        p_btn.clicked.connect(lambda: self._attach(self.prompt_edit))
        lay.addWidget(p_btn, alignment=QtCore.Qt.AlignLeft)

        lay.addWidget(QtWidgets.QLabel("Negative prompt -> what to keep out:"))
        self.neg_edit = QtWidgets.QPlainTextEdit(negative)
        lay.addWidget(self.neg_edit, 1)
        n_btn = QtWidgets.QPushButton("Attach .txt\u2026")
        n_btn.clicked.connect(lambda: self._attach(self.neg_edit))
        lay.addWidget(n_btn, alignment=QtCore.Qt.AlignLeft)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _attach(self, edit):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Attach instructions (.txt)", "", "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Could not read file", str(e))
            return
        if not text:
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Attach .txt")
        box.setText("Add this file to the current text, or replace it?")
        append_btn  = box.addButton("Append", QtWidgets.QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is append_btn:
            cur = edit.toPlainText().strip()
            edit.setPlainText((cur + ", " + text) if cur else text)
        elif clicked is replace_btn:
            edit.setPlainText(text)

    def prompt(self):
        return self.prompt_edit.toPlainText()

    def negative(self):
        return self.neg_edit.toPlainText()


class SemiAutoSettingsDialog(QtWidgets.QDialog):
    """Per-mask settings for a selected semi-automatic segment:
      - Edit target: SAM points (re-run) vs polygon vertices (manual).
      - Class ID for the saved label.
      - Polygon simplification (Apply Simplify drops near-collinear vertices).
    """
    def __init__(self, parent, target="points", cls=0, points_enabled=True):
        super().__init__(parent)
        self.setWindowTitle("Semi-Auto Segment Settings")
        lay = QtWidgets.QVBoxLayout(self)

        lay.addWidget(QtWidgets.QLabel("Edit target:"))
        self.points_radio   = QtWidgets.QRadioButton("SAM points (re-run SAM)")
        self.vertices_radio = QtWidgets.QRadioButton("Polygon vertices (manual)")
        if not points_enabled:
            # No live SAM model -> SAM re-run isn't possible; vertices only.
            self.points_radio.setEnabled(False)
            self.points_radio.setToolTip("Needs a SAM2/SAM3 segmenter active.")
            target = "vertices"
        (self.points_radio if target == "points" else self.vertices_radio).setChecked(True)
        lay.addWidget(self.points_radio)
        lay.addWidget(self.vertices_radio)

        cls_row = QtWidgets.QHBoxLayout()
        cls_row.addWidget(QtWidgets.QLabel("Class ID:"))
        self.cls_spin = QtWidgets.QSpinBox()
        self.cls_spin.setRange(0, 999)
        self.cls_spin.setValue(int(cls))
        cls_row.addWidget(self.cls_spin)
        cls_row.addStretch()
        lay.addLayout(cls_row)

        simp_row = QtWidgets.QHBoxLayout()
        simp_row.addWidget(QtWidgets.QLabel("Simplify (%):"))
        self.simplify_spin = QtWidgets.QDoubleSpinBox()
        self.simplify_spin.setRange(0.0, 10.0)
        self.simplify_spin.setSingleStep(0.1)
        self.simplify_spin.setValue(0.0)
        self.simplify_spin.setToolTip(
            "Distance tolerance as a percent of image size. 'Apply Simplify' drops "
            "polygon vertices that fall within this distance of the simplified outline.")
        simp_row.addWidget(self.simplify_spin)
        self.simplify_now = QtWidgets.QPushButton("Apply Simplify")
        simp_row.addWidget(self.simplify_now)
        lay.addLayout(simp_row)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def target(self):
        return "points" if self.points_radio.isChecked() else "vertices"

    def cls(self):
        return self.cls_spin.value()

    def simplify_eps(self):
        # percent-of-image -> normalized (0-1) distance tolerance
        return self.simplify_spin.value() / 100.0


class VariationPreviewDialog(QtWidgets.QDialog):
    """Side-by-side preview for the single-image variation flow.

    Left  pane: the original image.
    Right pane: the SD variation.
    Buttons: Regenerate (re-roll the SD call) | Cancel | Save.

    `regenerate_cb` is a parent-supplied callable that takes no args
    and returns a new PIL variation (or None on failure). Keeping the
    callable on the parent side avoids pulling pipe / box state into
    the dialog and keeps the dialog reusable for any generator."""

    def __init__(self, original_pil, variation_pil, parent=None, regenerate_cb=None):
        super().__init__(parent)
        self.setWindowTitle("Variation Preview")
        self.setStyleSheet("QDialog { background-color: #2b2b2b; } "
                           "QLabel { color: white; } "
                           "QPushButton { background-color: #555; color: white; border: none; "
                           "             border-radius: 6px; padding: 6px 16px; font-size: 14px; } "
                           "QPushButton:hover { background-color: #666; } "
                           "QPushButton:disabled { background-color: #3a3a3a; color: #888; }")
        self.resize(1200, 650)
        self.original    = original_pil
        self.variation   = variation_pil
        self.regenerate_cb = regenerate_cb
        self.accepted_save = False  # set True iff Save was clicked

        layout = QtWidgets.QVBoxLayout(self)

        img_row = QtWidgets.QHBoxLayout()
        self.left_label  = QtWidgets.QLabel()
        self.right_label = QtWidgets.QLabel()
        for lbl, caption in ((self.left_label, "Original"),
                             (self.right_label, "Variation")):
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setMinimumSize(500, 500)
            lbl.setStyleSheet("border: 1px solid #555;")
        cap_left  = QtWidgets.QLabel("Original")
        cap_right = QtWidgets.QLabel("Variation")
        for c in (cap_left, cap_right):
            c.setAlignment(QtCore.Qt.AlignCenter)
            c.setStyleSheet("font-size: 14px; color: #ccc;")

        left_col  = QtWidgets.QVBoxLayout(); left_col.addWidget(cap_left);  left_col.addWidget(self.left_label, 1)
        right_col = QtWidgets.QVBoxLayout(); right_col.addWidget(cap_right); right_col.addWidget(self.right_label, 1)
        img_row.addLayout(left_col, 1)
        img_row.addLayout(right_col, 1)
        layout.addLayout(img_row, 1)

        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(BTN_GAP)
        self.regen_btn  = QtWidgets.QPushButton("Regenerate")
        cancel_btn      = QtWidgets.QPushButton("Cancel")
        save_btn        = QtWidgets.QPushButton("Save")
        save_btn.setStyleSheet(btn_qss(BTN_GREEN, 14))
        self.regen_btn.setToolTip("Re-roll the Stable Diffusion variation with the same settings.")
        cancel_btn.setToolTip("Discard this variation and close without saving.")
        save_btn.setToolTip("Keep this variation and save it to the synthetic images folder.")
        self.regen_btn.clicked.connect(self._regen)
        cancel_btn.clicked.connect(self.reject)
        save_btn.clicked.connect(self._save)
        btn_row.addStretch()
        btn_row.addWidget(self.regen_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

        self._render()

    @staticmethod
    def _pil_to_pixmap(pil_img, target_size):
        arr = np.array(pil_img.convert("RGB"))
        h, w, _ = arr.shape
        qt = QtGui.QImage(arr.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt)
        return pix.scaled(target_size, QtCore.Qt.KeepAspectRatio,
                          QtCore.Qt.SmoothTransformation)

    def _render(self):
        # resizeEvent can fire before __init__ finishes wiring widgets
        # (Qt sometimes lays out the dialog during self.resize(...) /
        # setStyleSheet calls); guard against missing attrs so first
        # construction can't crash.
        if not hasattr(self, "left_label") or not hasattr(self, "right_label"):
            return
        if self.original is not None:
            self.left_label.setPixmap(self._pil_to_pixmap(self.original,  self.left_label.size()))
        if self.variation is not None:
            self.right_label.setPixmap(self._pil_to_pixmap(self.variation, self.right_label.size()))

    def resizeEvent(self, event):
        # Re-fit images when the dialog is resized.
        super().resizeEvent(event)
        self._render()

    def _save(self):
        self.accepted_save = True
        self.accept()

    def _regen(self):
        if self.regenerate_cb is None:
            return
        self.regen_btn.setEnabled(False)
        self.regen_btn.setText("Generating...")
        QtWidgets.QApplication.processEvents()
        try:
            new_var = self.regenerate_cb()
            if new_var is not None:
                self.variation = new_var
                self._render()
        except Exception as e:
            print(f"[VariationPreviewDialog] regenerate failed: {e}")
        finally:
            self.regen_btn.setEnabled(True)
            self.regen_btn.setText("Regenerate")


class BatchVariationViewer(QtWidgets.QDialog):
    """Modal flipper for the batch variation flow.

    Opens after `Variations for Folder` finishes its run. Lets the
    user step through every variation that landed on disk, and prune
    the bad ones one at a time. Both the .jpg and the corresponding
    .txt label file are deleted together so the synthetic dataset
    stays consistent.

    Paths passed in are absolute paths to the .jpg files inside
    `synthetic images/images/`. The label is inferred by replacing
    .../images/ with .../labels/ and the .jpg suffix with .txt."""

    def __init__(self, variation_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Variations")
        self.setStyleSheet("QDialog { background-color: #2b2b2b; } "
                           "QLabel { color: white; } "
                           "QPushButton { background-color: #555; color: white; border: none; "
                           "             border-radius: 6px; padding: 6px 16px; font-size: 14px; } "
                           "QPushButton:hover { background-color: #666; } "
                           "QPushButton:disabled { background-color: #3a3a3a; color: #888; }")
        self.resize(1024, 720)
        self.paths = list(variation_paths)
        self.idx   = 0

        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setMinimumSize(900, 560)
        self.image_label.setStyleSheet("border: 1px solid #555;")
        layout.addWidget(self.image_label, 1)

        self.counter_label = QtWidgets.QLabel()
        self.counter_label.setAlignment(QtCore.Qt.AlignCenter)
        self.counter_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.counter_label)

        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(BTN_GAP)
        self.prev_btn  = QtWidgets.QPushButton("◀ Prev")
        self.next_btn  = QtWidgets.QPushButton("Next ▶")
        self.del_btn   = QtWidgets.QPushButton("Delete this variation")
        self.del_btn.setStyleSheet(btn_qss(BTN_RED, 14))
        close_btn      = QtWidgets.QPushButton("Close")
        self.prev_btn.setToolTip("Show the previous generated variation.")
        self.next_btn.setToolTip("Show the next generated variation.")
        self.del_btn.setToolTip("Permanently delete this variation and its label file from disk.")
        close_btn.setToolTip("Close the viewer and keep the remaining variations.")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn.clicked.connect(self._next)
        self.del_btn.clicked.connect(self._delete)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.del_btn)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh()

    def _label_path_for(self, img_path):
        # synthetic images/images/variation_<stem>.jpg
        # -> synthetic images/labels/variation_<stem>.txt
        img_dir   = os.path.dirname(img_path)
        synth_dir = os.path.dirname(img_dir)
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(synth_dir, 'labels', f'{stem}.txt')

    def _refresh(self):
        # resizeEvent can fire before __init__ finishes wiring widgets;
        # bail out until the image label + counter exist.
        if not hasattr(self, "image_label") or not hasattr(self, "counter_label"):
            return
        n = len(self.paths)
        if n == 0:
            self.image_label.setText("No variations remaining.")
            self.image_label.setPixmap(QtGui.QPixmap())
            self.counter_label.setText("0 / 0")
            for b in (self.prev_btn, self.next_btn, self.del_btn):
                b.setEnabled(False)
            return
        for b in (self.prev_btn, self.next_btn, self.del_btn):
            b.setEnabled(True)
        if self.idx < 0: self.idx = 0
        if self.idx >= n: self.idx = n - 1
        path = self.paths[self.idx]
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self.image_label.setText(f"Could not load:\n{os.path.basename(path)}")
        else:
            scaled = pix.scaled(self.image_label.size(),
                                QtCore.Qt.KeepAspectRatio,
                                QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
        self.counter_label.setText(
            f"{self.idx + 1} / {n}  -  {os.path.basename(path)}"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _prev(self):
        if not self.paths: return
        self.idx = (self.idx - 1) % len(self.paths)
        self._refresh()

    def _next(self):
        if not self.paths: return
        self.idx = (self.idx + 1) % len(self.paths)
        self._refresh()

    def _delete(self):
        if not self.paths:
            return
        path = self.paths[self.idx]
        label_path = self._label_path_for(path)
        for p in (path, label_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                print(f"[BatchVariationViewer] could not delete {p}: {e}")
        self.paths.pop(self.idx)
        self._refresh()

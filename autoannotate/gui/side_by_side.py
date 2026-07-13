"""Side-by-side viewer pairing synthetic variations with their source images."""
import os
from pathlib import Path

from PyQt5 import QtWidgets, QtGui, QtCore

from .style import (BTN_BLUE, BTN_GREY, BTN_PURPLE, add_input_scheme_actions,
                    btn_qss, lock_during, tool_toggle_qss)
from .zoompan import ZoomPanImageView

class SideBySideWindow(QtWidgets.QWidget):
    """
    Compare synthetic images against ground-truth images side by side.

    Left column = synthetic images, right column = ground truth. Each side
    opens its own folder (png / jpg). Prev / Next steps through a combined
    list that pairs the two folders by filename so the same scene shows on
    both sides at once; when the names don't correspond it falls back to
    positional pairing, and an unmatched side shows a placeholder.
    """

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        # original-resolution pixmaps for the currently shown pair; kept so
        # we can rescale cleanly on window resize without quality loss.
        self._synth_pixmap = None
        self._gt_pixmap = None
        self.synth_images = []   # sorted list of synthetic image paths
        self.gt_images = []      # sorted list of ground-truth image paths
        self.pairs = []          # [(synth_path|None, gt_path|None), ...]
        self.current_index = 0
        # Logical sides ('synth' / 'gt') keep their own data + editable title.
        # _left / _right say which logical side is shown in each PHYSICAL slot;
        # the <-> button swaps them. Default mirrors the prior inverted view
        # (Ground Truth on the left). Content-swap (not widget reparenting) keeps
        # this bulletproof; the physical widgets never move.
        self.titles = {"synth": "Synthetic Images", "gt": "Ground Truth"}
        self._left, self._right = "gt", "synth"
        # Zoom / pan belong to the LOGICAL side, not the physical slot, so a
        # swap carries each side's view with its image. Physical widgets are
        # the live copy; these dicts hold the side that is off-slot.
        self.view_states = {"synth": None, "gt": None}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("View Images Side by Side")
        self.showFullScreen()
        self.setStyleSheet("background-color: #454545;")

        screen_h = QtWidgets.QApplication.primaryScreen().geometry().height()
        btn_h = max(40, screen_h // 16)
        font  = max(13, screen_h // 58)
        title_font = max(18, screen_h // 34)

        title_style      = (
            f"color: white; font-size: {title_font}px; font-weight: bold; "
            "padding: 6px;"
        )
        panel_style = (
            "border: 2px solid #5a3aa0; border-radius: 10px; "
            "background-color: #2b2b2b; color: #888; "
            f"font-size: {font}px;"
        )

        outer = QtWidgets.QVBoxLayout()
        outer.setSpacing(12)
        outer.setContentsMargins(20, 20, 20, 20)

        # Top bar: Back button
        back_btn = QtWidgets.QPushButton("Back")
        back_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        back_btn.setFixedSize(200, btn_h)
        back_btn.setToolTip("Return to the main menu.")
        back_btn.clicked.connect(self.go_back)
        outer.addWidget(back_btn, alignment=QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        # Title row: editable side titles with the <-> swap button between
        title_edit_style = (title_style + " background: transparent; border: none; "
                            "border-bottom: 1px solid #555;")
        title_row = QtWidgets.QHBoxLayout()
        title_row.setSpacing(16)

        self.left_title = QtWidgets.QLineEdit()
        self.left_title.setStyleSheet(title_edit_style)
        self.left_title.setAlignment(QtCore.Qt.AlignCenter)
        self.left_title.setToolTip("Click to edit this side's title. It follows the side when you swap.")
        # textEdited fires on USER input only (not on programmatic setText), so
        # the edited title is stored against whichever logical side is here now.
        self.left_title.textEdited.connect(lambda t: self.titles.__setitem__(self._left, t))

        self.swap_btn = QtWidgets.QPushButton("⟷")   # <-> (long left-right arrow)
        self.swap_btn.setStyleSheet(btn_qss(BTN_BLUE, title_font))
        self.swap_btn.setFixedSize(int(btn_h * 1.6), btn_h)
        self.swap_btn.setToolTip("Swap the two sides. Titles, folder buttons and images "
                                 "all switch places.")
        self.swap_btn.clicked.connect(self._swap_sides)

        self.right_title = QtWidgets.QLineEdit()
        self.right_title.setStyleSheet(title_edit_style)
        self.right_title.setAlignment(QtCore.Qt.AlignCenter)
        self.right_title.setToolTip("Click to edit this side's title. It follows the side when you swap.")
        self.right_title.textEdited.connect(lambda t: self.titles.__setitem__(self._right, t))

        title_row.addWidget(self.left_title, 1)
        title_row.addWidget(self.swap_btn, 0, QtCore.Qt.AlignVCenter)
        title_row.addWidget(self.right_title, 1)
        outer.addLayout(title_row)

        # Middle: two image columns (folder button + view + filename)
        # Physical slots that never move; _apply_sides / _render fill them with
        # whichever logical side (_left / _right) currently maps to each.
        cols = QtWidgets.QHBoxLayout()
        cols.setSpacing(40)

        def _make_col():
            col = QtWidgets.QVBoxLayout(); col.setSpacing(10)
            folder_btn = QtWidgets.QPushButton()
            folder_btn.setStyleSheet(btn_qss(BTN_PURPLE, font))
            folder_btn.setFixedHeight(btn_h)
            folder_btn.setToolTip("Choose the folder of images to show on this side.")
            col.addWidget(folder_btn)
            view = ZoomPanImageView()
            view.set_placeholder("No folder selected")
            view.setStyleSheet(panel_style)
            view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            view.setMinimumSize(200, 200)
            col.addWidget(view, 1)
            resize_btn = self._make_resize_button(view, font, btn_h)
            col.addWidget(resize_btn)
            name = QtWidgets.QLabel("")
            name.setStyleSheet(f"color: #ccc; font-size: {font}px;")
            name.setAlignment(QtCore.Qt.AlignCenter)
            col.addWidget(name)
            return col, folder_btn, view, resize_btn, name

        (left_col, self.left_folder_btn, self.left_view,
         self.left_resize_btn, self.left_name) = _make_col()
        (right_col, self.right_folder_btn, self.right_view,
         self.right_resize_btn, self.right_name) = _make_col()
        self.left_folder_btn.clicked.connect(lambda: self._select_folder_side(self._left))
        self.right_folder_btn.clicked.connect(lambda: self._select_folder_side(self._right))
        cols.addLayout(left_col, 1)
        cols.addLayout(right_col, 1)
        outer.addLayout(cols, 1)

        self._apply_sides()   # initial titles + folder-button labels

        # Bottom: Prev / Next + position indicator
        nav_row = QtWidgets.QHBoxLayout()
        nav_row.setSpacing(12)
        self.prev_btn = QtWidgets.QPushButton("◀ Prev")
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.prev_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        self.next_btn.setStyleSheet(btn_qss(BTN_GREY, font))
        self.prev_btn.setFixedHeight(btn_h)
        self.next_btn.setFixedHeight(btn_h)
        self.prev_btn.setMinimumWidth(220)
        self.next_btn.setMinimumWidth(220)
        self.prev_btn.setToolTip("Show the previous image pair (Left/Up arrow).")
        self.next_btn.setToolTip("Show the next image pair (Right/Down arrow).")
        self.prev_btn.clicked.connect(lambda: lock_during(self.prev_btn, self.show_prev))
        self.next_btn.clicked.connect(lambda: lock_during(self.next_btn, self.show_next))
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.position_lbl = QtWidgets.QLabel("")
        self.position_lbl.setStyleSheet(f"color: white; font-size: {font}px;")
        self.position_lbl.setAlignment(QtCore.Qt.AlignCenter)

        nav_row.addStretch()
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.position_lbl)
        nav_row.addWidget(self.next_btn)
        nav_row.addStretch()
        outer.addLayout(nav_row)

        self.setLayout(outer)

    def _make_resize_button(self, view, font, btn_h):
        """The per-pane Image Resize toggle + dropdown, matching the annotation
        window's: the toggle arms scroll/pinch zoom and drag pan, the dropdown
        holds Original Size. Each pane gets its own, so you can zoom into a
        synthetic image while its source stays fit to the panel.

        The actions are bound to `view`, the PHYSICAL widget. _swap_sides moves
        the view STATE between the two widgets rather than the widgets
        themselves, so these bindings stay valid forever."""
        btn = QtWidgets.QToolButton()
        btn.setText("Image Resize: OFF")
        btn.setStyleSheet(tool_toggle_qss(BTN_BLUE, font))
        btn.setFixedHeight(int(btn_h * 0.8))
        btn.setCheckable(True)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn.setToolTip(
            "Zoom (scroll / pinch) and pan (drag) this image. Your zoom stays "
            "after you turn this off, and follows the image when you swap sides; "
            "the dropdown's Original Size returns it to a fit.")

        def _toggled(on):
            btn.setText(f"Image Resize: {'ON' if on else 'OFF'}")
            view.set_resize_mode(on)

        btn.toggled.connect(_toggled)

        menu = QtWidgets.QMenu()
        menu.setToolTipsVisible(True)
        original_size_act = QtWidgets.QAction("Original Size", self)
        original_size_act.setToolTip("Reset this side's zoom/pan to the original (fit) size.")
        original_size_act.triggered.connect(view.reset_view)
        menu.addAction(original_size_act)
        # Trackpad/Mouse input scheme, shared with the annotation window.
        add_input_scheme_actions(menu, self)

        btn.setMenu(menu)
        btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        return btn

    def _sync_resize_controls(self):
        """Make each pane's toggle agree with the view state the pane is
        actually showing. Called after a swap, when the two views have
        exchanged states behind the buttons' backs."""
        for phys in ("left", "right"):
            view = getattr(self, f"{phys}_view")
            btn = getattr(self, f"{phys}_resize_btn")
            state = view.view_state()
            btn.blockSignals(True)
            btn.setChecked(state["resize_mode"])
            btn.setText(f"Image Resize: {'ON' if state['resize_mode'] else 'OFF'}")
            btn.blockSignals(False)

    # Folder loading
    def _pick_folder(self, caption):
        dialog = QtWidgets.QFileDialog(self, caption, "")
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet("QWidget { background-color: white; color: black; }")
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            files = dialog.selectedFiles()
            return files[0] if files else None
        return None

    @staticmethod
    def _list_images(folder):
        imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        imgs.sort()
        return imgs

    def select_synth_folder(self):
        folder = self._pick_folder("Select Synthetic Images Folder")
        if folder:
            self.synth_images = self._list_images(folder)
            self._build_pairs()
            self._reset_views()
            self._show_current()

    def select_gt_folder(self):
        folder = self._pick_folder("Select Ground Truth Folder")
        if folder:
            self.gt_images = self._list_images(folder)
            self._build_pairs()
            self._reset_views()
            self._show_current()

    def _reset_views(self):
        """Back to fit on both panes. A freshly opened folder has nothing to do
        with the zoom the user left on the last one."""
        for phys in ("left", "right"):
            getattr(self, f"{phys}_view").reset_view()
        self.view_states = {"synth": None, "gt": None}

    # Side mapping / swap (<->)
    def _select_folder_side(self, side):
        """Open the folder picker for whichever LOGICAL side is on this slot."""
        if side == "synth":
            self.select_synth_folder()
        else:
            self.select_gt_folder()

    def _apply_sides(self):
        """Push each logical side's title + folder-button label into its current
        physical slot. Pixmaps/filenames are refreshed by _render/_show_current."""
        if not hasattr(self, "left_title"):
            return
        folder_label = {"synth": "Open Synthetic Folder", "gt": "Open Ground Truth Folder"}
        for phys, logical in (("left", self._left), ("right", self._right)):
            title_w = getattr(self, f"{phys}_title")
            # block signals so the programmatic setText doesn't fire textEdited
            title_w.blockSignals(True)
            title_w.setText(self.titles[logical])
            title_w.blockSignals(False)
            getattr(self, f"{phys}_folder_btn").setText(folder_label[logical])

    def _swap_sides(self):
        """Swap which logical side is shown on the left vs right. Titles, folder
        buttons, images, filenames AND each side's zoom/pan/tint all move
        together. Pure state toggle + re-render; no widgets are reparented, so
        it can be hit repeatedly with or without folders loaded."""
        # Harvest the live view state off the physical widgets BEFORE the
        # mapping flips, so each logical side keeps the view it was showing.
        self.view_states[self._left] = self.left_view.view_state()
        self.view_states[self._right] = self.right_view.view_state()
        self._left, self._right = self._right, self._left
        self.left_view.apply_view_state(self.view_states[self._left])
        self.right_view.apply_view_state(self.view_states[self._right])
        self._sync_resize_controls()
        self._apply_sides()
        self._show_current()

    # Pairing logic
    def _build_pairs(self):
        """Pair synthetic and ground-truth images so the same scene lines up
        on both sides. Matches by filename stem (with a startswith fallback
        for variation suffixes like 'berry_01_var.png' -> 'berry_01.png'),
        and falls back to positional pairing when no names correspond."""
        synth, gt = self.synth_images, self.gt_images
        pairs = []
        if synth and gt:
            # One ground truth backs MANY synthetic images: the batch flow makes
            # several variations of a single original (berry_01_var1, _var2, ...)
            # and all of them belong beside berry_01. `matched_gt` therefore only
            # records what got used, it never makes a ground truth unavailable.
            # Consuming each ground truth once (the old behaviour) left every
            # variation after the first with a blank right-hand pane.
            matched_gt = set()

            def _best_gt(s_stem):
                """Most specific ground truth for this synth stem, or None.
                Exact stem wins; otherwise the LONGEST prefix match, so
                'berry_01_var1' prefers 'berry_01' over a shorter 'berry'."""
                exact = [g for g in gt if Path(g).stem == s_stem]
                if exact:
                    return exact[0]
                pref = [g for g in gt if s_stem.startswith(Path(g).stem)]
                if pref:
                    return max(pref, key=lambda g: len(Path(g).stem))
                rev = [g for g in gt if Path(g).stem.startswith(s_stem)]
                if rev:
                    return max(rev, key=lambda g: len(Path(g).stem))
                return None

            for s in synth:
                match = _best_gt(Path(s).stem)
                if match is not None:
                    matched_gt.add(match)
                pairs.append((s, match))
            matched_any = bool(matched_gt)
            for g in gt:
                if g not in matched_gt:
                    pairs.append((None, g))
            if not matched_any:
                # No filename correspondence -- pair by position instead.
                n = max(len(synth), len(gt))
                pairs = [
                    (synth[i] if i < len(synth) else None,
                     gt[i] if i < len(gt) else None)
                    for i in range(n)
                ]
        elif synth:
            pairs = [(s, None) for s in synth]
        elif gt:
            pairs = [(None, g) for g in gt]
        self.pairs = pairs
        self.current_index = 0

    # Navigation
    def show_prev(self):
        if self.pairs:
            self.current_index = (self.current_index - 1) % len(self.pairs)
            self._show_current()

    def show_next(self):
        if self.pairs:
            self.current_index = (self.current_index + 1) % len(self.pairs)
            self._show_current()

    def _load_pixmap(self, path):
        if not path:
            return None
        pm = QtGui.QPixmap(path)
        return pm if not pm.isNull() else None

    def _name_for(self, logical):
        """Filename label for a logical side at the current pair index."""
        if not self.pairs:
            return ""
        synth_path, gt_path = self.pairs[self.current_index]
        path = synth_path if logical == "synth" else gt_path
        return os.path.basename(path) if path else "(no match)"

    def _show_current(self):
        has_pairs = bool(self.pairs)
        self.prev_btn.setEnabled(has_pairs and len(self.pairs) > 1)
        self.next_btn.setEnabled(has_pairs and len(self.pairs) > 1)

        if not has_pairs:
            self._synth_pixmap = None
            self._gt_pixmap = None
            self.left_name.setText("")
            self.right_name.setText("")
            self.position_lbl.setText("")
            self._render()
            return

        synth_path, gt_path = self.pairs[self.current_index]
        self._synth_pixmap = self._load_pixmap(synth_path)
        self._gt_pixmap = self._load_pixmap(gt_path)
        # Filenames follow the logical side into whichever physical slot it's in.
        self.left_name.setText(self._name_for(self._left))
        self.right_name.setText(self._name_for(self._right))
        self.position_lbl.setText(f"{self.current_index + 1} / {len(self.pairs)}")
        self._render()

    def _render(self):
        """Hand each logical side's stored pixmap to its current physical view.
        The view scales it itself, through its own zoom/pan transform, so the
        _left / _right mapping only decides placement."""
        # showFullScreen() can fire a resize event before init_ui has built
        # the view widgets; bail out until they exist.
        if not hasattr(self, "left_view"):
            return
        pmaps = {"synth": self._synth_pixmap, "gt": self._gt_pixmap}
        for phys, logical in (("left", self._left), ("right", self._right)):
            view = getattr(self, f"{phys}_view")
            pixmap = pmaps[logical]
            if pixmap is None:
                view.set_pixmap(None)
                view.set_placeholder("No image" if self.pairs else "No folder selected")
            else:
                # keep_view: stepping through pairs at a zoom is the point of
                # zooming, so Prev/Next must not throw the zoom away. Original
                # Size (or a new folder) is how you get back to a fit.
                view.set_pixmap(pixmap, keep_view=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    def keyPressEvent(self, event):
        # Arrow keys also flip through pairs; Esc returns to the menu.
        if event.key() in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
            self.show_next()
        elif event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up):
            self.show_prev()
        elif event.key() == QtCore.Qt.Key_Escape:
            self.go_back()
        else:
            super().keyPressEvent(event)

    def go_back(self):
        from .splash import MainWindow
        self.main_window = MainWindow(self.model, self.processor)
        self.main_window.show()
        self.close()

"""AnnotationCanvas: the zoomable image widget where boxes and masks are drawn/edited."""
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

from . import session_state
from .spatial import SpatialGrid
from .style import class_color_qt

class AnnotationCanvas(QtWidgets.QWidget):
    """
    Widget for displaying images with interactive annotation overlay.
    Uses QWidget (not QLabel): all rendering in paintEvent, storing
    original-resolution pixmaps so size never grows on toggle.

    Edit-mode UX: click an annotation to select it (only the selected one
    shows the red X). Click the X or press Delete/Backspace to remove it.
    """
    _X_R = 12   # visual radius of delete circle
    _HIT = 16   # click hit radius

    # Signal emitted whenever the effective box set changes (manual draw added,
    # manual box removed, annotation deleted/restored). ManualWindow listens to
    # rebuild its `live_boxes` source-of-truth.
    boxes_changed = QtCore.pyqtSignal()

    # Semi-automatic SAM mask drawing. mask_point_added fires whenever the user
    # clicks a foreground/background point (so ManualWindow can re-run SAM live);
    # mask_commit_requested fires on Enter to finalize the in-progress mask.
    mask_point_added      = QtCore.pyqtSignal()
    mask_commit_requested = QtCore.pyqtSignal()
    # Semi-auto (Google-Draw style): emitted when the user closes the outline
    # (clicks the first point again / double-clicks / Enter). Only then does SAM
    # run + commit; points just accumulate before that.
    mask_close_requested  = QtCore.pyqtSignal()
    # Edit-Semi-Auto-Segments: apply the in-progress point edit to the selected
    # mask (Enter). Point re-runs reuse mask_point_added (same SAM pass).
    semiauto_apply_requested = QtCore.pyqtSignal()
    # Open the per-mask settings dialog (edit target / class id / simplify) for
    # the selected semi-auto mask ('S' key).
    semiauto_settings_requested = QtCore.pyqtSignal()
    # Delete the whole selected semi-auto mask (X badge / Delete key).
    semiauto_delete_requested = QtCore.pyqtSignal()
    # A mask was just selected for vertex editing, which lets the controller
    # auto-simplify a dense model contour into a workable handle count.
    mask_selected = QtCore.pyqtSignal()
    # User tried to remove a vertex from a 3-point mask (the polygon floor), so
    # the controller asks whether to delete the whole mask instead.
    semiauto_min_vertex_delete = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # stored pixmaps at original resolution
        self._clean_pixmap = None   # no overlays
        self._baked_pixmap = None   # with baked overlays
        self._orig_w       = None
        self._orig_h       = None
        # Zoom/pan view transform (Image Resize mode). _zoom==1.0 with zero
        # pan == the original fit-to-window behavior. Every overlay coordinate
        # conversion routes through _get_scale_offset(), so changing these
        # transforms the image AND all annotations/handles in lockstep.
        self._zoom         = 1.0
        self._pan_x        = 0.0
        self._pan_y        = 0.0
        self._resize_mode  = False
        # Right-button drag pan (Image Resize mode). Armed on press; only once
        # the cursor actually travels does it become a pan, so a right-CLICK
        # still performs its normal action on release.
        self._pan_drag_last  = None
        self._pan_drag_moved = False
        # Darken Tint: view-only dimming overlay (set via the Image Resize
        # dropdown). Pure paint-time effect, never baked or saved.
        self._dark_tint    = False
        # draw-mode: drawn boxes live in self.annotations as manual rect
        # entries (source='manual', type='rect'). A drawn box is BOTH a
        # prompt to box-prompted detectors AND a saved annotation. The
        # prompt_boxes / annotation_boxes attributes below are read-only
        # properties derived from self.annotations.
        self._drag_start = None
        self._drag_cur   = None
        self.draw_mode   = False
        # Draw-subject: 'prompt' => new drags become input-only yellow PROMPT
        # boxes (source='prompt', never saved); 'annotation' => saved manual
        # rects (source='manual'). Set by ManualWindow.set_draw_subject().
        self.draw_subject = 'annotation'
        # Class id stamped onto each new drawn rect so multi-class box prompts
        # (and manual boxes) remember which class they are. ManualWindow keeps
        # this in sync with its active-class picker via set_active_draw_cls().
        self.active_draw_cls = 0
        # Interactive SAM mask drawing (seg view + SAM model only). Points are
        # stored in ABSOLUTE image coords as [x, y, 1] (all foreground);
        # _mask_preview_poly is the live SAM mask (normalized 0-1 [[x,y],...])
        # painted until the user commits it. _mask_draw_kind selects behaviour:
        #   'autodraw' -> a single point (each click replaces it)
        #   'semiauto' -> connected points accumulate into one outline
        self.mask_draw_mode    = False
        self._mask_draw_kind   = "semiauto"
        self._mask_points      = []
        self._mask_preview_poly = None
        # Extra preview blobs (autodraw multi-object). The primary blob lives in
        # _mask_preview_poly; these are the additional SEPARATE pieces, drawn with
        # no connector and committed as their own masks.
        self._mask_preview_extra = []
        self._mask_cursor      = None   # widget pos for the semi-auto rubber-band
        # Edit-Semi-Auto-Segments mode: select a committed semi-auto mask and
        # re-edit its SAM prompt points (add / remove / drag) with a live SAM
        # re-run. _semiauto_sel_idx indexes the selected annotation; _mask_points
        # / _mask_preview_poly are reused for the in-progress edit; _semiauto_drag_pt
        # is the index of the point being dragged.
        self.semiauto_edit_mode = False
        self._semiauto_sel_idx  = None
        self._semiauto_drag_pt  = None
        # Edit target within the semi-auto edit mode: "points" re-edits the SAM
        # prompt points (live re-run); "vertices" drags the polygon outline
        # directly (manual, no SAM). _vertex_drag_idx is the dragged vertex;
        # _semiauto_orig_data snapshots the polygon so Esc can revert vertex edits.
        self._semiauto_edit_target = "points"
        self._vertex_drag_idx      = None
        self._semiauto_orig_data   = None
        # Widget-coord point on the polygon outline nearest the cursor while in
        # the vertex-edit target; painted as a "+" ghost to advertise add-on-click.
        self._vertex_ghost         = None
        # edit-mode
        self.edit_mode   = False
        # annotations: [{'type': 'poly'|'rect', 'data': ..., 'deleted': bool}]
        #   'poly' -> data = [[x,y],...] normalised 0-1
        #   'rect' -> data = [cx, cy, w, h] normalised 0-1
        self.annotations = []
        self.selected_index = None  # index into self.annotations, or None
        # Multi-selection: superset of `selected_index`. The single
        # selected_index drives the resize handles + X-badge UI; the
        # set drives bulk delete and the cyan border on every selected
        # ann. When empty, single-selection rules apply.
        self.selected_indices = set()
        # Marquee (rubber-band) drag-select state. Only active when
        # multi_select_mode is on (toggled via the right-side "Select
        # Multiple" button); the previous gesture-based marquee was
        # replaced with this explicit-mode workflow.
        self._marquee_start = None  # widget coords during initial drag
        self._marquee_cur   = None
        self.multi_select_mode    = False
        self._persistent_marquee  = None  # [x1, y1, x2, y2] in IMAGE coords; survives mouse release
        self._marquee_handle      = None  # 'tl' | 'tr' | 'bl' | 'br' while resizing the marquee
        self._marquee_phase       = None  # 'drawing' | 'resizing' | None
        # Active rect-handle resize state. Polygons don't get handles.
        self._resize_handle  = None  # 'tl'|'tr'|'bl'|'br' while dragging
        self._resize_ann_idx = None
        self.undo_stack  = []
        self.redo_stack  = []
        self.setMouseTracking(True)
        # StrongFocus so we receive Delete/Backspace key events.
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.setStyleSheet("border: 1px solid #666; background-color: #333;")

    # image setters
    def set_clean_image(self, cv2_bgr):
        rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self._orig_w, self._orig_h = w, h
        qt = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._clean_pixmap = QtGui.QPixmap.fromImage(qt)
        self.update()

    def set_baked_image(self, cv2_bgr):
        rgb = cv2.cvtColor(cv2_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qt = QtGui.QImage(rgb.tobytes(), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self._baked_pixmap = QtGui.QPixmap.fromImage(qt)
        self.update()

    def set_dark_tint(self, enabled):
        """Toggle the Roboflow-style dimming overlay. View-only: it lives in
        paintEvent, so saved/baked images are unaffected."""
        self._dark_tint = bool(enabled)
        self.update()

    def clear_all(self):
        self._baked_pixmap = None
        self.annotations   = []
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.selected_index = None
        self.selected_indices = set()
        self._marquee_start = None
        self._marquee_cur   = None
        self._persistent_marquee = None
        self._marquee_handle     = None
        self._marquee_phase      = None
        self.undo_stack    = []
        self.redo_stack    = []

    # public API
    def set_draw_mode(self, enabled):
        self.draw_mode = enabled
        self.setCursor(QtCore.Qt.CrossCursor if enabled else QtCore.Qt.ArrowCursor)

    # Semi-automatic SAM mask drawing
    def set_mask_draw_mode(self, enabled, kind="semiauto"):
        """Toggle point-prompted SAM mask drawing. `kind` is 'autodraw' (refine
        points accumulate -- a click outside the mask adds a positive point,
        inside adds a negative one) or 'semiauto' (connected outline points
        accumulate).
        Wipes any in-progress session so a fresh entry never inherits stale
        points/preview."""
        self.mask_draw_mode = enabled
        if enabled:
            self._mask_draw_kind = "autodraw" if kind == "autodraw" else "semiauto"
        self.clear_mask_session()
        self.setCursor(QtCore.Qt.CrossCursor if enabled else QtCore.Qt.ArrowCursor)
        if enabled:
            self.setFocus()
        self.update()

    def clear_mask_session(self):
        """Drop the current in-progress points and preview mask."""
        self._mask_points = []
        self._mask_preview_poly = None
        self._mask_preview_extra = []
        self._mask_cursor = None
        self.update()

    def _near_first_point(self, pos, tol=12):
        """True when pos is within tol px of the first outline point (the
        close-the-shape gesture)."""
        if not self._mask_points:
            return False
        scale, off_x, off_y = self._get_scale_offset()
        fx, fy, _ = self._mask_points[0]
        wx = fx * scale + off_x
        wy = fy * scale + off_y
        return (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2 <= tol * tol

    def get_mask_points_image_coords(self):
        """Current prompt points as [((x, y), label), ...] in image coords."""
        return [((x, y), lab) for x, y, lab in self._mask_points]

    def set_mask_preview(self, poly_norm):
        """Install the live SAM mask (normalized [[x,y],...]) to paint. Resets any
        extra (multi-object) preview blobs; call set_mask_preview_extra after."""
        self._mask_preview_poly = poly_norm
        self._mask_preview_extra = []
        self.update()

    def get_mask_preview(self):
        return self._mask_preview_poly

    def set_mask_preview_extra(self, polys):
        """Additional SEPARATE preview blobs (autodraw multi-object); each is a
        normalized [[x,y],...]. Drawn with no connector, committed as own masks."""
        self._mask_preview_extra = [p for p in (polys or []) if p and len(p) >= 3]
        self.update()

    def get_mask_preview_extra(self):
        return list(self._mask_preview_extra)

    # Edit Semi-Automatic Segments
    def set_semiauto_edit_mode(self, enabled):
        """Enter/leave the mode that re-edits committed semi-auto masks. Clears
        any selection + in-progress edit so a fresh entry is clean."""
        self.semiauto_edit_mode = enabled
        self.clear_semiauto_selection()
        self.setCursor(QtCore.Qt.ArrowCursor)
        if enabled:
            self.setFocus()
        self.update()

    def clear_semiauto_selection(self):
        self._semiauto_sel_idx = None
        self._semiauto_drag_pt = None
        self._vertex_drag_idx  = None
        self._semiauto_orig_data = None
        self._mask_points = []
        self._mask_preview_poly = None
        self._mask_preview_extra = []
        self._vertex_ghost = None
        self.update()

    def cancel_inprogress(self):
        """Discard any in-progress draw draft or mask edit. Reverts live vertex
        edits first so the committed mask is left exactly as it was."""
        if (self._semiauto_orig_data is not None
                and self._semiauto_sel_idx is not None
                and self._semiauto_sel_idx < len(self.annotations)):
            self.annotations[self._semiauto_sel_idx]['data'] = \
                [list(p) for p in self._semiauto_orig_data]
        self.clear_semiauto_selection()
        self.clear_mask_session()

    def get_semiauto_selected_index(self):
        return self._semiauto_sel_idx

    def _semiauto_selected_ann(self):
        """The selected semi-auto annotation, or None. Self-heals a stale index
        (e.g. annotations replaced underneath) by clearing the selection."""
        i = self._semiauto_sel_idx
        if i is None or i < 0 or i >= len(self.annotations):
            if i is not None:
                self._semiauto_sel_idx = None
                self._semiauto_drag_pt = None
                self._vertex_drag_idx  = None
            return None
        return self.annotations[i]

    def has_unfinished_semiauto(self):
        """True when the draw flow has uncommitted SAM points (a mask the user
        started but hasn't committed with Enter)."""
        return bool(self.mask_draw_mode and self._mask_points)

    def _invalidate_semiauto_selection(self):
        """Drop the edit selection (indices) because the annotation list is
        being replaced. Keeps the draw-flow point buffer intact unless we're in
        edit mode, where those points belonged to the now-gone selection."""
        self._semiauto_sel_idx = None
        self._semiauto_drag_pt = None
        self._vertex_drag_idx  = None
        self._semiauto_orig_data = None
        if self.semiauto_edit_mode:
            self._mask_points = []
            self._mask_preview_poly = None
            self._mask_preview_extra = []

    def set_semiauto_edit_target(self, target):
        """Switch the selected mask between 'points' (SAM) and 'vertices'
        (polygon outline) editing."""
        self._semiauto_edit_target = "vertices" if target == "vertices" else "points"
        self._semiauto_drag_pt = None
        self._vertex_drag_idx  = None
        self._vertex_ghost     = None
        self.update()

    def get_semiauto_edit_target(self):
        return self._semiauto_edit_target

    def has_semiauto_masks(self):
        return any(not a.get('deleted') and a.get('semiauto')
                   and a.get('type') == 'poly' for a in self.annotations)

    def has_editable_masks(self):
        """True if any non-deleted polygon exists, model-generated masks
        included. Gates the Edit Masks tool; vertex editing needs no SAM model,
        so this is independent of segmenter availability."""
        return any(not a.get('deleted') and a.get('type') == 'poly'
                   for a in self.annotations)

    def _select_semiauto(self, idx):
        """Select a committed semi-auto mask and load its stored SAM prompt
        points (image coords) into the live edit buffer. Snapshots the polygon
        so vertex edits can be reverted with Esc."""
        self._semiauto_sel_idx = idx
        ann = self.annotations[idx]
        self._semiauto_orig_data = [list(p) for p in ann['data']]
        self._mask_points = []
        for sp in ann.get('sam_points', []):
            x, y, lab = sp
            self._mask_points.append([x * self._orig_w, y * self._orig_h, lab])
        # Seed the preview with the mask's current polygon so it shows even
        # before the first SAM re-run.
        self._mask_preview_poly = [list(p) for p in ann['data']]
        self.update()

    def _hit_vertex(self, pos, tol=8):
        """Index of the selected mask's polygon vertex within tol px of pos."""
        ann = self._semiauto_selected_ann()
        if ann is None:
            return None
        pts = self._to_label(ann['data'])
        for i, (wx, wy) in enumerate(pts):
            if (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2 <= tol * tol:
                return i
        return None

    def _insert_vertex(self, pos):
        """Insert a new polygon vertex (at pos, normalized) into the selected
        mask's outline, between the endpoints of the nearest edge."""
        ann = self._semiauto_selected_ann()
        if ann is None:
            return
        data = ann['data']
        p = self._widget_point_to_image(pos)
        if p is None or len(data) < 2:
            return
        nx, ny = p[0] / self._orig_w, p[1] / self._orig_h

        def _seg_d2(px, py, ax, ay, bx, by):
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                return (px - ax) ** 2 + (py - ay) ** 2
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
            cx, cy = ax + t * dx, ay + t * dy
            return (px - cx) ** 2 + (py - cy) ** 2

        best_i, best_d = 0, None
        n = len(data)
        for i in range(n):
            ax, ay = data[i]
            bx, by = data[(i + 1) % n]
            d = _seg_d2(nx, ny, ax, ay, bx, by)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        data.insert(best_i + 1, [nx, ny])
        self._mask_preview_poly = [list(pp) for pp in data]
        self.update()

    def _nearest_outline_point(self, pos):
        """(d2, wx, wy) of the point on the selected mask's outline nearest the
        widget-coord pos, projecting onto each edge. None if no mask/too few pts."""
        ann = self._semiauto_selected_ann()
        if ann is None:
            return None
        pts = self._to_label(ann['data'])
        n = len(pts)
        if n < 2:
            return None
        px, py = pos.x(), pos.y()
        best = None
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                cx, cy = ax, ay
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
                cx, cy = ax + t * dx, ay + t * dy
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if best is None or d2 < best[0]:
                best = (d2, cx, cy)
        return best

    def _point_in_preview(self, pos):
        """True when widget-coord pos falls inside ANY current SAM preview blob
        (primary or extra). Decides positive (outside) vs negative (inside) clicks."""
        previews = []
        if self._mask_preview_poly:
            previews.append(self._mask_preview_poly)
        previews.extend(self._mask_preview_extra)
        for poly in previews:
            if not poly or len(poly) < 3:
                continue
            wpts = self._to_label(poly)
            if len(wpts) < 3:
                continue
            qpoly = QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in wpts])
            if qpoly.containsPoint(pos, QtCore.Qt.OddEvenFill):
                return True
        return False

    def _hit_semiauto_poly(self, pos):
        """Index of the topmost non-deleted semi-auto poly whose widget polygon
        contains pos, else None. A SpatialGrid prunes to the masks whose bbox
        covers the click, so the (relatively expensive) point-in-polygon test
        only runs on the few candidates under the cursor, not every mask."""
        if not self._orig_w:
            return None
        p = self._widget_point_to_image(pos)
        if p is None:
            return None
        nx, ny = p[0] / self._orig_w, p[1] / self._orig_h
        grid = SpatialGrid()
        for i, ann in enumerate(self.annotations):
            # Any non-deleted polygon is editable now, model-generated masks
            # (source='detector', no 'semiauto' flag) included, not just the
            # hand-drawn semi-auto ones.
            if ann.get('deleted') or ann.get('type') != 'poly':
                continue
            grid.insert(self._ann_bbox_norm_xyxy(ann), i)
        # Topmost (highest index = drawn last = on top) wins.
        for i in sorted(grid.query_point(nx, ny), reverse=True):
            pts = self._to_label(self.annotations[i]['data'])
            if len(pts) >= 3:
                poly = QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in pts])
                if poly.containsPoint(pos, QtCore.Qt.OddEvenFill):
                    return i
        return None

    def _hit_mask_point(self, pos, tol=8):
        """Index into _mask_points whose widget position is within tol px of
        pos, else None."""
        scale, off_x, off_y = self._get_scale_offset()
        for i, (ix, iy, _lab) in enumerate(self._mask_points):
            wx = ix * scale + off_x
            wy = iy * scale + off_y
            if (pos.x() - wx) ** 2 + (pos.y() - wy) ** 2 <= tol * tol:
                return i
        return None

    def _widget_point_to_image(self, pos):
        """Map a widget-pixel QPoint -> absolute image-pixel (x, y), or None
        if no image is loaded or the click falls outside the image area."""
        if not self._orig_w:
            return None
        scale, off_x, off_y = self._get_scale_offset()
        if scale <= 0:
            return None
        ix = (pos.x() - off_x) / scale
        iy = (pos.y() - off_y) / scale
        if ix < 0 or iy < 0 or ix > self._orig_w or iy > self._orig_h:
            return None
        return (float(ix), float(iy))

    def set_draw_subject(self, subject):
        """Choose what a new drag creates: 'prompt' (input-only class-colored
        prompt box, source='prompt'), 'neg_prompt' (input-only RED negative
        prompt box, source='neg_prompt') or 'annotation' (saved manual rect,
        source='manual'). All input-only boxes are never saved. Driven by
        ManualWindow._refresh_draw_subject()."""
        self.draw_subject = subject if subject in ('prompt', 'neg_prompt') else 'annotation'

    def reclassify_user_rects(self, to_prompt):
        """Re-tag boxes the user already drew so they match the bucket new draws
        use after a prompt-mode / detector switch -- no redraw needed. Flips every
        non-deleted USER rect between 'manual' (saved, green) and 'prompt' (input-
        only detector prompt, yellow). Never touches detector output or polygons.
        Returns True if anything changed."""
        want  = 'prompt' if to_prompt else 'manual'
        other = 'manual' if to_prompt else 'prompt'
        changed = False
        for ann in self.annotations:
            if ann.get('deleted') or ann.get('type') != 'rect':
                continue
            if ann.get('source') == other:
                ann['source'] = want
                changed = True
        if changed:
            self.update()
            self.boxes_changed.emit()
        return changed

    def set_edit_mode(self, enabled):
        self.edit_mode = enabled
        self.selected_index = None
        self.selected_indices = set()
        self._marquee_start = None
        self._marquee_cur   = None
        if not enabled:
            # Multi-select requires edit mode (it operates on the
            # selectable annotation set), so it has to go down too.
            self.multi_select_mode    = False
            self._persistent_marquee  = None
            self._marquee_handle      = None
            self._marquee_phase       = None
        # Keep the cross-cursor visible when draw_mode is also on so the
        # user sees that drawing is still active alongside editing.
        self.setCursor(QtCore.Qt.CrossCursor if self.draw_mode else QtCore.Qt.ArrowCursor)
        if enabled:
            self.setFocus()
        self.update()

    def set_multi_select_mode(self, enabled):
        """Toggle the Select-Multiple workflow. Wiped on every toggle so
        a fresh entry into the mode does not inherit a stale marquee or
        a selection set from the previous round."""
        self.multi_select_mode    = enabled
        self._persistent_marquee  = None
        self._marquee_handle      = None
        self._marquee_phase       = None
        self._marquee_start       = None
        self._marquee_cur         = None
        self.selected_index       = None
        self.selected_indices     = set()
        self.update()

    def _clear_manual_rect_anns(self, silent=False):
        """Hard-remove non-deleted manual rect entries from self.annotations.
        When silent=True, skip the update()/emit() side effects (used by the
        property setters during mid-transition writes inside mode switches).
        Returns True if anything was removed."""
        before = len(self.annotations)
        self.annotations = [a for a in self.annotations
                            if not (a.get('source') == 'manual'
                                    and a['type'] == 'rect'
                                    and not a['deleted'])]
        if len(self.annotations) == before:
            return False
        self.selected_index = None
        self.selected_indices = set()
        self.undo_stack = []
        self.redo_stack = []
        if not silent:
            self.update()
            self.boxes_changed.emit()
        return True

    def clear_boxes(self):
        """Clear all drawn boxes (manual rect entries in self.annotations)."""
        self._clear_manual_rect_anns()

    def clear_prompt_boxes(self):
        # Compat alias, drawn boxes are unified now.
        self._clear_manual_rect_anns()

    def clear_annotation_boxes(self):
        # Compat alias, drawn boxes are unified now.
        self._clear_manual_rect_anns()

    @staticmethod
    def _ann_bbox_norm_xyxy(ann):
        """Normalized (x1, y1, x2, y2) bounding box of a rect or poly ann."""
        if ann['type'] == 'rect':
            cx, cy, w, h = ann['data']
            return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        xs = [p[0] for p in ann['data']]; ys = [p[1] for p in ann['data']]
        return (min(xs), min(ys), max(xs), max(ys))

    @staticmethod
    def _bbox_iou(a, b):
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        u = ua + ub - inter
        return inter / u if u > 0 else 0.0

    # Duplicate-overlap thresholds. Mask IoU is used whenever BOTH shapes are
    # polygons (precise: neighbouring/clustered objects score ~0, so they are
    # never wrongly suppressed), so it can be aggressive. The coarse box IoU
    # fallback (box mode) is kept stricter to avoid cancelling clustered boxes.
    _DUP_MASK_IOU = 0.40
    _DUP_BOX_IOU  = 0.60

    @staticmethod
    def _poly_iou_norm(poly_a, poly_b):
        """Mask IoU of two normalized polygons via shapely, or None if it
        cannot be computed (shapely missing / degenerate geometry)."""
        try:
            from shapely.geometry import Polygon as _P
            pa = _P(poly_a); pb = _P(poly_b)
            if not pa.is_valid:
                pa = pa.buffer(0)
            if not pb.is_valid:
                pb = pb.buffer(0)
            union = pa.union(pb).area
            return (pa.intersection(pb).area / union) if union > 0 else 0.0
        except Exception:
            return None

    def _is_duplicate_of(self, a, b):
        """True when annotation `a` is a duplicate of `b` (the same physical
        object). Uses precise MASK IoU when both are polygons, so two berries
        whose outlines merely sit side by side score ~0 and are kept, and falls
        back to a stricter bounding-box IoU for box-mode rectangles."""
        if a.get('type') == 'poly' and b.get('type') == 'poly':
            iou = self._poly_iou_norm(a['data'], b['data'])
            if iou is not None:
                return iou > self._DUP_MASK_IOU
        return self._bbox_iou(self._ann_bbox_norm_xyxy(a),
                              self._ann_bbox_norm_xyxy(b)) > self._DUP_BOX_IOU

    def set_annotations(self, polys=None, rects=None, poly_sources=None, rect_sources=None,
                        poly_cls=None, rect_cls=None):
        """Replace all annotations. `poly_sources[i]` / `rect_sources[i]` is
        'detector' or 'manual'; both default to 'detector' when omitted.
        The source decides outline color in edit mode and lets manual draws
        survive across regenerate calls. `poly_cls[i]` / `rect_cls[i]` are
        optional per-annotation class ids; when omitted the 'cls' key is left
        absent and readers fall back to 0 as before."""
        anns = []
        if polys:
            for i, poly in enumerate(polys):
                if len(poly) >= 3:
                    src_tag = (poly_sources[i] if poly_sources and i < len(poly_sources) else 'detector')
                    ann = {'type': 'poly', 'data': [list(p) for p in poly],
                           'deleted': False, 'source': src_tag}
                    if poly_cls is not None and i < len(poly_cls):
                        ann['cls'] = int(poly_cls[i])
                    anns.append(ann)
        if rects:
            for i, rect in enumerate(rects):
                src_tag = (rect_sources[i] if rect_sources and i < len(rect_sources) else 'detector')
                ann = {'type': 'rect', 'data': list(rect),
                       'deleted': False, 'source': src_tag}
                if rect_cls is not None and i < len(rect_cls):
                    ann['cls'] = int(rect_cls[i])
                anns.append(ann)
        # Input-only prompt boxes (positive AND negative) are NOT part of the
        # replaceable annotation set -- carry the existing ones over so they
        # survive every Regenerate / mode switch / detector run that rebuilds
        # annotations.
        prompts = [dict(a) for a in self.annotations
                   if a.get('source') in ('prompt', 'neg_prompt') and not a.get('deleted')]
        # Semi-auto / auto-draw SAM masks (poly, semiauto=True) are sticky: the
        # user authored them point-by-point, so a segmenter re-run / mode switch
        # must NEVER re-segment or drop them. Carry the full dicts (data,
        # sam_points, cls, semiauto) so all per-mask metadata survives. They are
        # kept out of live_boxes, so the detector polys in `anns` never
        # duplicate them. (Manual drawn-BOX polys are not tagged semiauto and
        # still regenerate from live_boxes as before.)
        sam_masks = [dict(a) for a in self.annotations
                     if a.get('semiauto') and a.get('type') == 'poly'
                     and not a.get('deleted')]
        # Hand-drawn masks WIN: drop any freshly-generated annotation that is a
        # duplicate of a sticky mask, so a Regenerate never stacks a model layer
        # on top of a region the user already masked. Uses precise mask IoU, so
        # clustered/neighbouring detections are preserved. A SpatialGrid prunes
        # the comparison to spatial neighbours (O(n) instead of O(n^2)).
        # kept_new[i] = whether the i-th freshly-built ann (polys then rects, in
        # input order) survived the sticky-mask dedup below. The caller shrinks
        # its parallel live_boxes / live_polys_cache in lockstep so the
        # positional alignment that _switch_to_mode relies on
        # (annotations[i] <-> live_boxes[i]) cannot drift when a detector poly is
        # dropped for duplicating a hand-drawn mask -- the cause of the "purple
        # masks vanish on Regenerate / mode switch" report.
        kept_new = [True] * len(anns)
        if sam_masks:
            grid = SpatialGrid.build(sam_masks, self._ann_bbox_norm_xyxy)
            survivors = []
            for _i, a in enumerate(anns):
                dup = any(self._is_duplicate_of(a, m)
                          for m in grid.query_bbox(self._ann_bbox_norm_xyxy(a)))
                kept_new[_i] = not dup
                if not dup:
                    survivors.append(a)
            anns = survivors
        self.annotations = anns + sam_masks + prompts
        self.selected_index = None
        self.selected_indices = set()
        self._persistent_marquee = None
        self._marquee_handle = None
        self._marquee_phase  = None
        self._invalidate_semiauto_selection()
        self.undo_stack  = []
        self.redo_stack  = []
        self.update()
        return kept_new

    def load_annotation_state(self, state):
        prompts = [dict(a) for a in self.annotations
                   if a.get('source') in ('prompt', 'neg_prompt') and not a.get('deleted')]
        self.annotations = [dict(a) for a in state] + prompts
        self.selected_index = None
        self.selected_indices = set()
        self._persistent_marquee = None
        self._marquee_handle = None
        self._marquee_phase  = None
        self._invalidate_semiauto_selection()
        self.undo_stack  = []
        self.redo_stack  = []
        self.update()

    def get_active_annotations(self):
        return [a for a in self.annotations if not a['deleted']]

    def get_saveable_annotations(self):
        """Active annotations minus input-only prompt boxes (positive 'prompt'
        and negative 'neg_prompt'), which are never written to label files or
        baked into saved images."""
        return [a for a in self.annotations
                if not a['deleted'] and a.get('source') not in ('prompt', 'neg_prompt')]

    _UNDO_MAX = 50

    def _push_undo(self):
        """Snapshot the FULL annotation state (geometry + flags), so undo/redo
        cover draws, deletes, and edits, not just the 'deleted' flag. Call
        BEFORE mutating self.annotations."""
        import copy
        self.undo_stack.append(copy.deepcopy(self.annotations))
        self.redo_stack.clear()
        if len(self.undo_stack) > self._UNDO_MAX:
            self.undo_stack.pop(0)

    def push_undo_semiauto_edit(self):
        """Undo snapshot for an in-progress drawn-mask edit. Uses
        _semiauto_orig_data for the selected mask so live vertex drags (which
        edit ann['data'] in place) are captured at their PRE-edit state."""
        import copy
        snap = copy.deepcopy(self.annotations)
        i = self._semiauto_sel_idx
        if (i is not None and 0 <= i < len(snap)
                and self._semiauto_orig_data is not None):
            snap[i]['data'] = [list(p) for p in self._semiauto_orig_data]
        self.undo_stack.append(snap)
        self.redo_stack.clear()
        if len(self.undo_stack) > self._UNDO_MAX:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            return
        import copy
        self.redo_stack.append(copy.deepcopy(self.annotations))
        self.annotations = self.undo_stack.pop()
        self.selected_index = None
        self.selected_indices = set()
        self._invalidate_semiauto_selection()
        self.update()
        self.boxes_changed.emit()

    def redo(self):
        if not self.redo_stack:
            return
        import copy
        self.undo_stack.append(copy.deepcopy(self.annotations))
        self.annotations = self.redo_stack.pop()
        self.selected_index = None
        self.selected_indices = set()
        self._invalidate_semiauto_selection()
        self.update()
        self.boxes_changed.emit()


    def _widget_xyxy_to_image(self, x1, y1, x2, y2):
        """Convert widget pixel xyxy -> absolute image pixel xyxy. Returns
        None if no image is loaded or the box ends up degenerate."""
        if not self._orig_w:
            return None
        scale, off_x, off_y = self._get_scale_offset()
        if scale <= 0:
            return None
        ix1 = (x1 - off_x) / scale
        iy1 = (y1 - off_y) / scale
        ix2 = (x2 - off_x) / scale
        iy2 = (y2 - off_y) / scale
        ix1 = max(0.0, min(float(self._orig_w), ix1))
        iy1 = max(0.0, min(float(self._orig_h), iy1))
        ix2 = max(0.0, min(float(self._orig_w), ix2))
        iy2 = max(0.0, min(float(self._orig_h), iy2))
        if ix2 - ix1 < 1 or iy2 - iy1 < 1:
            return None
        return (ix1, iy1, ix2, iy2)

    def _image_xyxy_to_widget(self, x1, y1, x2, y2):
        """Convert absolute image pixel xyxy -> current widget pixel xyxy.
        Returns None if no image is loaded."""
        if not self._orig_w:
            return None
        scale, off_x, off_y = self._get_scale_offset()
        return (x1 * scale + off_x, y1 * scale + off_y,
                x2 * scale + off_x, y2 * scale + off_y)

    def _drawn_boxes_image_coords(self, source='manual'):
        """Image-coord xyxy for every non-deleted rect ann of the given source
        ('manual' = saved draws, 'prompt' = input-only prompt boxes)."""
        if not self._orig_w:
            return []
        out = []
        for ann in self.annotations:
            if ann['deleted'] or ann.get('source') != source or ann['type'] != 'rect':
                continue
            cx, cy, w, h = ann['data']
            x1 = (cx - w / 2) * self._orig_w
            y1 = (cy - h / 2) * self._orig_h
            x2 = (cx + w / 2) * self._orig_w
            y2 = (cy + h / 2) * self._orig_h
            out.append([x1, y1, x2, y2])
        return out

    def set_active_draw_cls(self, cls):
        """Set the class id stamped onto newly drawn rects (multi-class box
        prompts / manual boxes). Driven by ManualWindow's active-class picker."""
        self.active_draw_cls = int(cls or 0)

    def get_prompt_boxes_in_image_coords(self):
        """Input-only prompt boxes (source='prompt') used as detector prompts."""
        return self._drawn_boxes_image_coords('prompt')

    def get_prompt_boxes_with_cls_in_image_coords(self):
        """Prompt boxes as (boxes_xyxy, cls_ids), parallel lists, in image
        coords. cls_ids come from each box's stored class so multi-class box
        prompts feed the detector with a real per-box class array (default 0)."""
        return self._input_boxes_with_cls('prompt')

    def get_neg_prompt_boxes_in_image_coords(self):
        """Input-only NEGATIVE prompt boxes (source='neg_prompt') in image
        coords. One red type; used as appearance exemplars to suppress matches."""
        return self._input_boxes_with_cls('neg_prompt')[0]

    def _input_boxes_with_cls(self, source):
        """Shared: (boxes_xyxy, cls_ids) for a given input-only rect source."""
        if not self._orig_w:
            return [], []
        boxes, cls = [], []
        for a in self.annotations:
            if a.get('deleted') or a.get('source') != source or a.get('type') != 'rect':
                continue
            cxn, cyn, wn, hn = a['data']
            x1 = (cxn - wn / 2) * self._orig_w
            y1 = (cyn - hn / 2) * self._orig_h
            x2 = (cxn + wn / 2) * self._orig_w
            y2 = (cyn + hn / 2) * self._orig_h
            boxes.append([x1, y1, x2, y2])
            cls.append(int(a.get('cls', 0) or 0))
        return boxes, cls

    def get_annotation_boxes_in_image_coords(self):
        """Drawn boxes saved as final annotations. Same set as prompts, a
        drawn box serves both roles."""
        return self._drawn_boxes_image_coords()

    def get_boxes_with_cls_in_image_coords(self):
        """Manually drawn boxes as (boxes_xyxy, cls_ids), parallel lists, in
        image coords. The manual-draw counterpart of
        get_prompt_boxes_with_cls_in_image_coords, so a hand-drawn box keeps the
        class it was drawn as when it is segmented and saved."""
        return self._input_boxes_with_cls('manual')

    def get_boxes_in_image_coords(self):
        """LEGACY alias for callers that haven't migrated."""
        return self._drawn_boxes_image_coords()

    # Property aliases so existing `self.image_label.prompt_boxes` and
    # `self.image_label.annotation_boxes` reads/writes continue to work.
    # Reads return the derived drawn-box list; writing `[]` clears all
    # manual rect anns SILENTLY (no boxes_changed.emit) to match the
    # mid-transition usage inside _switch_to_mode / display_*_with_borders.
    @property
    def prompt_boxes(self):
        return self._drawn_boxes_image_coords()

    @prompt_boxes.setter
    def prompt_boxes(self, value):
        if value:
            raise TypeError("Drawn boxes are stored in self.annotations; "
                            "non-empty assignment to prompt_boxes is not supported.")
        # No-op for `= []`: callers that "clear the bucket" do so AFTER
        # set_annotations has already installed the new state. Use
        # clear_boxes() explicitly when you really want to drop drawn boxes.

    @property
    def annotation_boxes(self):
        return self._drawn_boxes_image_coords()

    @annotation_boxes.setter
    def annotation_boxes(self, value):
        if value:
            raise TypeError("Drawn boxes are stored in self.annotations; "
                            "non-empty assignment to annotation_boxes is not supported.")
        # No-op for `= []`: see prompt_boxes.setter for the rationale.

    def get_active_rects_in_image_coords(self):
        """Return non-deleted rect annotations as absolute xyxy in image coords."""
        if not self._orig_w:
            return []
        out = []
        for ann in self.annotations:
            if ann['deleted'] or ann['type'] != 'rect':
                continue
            cx, cy, w, h = ann['data']
            x1 = (cx - w / 2) * self._orig_w
            y1 = (cy - h / 2) * self._orig_h
            x2 = (cx + w / 2) * self._orig_w
            y2 = (cy + h / 2) * self._orig_h
            out.append([x1, y1, x2, y2])
        return out

    def get_active_rects_with_sources(self):
        """Like get_active_rects_in_image_coords but returns (xyxy, source) tuples
        so the caller can preserve the detector/manual distinction."""
        if not self._orig_w:
            return []
        out = []
        for ann in self.annotations:
            if ann['deleted'] or ann['type'] != 'rect':
                continue
            cx, cy, w, h = ann['data']
            x1 = (cx - w / 2) * self._orig_w
            y1 = (cy - h / 2) * self._orig_h
            x2 = (cx + w / 2) * self._orig_w
            y2 = (cy + h / 2) * self._orig_h
            out.append(([x1, y1, x2, y2], ann.get('source', 'detector')))
        return out

    def get_manual_rects_in_image_coords(self):
        """LEGACY: rect-only manual boxes. Prefer get_manual_anns_as_boxes_in_image_coords."""
        return [b for b, s in self.get_active_rects_with_sources() if s == 'manual']

    def get_manual_anns_as_boxes_with_classes_in_image_coords(self):
        """(xyxy, cls) for every non-deleted manual annotation, REGARDLESS of
        underlying type. For rect anns, the rect itself; for poly anns, the
        polygon's bounding box.

        This is the helper to use for carry-forward on regenerate, because in
        seg mode prior-iteration manual draws are stored as polygons (SAM's
        mask of the user's box), not as rects. The rect-only variant misses
        those and silently drops the user's earlier draws.

        Each box carries the class it was DRAWN as. Callers must not re-tag them
        with the currently-selected class: the user can draw class A, switch the
        dropdown to class B, and regenerate, and their class-A boxes have to
        stay class A.
        """
        if not self._orig_w:
            return []
        out = []
        for ann in self.annotations:
            if ann['deleted'] or ann.get('source') != 'manual':
                continue
            cls = int(ann.get('cls', 0))
            if ann['type'] == 'rect':
                cx, cy, w, h = ann['data']
                out.append(([
                    (cx - w / 2) * self._orig_w,
                    (cy - h / 2) * self._orig_h,
                    (cx + w / 2) * self._orig_w,
                    (cy + h / 2) * self._orig_h,
                ], cls))
            elif ann['type'] == 'poly':
                xs = [p[0] for p in ann['data']]
                ys = [p[1] for p in ann['data']]
                out.append(([
                    min(xs) * self._orig_w, min(ys) * self._orig_h,
                    max(xs) * self._orig_w, max(ys) * self._orig_h,
                ], cls))
        return out

    def get_manual_anns_as_boxes_in_image_coords(self):
        """Boxes only, same order as the with-classes variant it delegates to."""
        return [b for b, _ in
                self.get_manual_anns_as_boxes_with_classes_in_image_coords()]

    def get_active_polys_with_sources(self):
        """Returns (poly, source) tuples for non-deleted poly annotations."""
        out = []
        for ann in self.annotations:
            if ann['deleted'] or ann['type'] != 'poly':
                continue
            out.append(([list(p) for p in ann['data']], ann.get('source', 'detector')))
        return out

    # internal helpers
    def _get_scale_offset(self):
        if not self._orig_w or not self._orig_h:
            return 1.0, 0.0, 0.0
        lw, lh = self.width(), self.height()
        base   = min(lw / self._orig_w, lh / self._orig_h)   # fit-to-window
        scale  = base * self._zoom
        off_x  = (lw - self._orig_w * scale) / 2 + self._pan_x
        off_y  = (lh - self._orig_h * scale) / 2 + self._pan_y
        return scale, off_x, off_y

    # Zoom / pan (Image Resize mode)
    def set_resize_mode(self, enabled):
        """Arm zoom/pan per the input scheme (Trackpad: two-finger scroll pans,
        pinch or Ctrl+wheel zooms; Mouse: wheel zooms, right-drag pans). The
        left button keeps DRAWING and editing even while the mode is on (every
        coordinate routes through _get_scale_offset, so a zoomed draw lands
        exactly where it looks). The zoom/pan PERSISTS after the mode is turned
        off; only reset_view() (Save & Confirm / new image) returns to fit."""
        self._resize_mode = bool(enabled)
        self._pan_drag_last  = None
        self._pan_drag_moved = False

    def reset_view(self):
        """Return to fit-to-window (zoom 1.0, no pan)."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def resizeEvent(self, event):
        # Re-clamp pan when the window/widget changes size so a zoomed-in,
        # panned image can't get stranded off-screen after a resize. The base
        # fit-scale is recomputed every paint; only the pan offset needs this.
        self._clamp_view()
        super().resizeEvent(event)

    def _clamp_view(self):
        """Keep zoom in [1.0, 8.0] and stop the image being panned entirely
        off-screen (recentre when back at fit)."""
        self._zoom = max(1.0, min(8.0, self._zoom))
        if self._zoom <= 1.0 or not self._orig_w or not self._orig_h:
            self._pan_x = 0.0
            self._pan_y = 0.0
            return
        lw, lh = self.width(), self.height()
        base = min(lw / self._orig_w, lh / self._orig_h)
        scale = base * self._zoom
        img_w = self._orig_w * scale
        img_h = self._orig_h * scale
        max_x = max(0.0, (img_w - lw) / 2)
        max_y = max(0.0, (img_h - lh) / 2)
        self._pan_x = max(-max_x, min(max_x, self._pan_x))
        self._pan_y = max(-max_y, min(max_y, self._pan_y))

    def _zoom_at(self, factor, wx, wy):
        """Multiply zoom by `factor`, keeping the image point under widget
        pixel (wx, wy) fixed (zoom toward cursor)."""
        if not self._orig_w or not self._orig_h:
            return
        scale, off_x, off_y = self._get_scale_offset()
        ix = (wx - off_x) / scale
        iy = (wy - off_y) / scale
        self._zoom *= factor
        self._clamp_view()
        lw, lh = self.width(), self.height()
        base = min(lw / self._orig_w, lh / self._orig_h)
        nscale = base * self._zoom
        center_off_x = (lw - self._orig_w * nscale) / 2
        center_off_y = (lh - self._orig_h * nscale) / 2
        self._pan_x = (wx - ix * nscale) - center_off_x
        self._pan_y = (wy - iy * nscale) - center_off_y
        self._clamp_view()
        self.update()

    def wheelEvent(self, event):
        if not self._resize_mode or not self._orig_w:
            super().wheelEvent(event)
            return
        # The left button stays free for drawing on the zoomed view in both
        # schemes. Trackpad: scroll PANS, Ctrl/Cmd+scroll (or pinch, in
        # event()) zooms. Mouse: the wheel ZOOMS; panning is a right-drag
        # handled in the mouse handlers, never a wheel action.
        mods = event.modifiers()
        pd = event.pixelDelta()
        dx, dy = pd.x(), pd.y()
        if dx == 0 and dy == 0:
            ad = event.angleDelta()
            dx, dy = ad.x() / 2.0, ad.y() / 2.0
        action = session_state.classify_wheel(
            session_state.input_scheme(), dx, dy,
            ctrl=bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier)),
            shift=bool(mods & QtCore.Qt.ShiftModifier))
        if action is None:
            return
        if action[0] == "zoom":
            pos = event.pos()
            self._zoom_at(1.0015 ** action[1], pos.x(), pos.y())
        else:
            self._pan_x += action[1]
            self._pan_y += action[2]
            self._clamp_view()
            self.update()
        event.accept()

    def event(self, e):
        # macOS trackpad pinch-to-zoom arrives as a native gesture, not a wheel.
        if (self._resize_mode and self._orig_w
                and e.type() == QtCore.QEvent.NativeGesture):
            try:
                if e.gestureType() == QtCore.Qt.ZoomNativeGesture:
                    p = e.pos()
                    self._zoom_at(1.0 + float(e.value()), p.x(), p.y())
                    e.accept()
                    return True
            except Exception:
                pass
        return super().event(e)

    def _to_label(self, poly_norm):
        scale, off_x, off_y = self._get_scale_offset()
        return [
            (x * self._orig_w * scale + off_x, y * self._orig_h * scale + off_y)
            for x, y in poly_norm
        ]

    def _rect_to_label(self, cx, cy, w, h):
        scale, off_x, off_y = self._get_scale_offset()
        x1 = (cx - w / 2) * self._orig_w * scale + off_x
        y1 = (cy - h / 2) * self._orig_h * scale + off_y
        x2 = (cx + w / 2) * self._orig_w * scale + off_x
        y2 = (cy + h / 2) * self._orig_h * scale + off_y
        return x1, y1, x2, y2

    def _x_center_for_ann(self, ann):
        if ann['type'] == 'poly':
            pts = self._to_label(ann['data'])
            if not pts:
                return 0, 0
            # Float the delete badge just OUTSIDE the polygon's top-right
            # bounding-box corner. Every vertex lies on/within the bbox, so an
            # outward offset keeps the X clear of every draggable handle, so a
            # point-drag can't land on the badge and misfire as a mask delete.
            gap = self._HIT + self._HANDLE_R
            bx = max(p[0] for p in pts) + gap
            by = min(p[1] for p in pts) - gap
            # Keep the whole badge on-screen.
            bx = min(bx, self.width() - self._X_R)
            by = max(by, self._X_R)
            return int(bx), int(by)
        else:  # rect: center of the rect, not the corner.
            x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
            return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _hit_test_ann(self, pos):
        """Return the index of the topmost non-deleted annotation under `pos`,
        or None. Polygons use point-in-polygon; rects use bbox containment."""
        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if ann['deleted']:
                continue
            if ann['type'] == 'rect':
                x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
                if x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2:
                    return idx
            else:  # poly
                pts = self._to_label(ann['data'])
                if len(pts) < 3:
                    continue
                qpoly = QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in pts])
                if qpoly.containsPoint(pos, QtCore.Qt.OddEvenFill):
                    return idx
        return None

    def _delete_selected_ann(self):
        """Soft-delete every annotation in the current selection.

        Honors selected_indices when non-empty (multi-select bulk
        delete) and falls back to selected_index for the legacy
        single-selection path. Returns True if at least one annotation
        was deleted."""
        # Build a working set of indices to delete.
        targets = set(self.selected_indices)
        if self.selected_index is not None:
            targets.add(self.selected_index)
        # Filter to valid, non-deleted indices.
        live = [i for i in targets
                if 0 <= i < len(self.annotations)
                and not self.annotations[i]['deleted']]
        if not live:
            return False
        self._push_undo()
        for i in live:
            self.annotations[i]['deleted'] = True
        self.selected_index = None
        self.selected_indices = set()
        self.update()
        self.boxes_changed.emit()
        return True

    # rect resize handles (edit-mode only, selected rect only)
    _HANDLE_R  = 6   # half-size of the visible square handle (px)
    _HANDLE_HR = 10  # click hit radius around each handle center (px)

    def _rect_handle_centers(self, idx):
        """Return {'tl','tr','bl','br': (x,y)} widget-coord handle centers
        for annotations[idx], assuming it's a non-deleted rect."""
        ann = self.annotations[idx]
        if ann['type'] != 'rect' or ann['deleted']:
            return None
        x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
        return {
            'tl': (x1, y1),
            'tr': (x2, y1),
            'bl': (x1, y2),
            'br': (x2, y2),
        }

    def _hit_rect_handle(self, pos, idx):
        """Return 'tl'|'tr'|'bl'|'br' if `pos` is within _HANDLE_HR of any
        corner handle of annotations[idx]; else None."""
        centers = self._rect_handle_centers(idx)
        if centers is None:
            return None
        r2 = self._HANDLE_HR ** 2
        for name, (cx, cy) in centers.items():
            if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= r2:
                return name
        return None

    def _update_rect_from_resize(self, pos):
        """Move the dragged corner to `pos` (clipped to the image area, with
        a min widget-pixel size floor) and write the new normalized rect back
        to self.annotations[_resize_ann_idx]['data']."""
        idx = self._resize_ann_idx
        if idx is None or idx >= len(self.annotations):
            return
        ann = self.annotations[idx]
        if ann['type'] != 'rect':
            return
        x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
        # Clip mouse pos to the image area (so handles can't escape onto the
        # gray letterbox; matches the draw-mode behavior).
        ix1, iy1, ix2, iy2 = self._image_area_widget_rect()
        px = max(ix1, min(pos.x(), ix2))
        py = max(iy1, min(pos.y(), iy2))
        h = self._resize_handle
        if h == 'tl':
            nx1, ny1, nx2, ny2 = px, py, x2, y2
        elif h == 'tr':
            nx1, ny1, nx2, ny2 = x1, py, px, y2
        elif h == 'bl':
            nx1, ny1, nx2, ny2 = px, y1, x2, py
        elif h == 'br':
            nx1, ny1, nx2, ny2 = x1, y1, px, py
        else:
            return
        # Enforce a minimum widget-pixel size so the rect doesn't collapse to
        # zero area (which would make the next hit-test impossible). Pin the
        # ANCHORED edge and push the dragged edge out by the floor amount.
        MIN_SIZE = 8
        if nx2 - nx1 < MIN_SIZE:
            if h in ('tl', 'bl'):
                nx1 = nx2 - MIN_SIZE
            else:
                nx2 = nx1 + MIN_SIZE
        if ny2 - ny1 < MIN_SIZE:
            if h in ('tl', 'tr'):
                ny1 = ny2 - MIN_SIZE
            else:
                ny2 = ny1 + MIN_SIZE
        # Convert widget coords back to normalized (cx, cy, w, h).
        img = self._widget_xyxy_to_image(nx1, ny1, nx2, ny2)
        if img is None or self._orig_w is None:
            return
        ix1n, iy1n, ix2n, iy2n = img
        cxn = (ix1n + ix2n) / 2 / self._orig_w
        cyn = (iy1n + iy2n) / 2 / self._orig_h
        wn  = (ix2n - ix1n) / self._orig_w
        hn  = (iy2n - iy1n) / self._orig_h
        ann['data'] = [cxn, cyn, wn, hn]

    # mouse events
    @staticmethod
    def _cursor_for_corner(corner):
        """Map a corner name to its Qt diagonal-resize cursor. tl/br go
        on one (\\) diagonal, tr/bl on the other (/), so the cursor's
        slant matches the direction the corner moves under the drag."""
        return (QtCore.Qt.SizeFDiagCursor
                if corner in ('tl', 'br')
                else QtCore.Qt.SizeBDiagCursor)

    def _resolve_hover_cursor(self, pos):
        """Pick the right cursor for `pos` based on what's under the
        mouse RIGHT NOW. Walks the same hit tests mousePressEvent uses,
        so what the cursor advertises and what the next click actually
        does stay in sync. Called from mouseMoveEvent in the idle
        (no-drag) path."""
        if not self.edit_mode:
            return None  # mode-default cursor; set elsewhere
        # Marquee corner takes priority; it sits on top of any ann.
        if (self.multi_select_mode and self._persistent_marquee is not None):
            mwidget = self._image_xyxy_to_widget(*self._persistent_marquee)
            if mwidget is not None:
                h = self._hit_marquee_handle(pos, mwidget)
                if h is not None:
                    return self._cursor_for_corner(h)
        # Rect-annotation corner on the PRIMARY single selection
        # (resize handles are suppressed in multi-select mode).
        if (not self.multi_select_mode
                and self.selected_index is not None
                and self.selected_index < len(self.annotations)):
            sel = self.annotations[self.selected_index]
            if (not sel['deleted'] and sel['type'] == 'rect'):
                h = self._hit_rect_handle(pos, self.selected_index)
                if h is not None:
                    return self._cursor_for_corner(h)
        return None

    def _apply_hover_cursor(self, pos):
        """Resolve + apply the hover cursor. Falls back to the
        mode-default cursor (crosshair in draw_mode, arrow otherwise)
        when nothing under the pointer needs a special shape."""
        hover = self._resolve_hover_cursor(pos)
        if hover is not None:
            self.setCursor(hover)
            return
        # No handle under the mouse, so restore the mode-default.
        self.setCursor(QtCore.Qt.CrossCursor if self.draw_mode else QtCore.Qt.ArrowCursor)

    def _hit_marquee_handle(self, pos, marquee_widget_xyxy):
        """Return 'tl' | 'tr' | 'bl' | 'br' if `pos` is within the click
        radius of one of the persistent marquee's corner handles."""
        wx1, wy1, wx2, wy2 = marquee_widget_xyxy
        centers = {'tl': (wx1, wy1), 'tr': (wx2, wy1),
                   'bl': (wx1, wy2), 'br': (wx2, wy2)}
        r2 = self._HANDLE_HR ** 2
        for name, (cx, cy) in centers.items():
            if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= r2:
                return name
        return None

    def _update_marquee_from_resize(self, pos):
        """Drag one corner of the persistent marquee to `pos`. Mirrors
        _update_rect_from_resize but writes back to _persistent_marquee
        instead of an annotation. Persistent marquee is stored in IMAGE
        coords so it stays anchored when the widget is resized."""
        if self._marquee_handle is None or self._persistent_marquee is None:
            return
        mwidget = self._image_xyxy_to_widget(*self._persistent_marquee)
        if mwidget is None:
            return
        wx1, wy1, wx2, wy2 = mwidget
        ix1, iy1, ix2, iy2 = self._image_area_widget_rect()
        px = max(ix1, min(pos.x(), ix2))
        py = max(iy1, min(pos.y(), iy2))
        h = self._marquee_handle
        if   h == 'tl': nx1, ny1, nx2, ny2 = px, py, wx2, wy2
        elif h == 'tr': nx1, ny1, nx2, ny2 = wx1, py, px, wy2
        elif h == 'bl': nx1, ny1, nx2, ny2 = px, wy1, wx2, py
        elif h == 'br': nx1, ny1, nx2, ny2 = wx1, wy1, px, py
        else: return
        if nx2 < nx1: nx1, nx2 = nx2, nx1
        if ny2 < ny1: ny1, ny2 = ny2, ny1
        # Floor pixel size so the marquee can't collapse to zero.
        if nx2 - nx1 < 8 or ny2 - ny1 < 8:
            return
        img_box = self._widget_xyxy_to_image(nx1, ny1, nx2, ny2)
        if img_box is None:
            return
        self._persistent_marquee = list(img_box)

    def _recompute_marquee_selection(self):
        """Reset selection to exactly the annotations intersecting the
        current persistent marquee (dedup-by-IoU is in
        _select_in_widget_rect). Called after every marquee draw or
        resize so the cyan/X overlay tracks the marquee live."""
        self.selected_indices = set()
        self.selected_index   = None
        if self._persistent_marquee is None:
            return
        mwidget = self._image_xyxy_to_widget(*self._persistent_marquee)
        if mwidget is None:
            return
        mx1, my1, mx2, my2 = mwidget
        self._select_in_widget_rect(mx1, my1, mx2, my2)

    def _select_in_widget_rect(self, mx1, my1, mx2, my2):
        """Pick every non-deleted annotation whose widget-space bbox
        intersects the marquee rect, then collapse perfectly-overlapping
        hits to a single representative so the user doesn't end up with
        five stacked duplicates in the selection. Honors the Shift key
        held when the marquee was started by EXTENDING the current
        selection instead of replacing it."""
        marquee = (mx1, my1, mx2, my2)
        hits = []
        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if ann['deleted']:
                continue
            if ann['type'] == 'rect':
                x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
            else:
                pts = self._to_label(ann['data'])
                if len(pts) < 3:
                    continue
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            # Intersect-with-marquee: a hit is anything whose bbox
            # overlaps the marquee rect (cheap rect-rect test).
            if x2 < marquee[0] or x1 > marquee[2]:
                continue
            if y2 < marquee[1] or y1 > marquee[3]:
                continue
            hits.append((idx, [x1, y1, x2, y2]))
        # Dedupe near-perfect overlaps (IoU > 0.9). Iterating in
        # reverse-Z order means the topmost ann wins.
        def _iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
            ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            u = ua + ub - inter
            return inter / u if u > 0 else 0.0
        kept_idx = []
        kept_bbs = []
        for idx, bb in hits:
            if any(_iou(bb, kb) > 0.9 for kb in kept_bbs):
                continue
            kept_idx.append(idx)
            kept_bbs.append(bb)
        if not kept_idx:
            return
        # If the press cleared the selection set we replace it; if it
        # left the set in place (Shift was held on press) we extend.
        # `selected_indices` already reflects that decision.
        self.selected_indices = set(self.selected_indices) | set(kept_idx)
        if self.selected_index is None or self.selected_index not in self.selected_indices:
            self.selected_index = kept_idx[0]

    def _right_click_action(self, pos):
        """Everything a right CLICK means, in mode order. Shared by a plain
        press and by a resize-mode right press that came back up without
        dragging (a right DRAG pans instead; see mousePressEvent)."""
        if self.mask_draw_mode and self._orig_w:
            hit = self._hit_mask_point(pos)
            if hit is not None:
                self._mask_points.pop(hit)
                if self._mask_draw_kind == "autodraw":
                    self.mask_point_added.emit()   # refresh the live preview
                self.update()
            return
        if self.semiauto_edit_mode and self._orig_w:
            if self._semiauto_sel_idx is None:
                return
            if self._semiauto_edit_target == "vertices":
                hit_v = self._hit_vertex(pos)
                if hit_v is not None:
                    ann = self._semiauto_selected_ann()
                    if ann is not None:
                        if len(ann['data']) > 3:
                            del ann['data'][hit_v]
                            self._mask_preview_poly = [list(p) for p in ann['data']]
                            self.update()
                        else:
                            # At the 3-point floor, can't go lower; offer to
                            # delete the whole mask instead.
                            self.semiauto_min_vertex_delete.emit()
                return
            hit_pt = self._hit_mask_point(pos)
            if hit_pt is not None:
                self._mask_points.pop(hit_pt)      # remove a point
                self.mask_point_added.emit()
                self.update()
            return
        # Default: delete drawn manual/prompt rect anns whose widget projection
        # contains the click point. Detector-output and polygon anns untouched.
        removed = False
        if self._orig_w:
            for ann in self.annotations:
                if ann['deleted'] or ann.get('source') not in ('manual', 'prompt') or ann['type'] != 'rect':
                    continue
                cx, cy, w, h = ann['data']
                x1 = (cx - w / 2) * self._orig_w
                y1 = (cy - h / 2) * self._orig_h
                x2 = (cx + w / 2) * self._orig_w
                y2 = (cy + h / 2) * self._orig_h
                wb = self._image_xyxy_to_widget(x1, y1, x2, y2)
                if wb is None:
                    continue
                if self._hit_box(pos, wb):
                    ann['deleted'] = True
                    removed = True
        self.update()
        if removed:
            self.boxes_changed.emit()

    def mousePressEvent(self, event):
        # Image Resize: a right-button DRAG pans the zoomed view (the Mouse
        # scheme's pan gesture; harmless under Trackpad). Only armed here; if
        # the button comes straight back up, mouseReleaseEvent runs the normal
        # right-click action, so click-to-delete and point removal still work.
        if (self._resize_mode and self._orig_w
                and event.button() == QtCore.Qt.RightButton):
            self._pan_drag_last  = event.pos()
            self._pan_drag_moved = False
            return
        # Interactive SAM mask drawing takes first crack. All points are
        # FOREGROUND. Consumes the gesture so it never falls through to draw/edit.
        #   autodraw -> a single point; SAM previews live, Enter commits.
        #   semiauto -> connected outline points accumulate with NO SAM. SAM runs
        #               + commits only when the outline is CLOSED (click the
        #               first point again / double-click / Enter), Google-Draw
        #               curve-tool style.
        # Right-click removes the nearest point. Image Resize does not suspend
        # this: panning never uses the left button, so clicks stay meaningful.
        if self.mask_draw_mode and self._orig_w:
            if event.button() == QtCore.Qt.RightButton:
                self._right_click_action(event.pos())
                return
            if event.button() == QtCore.Qt.LeftButton:
                if self._mask_draw_kind == "autodraw":
                    p = self._widget_point_to_image(event.pos())
                    if p is not None:
                        # Roboflow-style refine: a click OUTSIDE the current mask
                        # adds a positive (expand) point; a click INSIDE it adds a
                        # negative (prune) point. Points accumulate, so two nearby
                        # clicks join objects and inside-clicks subtract regions.
                        lab = 0 if self._point_in_preview(event.pos()) else 1
                        self._mask_points.append([p[0], p[1], lab])
                        self.mask_point_added.emit()            # live SAM re-run
                        self.update()
                    return
                # semiauto: clicking the first point (with >=3 placed) closes
                # the outline and triggers SAM + commit.
                if len(self._mask_points) >= 3 and self._near_first_point(event.pos()):
                    self.mask_close_requested.emit()
                    return
                p = self._widget_point_to_image(event.pos())
                if p is not None:
                    self._mask_points.append([p[0], p[1], 1])   # NO SAM yet
                    self.update()
            return
        # Edit-Semi-Auto-Segments: click to select a committed mask, then edit
        # either its SAM points (live re-run) or its polygon vertices (manual).
        if self.semiauto_edit_mode and self._orig_w \
                and event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            if event.button() == QtCore.Qt.RightButton:
                self._right_click_action(event.pos())
                return
            if self._semiauto_sel_idx is None:
                if event.button() == QtCore.Qt.LeftButton:
                    hit = self._hit_semiauto_poly(event.pos())
                    if hit is not None:
                        self._select_semiauto(hit)
                        # Only re-run SAM on selection when the mask carries
                        # stored SAM points (hand-drawn). Model masks have none,
                        # so we keep the seeded polygon preview rather than
                        # clearing it to an empty SAM result.
                        if (self._semiauto_edit_target == "points"
                                and self.annotations[hit].get('sam_points')):
                            self.mask_point_added.emit()  # show SAM preview
                        else:
                            # Vertex editing: let the controller auto-thin a
                            # dense model contour so the handles are workable.
                            self.mask_selected.emit()
                return
            # X badge on the selected mask deletes the whole mask. Vertex-drag
            # WINS this hit-test: if the click is on a handle, it is never read
            # as an X press, so moving a point can't misfire as a mask delete.
            if event.button() == QtCore.Qt.LeftButton:
                _ann = self._semiauto_selected_ann()
                on_vertex = (self._semiauto_edit_target == "vertices"
                             and self._hit_vertex(event.pos()) is not None)
                if _ann is not None and not on_vertex:
                    _cx, _cy = self._x_center_for_ann(_ann)
                    if (event.pos().x() - _cx) ** 2 + (event.pos().y() - _cy) ** 2 \
                            <= self._HIT ** 2:
                        self.semiauto_delete_requested.emit()
                        return
            if self._semiauto_edit_target == "vertices":
                hit_v = self._hit_vertex(event.pos())
                if hit_v is not None and event.button() == QtCore.Qt.LeftButton:
                    self._vertex_drag_idx = hit_v        # start dragging a vertex
                    return
                if event.button() == QtCore.Qt.LeftButton:
                    self._insert_vertex(event.pos())     # add a vertex on the outline
                return
            # points target
            hit_pt = self._hit_mask_point(event.pos())
            if hit_pt is not None and event.button() == QtCore.Qt.LeftButton:
                self._semiauto_drag_pt = hit_pt          # start dragging a point
                return
            if event.button() == QtCore.Qt.LeftButton:    # add a foreground point
                p = self._widget_point_to_image(event.pos())
                if p is not None:
                    self._mask_points.append([p[0], p[1], 1])
                    self.mask_point_added.emit()
                    self.update()
            return
        if event.button() == QtCore.Qt.LeftButton:
            # Edit mode gets first crack at the click. It only consumes the click
            # if it hit an annotation (X badge or body); empty-canvas clicks fall
            # through so draw_mode can still start a drag when both modes are on.
            edit_consumed = False
            if self.edit_mode:
                pos = event.pos()
                # 1) X-badge hit on ANY currently-selected annotation.
                # Each selected ann renders its own X in multi-select
                # mode, so the hit test has to walk the whole set.
                targets = set(self.selected_indices)
                if self.selected_index is not None:
                    targets.add(self.selected_index)
                badge_hit = None
                for idx in targets:
                    if idx < 0 or idx >= len(self.annotations):
                        continue
                    a = self.annotations[idx]
                    if a['deleted']:
                        continue
                    cx, cy = self._x_center_for_ann(a)
                    if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= self._HIT ** 2:
                        badge_hit = idx
                        break
                if badge_hit is not None:
                    self._push_undo()
                    self.annotations[badge_hit]['deleted'] = True
                    self.selected_indices.discard(badge_hit)
                    if self.selected_index == badge_hit:
                        self.selected_index = next(iter(self.selected_indices), None)
                    self.update()
                    self.boxes_changed.emit()
                    return
                # 2) Rect-resize handle on the PRIMARY single-selected
                # rect (suppressed in multi-select mode, where the
                # marquee is the resizable thing, not individual rects).
                if (not self.multi_select_mode
                        and self.selected_index is not None
                        and self.selected_index < len(self.annotations)):
                    sel = self.annotations[self.selected_index]
                    if not sel['deleted'] and sel['type'] == 'rect':
                        h = self._hit_rect_handle(pos, self.selected_index)
                        if h is not None:
                            self._resize_handle  = h
                            self._resize_ann_idx = self.selected_index
                            return
                # 3) Marquee corner-handle hit (multi-select mode only).
                if (self.multi_select_mode and self._persistent_marquee is not None):
                    mwidget = self._image_xyxy_to_widget(*self._persistent_marquee)
                    if mwidget is not None:
                        h = self._hit_marquee_handle(pos, mwidget)
                        if h is not None:
                            self._marquee_handle = h
                            self._marquee_phase  = 'resizing'
                            return
                # 4) Annotation body hit.
                hit = self._hit_test_ann(pos)
                shift_held = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
                if hit is not None:
                    if shift_held:
                        # Toggle hit in/out of the multi-selection.
                        if hit in self.selected_indices:
                            self.selected_indices.discard(hit)
                            if self.selected_index == hit:
                                self.selected_index = next(iter(self.selected_indices), None)
                        else:
                            self.selected_indices.add(hit)
                            if self.selected_index is None:
                                self.selected_index = hit
                    else:
                        # Plain click: replace selection with this one.
                        # In multi-select mode this also dismisses the
                        # marquee so the user can directly act on a
                        # single ann (resize/delete) without escape.
                        self.selected_index = hit
                        self.selected_indices = {hit}
                        if self.multi_select_mode:
                            self._persistent_marquee = None
                    edit_consumed = True
                else:
                    # Empty-canvas click. In multi-select mode, start a
                    # new persistent marquee (dismisses any previous
                    # one). Outside multi-select, just clear selection.
                    if self.multi_select_mode and not self.draw_mode:
                        self._persistent_marquee = None
                        self.selected_index = None
                        self.selected_indices = set()
                        self._marquee_start = pos
                        self._marquee_cur   = pos
                        self._marquee_phase = 'drawing'
                        edit_consumed = True
                    elif not self.draw_mode:
                        self.selected_index = None
                        self.selected_indices = set()
                    else:
                        # draw_mode handles the drag below, so clear any
                        # leftover selection so the yellow-prompt path
                        # gets a clean slate.
                        self.selected_index = None
                        self.selected_indices = set()
                self.update()
            if not edit_consumed and self.draw_mode:
                self._drag_start = event.pos()
                self._drag_cur   = event.pos()
        elif event.button() == QtCore.Qt.RightButton:
            self._right_click_action(event.pos())

    def mouseDoubleClickEvent(self, event):
        # Semi-auto: double-click closes the outline (alternative to clicking the
        # first point). The double-click's second press already appended a point
        # at the same spot, so drop that duplicate before closing.
        if self.mask_draw_mode and self._mask_draw_kind == "semiauto" \
                and event.button() == QtCore.Qt.LeftButton:
            if len(self._mask_points) >= 2 and self._mask_points[-1] == self._mask_points[-2]:
                self._mask_points.pop()
            if len(self._mask_points) >= 3:
                self.mask_close_requested.emit()
            else:
                self.update()
            return
        # Edit (vertices target): double-click a vertex to delete it. At the
        # 3-point minimum a polygon can't lose a vertex, so hand off to the
        # controller to ask whether to delete the whole mask.
        if (self.semiauto_edit_mode
                and self._semiauto_edit_target == "vertices"
                and self._semiauto_sel_idx is not None
                and event.button() == QtCore.Qt.LeftButton):
            hit_v = self._hit_vertex(event.pos())
            if hit_v is not None:
                ann = self._semiauto_selected_ann()
                if ann is not None:
                    if len(ann['data']) > 3:
                        del ann['data'][hit_v]
                        self._vertex_drag_idx = None   # the first press armed a drag
                        self._mask_preview_poly = [list(p) for p in ann['data']]
                        self.update()
                    else:
                        self._vertex_drag_idx = None
                        self.semiauto_min_vertex_delete.emit()
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        # Armed right-button pan. A dead-zone keeps a slightly wobbly right
        # CLICK from turning into a pan and losing its click action.
        if self._pan_drag_last is not None:
            d = event.pos() - self._pan_drag_last
            if self._pan_drag_moved or d.manhattanLength() >= 4:
                self._pan_drag_moved = True
                self._pan_x += d.x()
                self._pan_y += d.y()
                self._pan_drag_last = event.pos()
                self._clamp_view()
                self.update()
            return
        # Semi-auto drawing: track the cursor so paintEvent can draw a rubber-band
        # line from the last placed point to the cursor (Google-Draw style).
        if self.mask_draw_mode and self._mask_draw_kind == "semiauto" \
                and self._mask_points:
            self._mask_cursor = event.pos()
            self.update()
            return
        # Dragging a SAM point in Edit-Semi-Auto-Segments mode (re-run on release).
        if self.semiauto_edit_mode and self._semiauto_drag_pt is not None:
            p = self._widget_point_to_image(event.pos())
            if p is not None:
                lab = self._mask_points[self._semiauto_drag_pt][2]
                self._mask_points[self._semiauto_drag_pt] = [p[0], p[1], lab]
                self.update()
            return
        # Dragging a polygon vertex (manual reshape, edits ann['data'] live).
        if self.semiauto_edit_mode and self._vertex_drag_idx is not None:
            p = self._widget_point_to_image(event.pos())
            ann = self._semiauto_selected_ann()
            if p is not None and ann is not None:
                if self._vertex_drag_idx < len(ann['data']):
                    ann['data'][self._vertex_drag_idx] = [p[0] / self._orig_w,
                                                          p[1] / self._orig_h]
                    self._mask_preview_poly = [list(pp) for pp in ann['data']]
                    self.update()
            else:
                self._vertex_drag_idx = None
            return
        # Vertex-edit hover: show a "+" ghost on the outline where a click would
        # add a vertex (cleared when hovering directly over an existing vertex).
        if self.semiauto_edit_mode and self._semiauto_sel_idx is not None \
                and self._semiauto_edit_target == "vertices":
            ghost = None
            if self._hit_vertex(event.pos()) is None:
                near = self._nearest_outline_point(event.pos())
                if near is not None and near[0] <= 14 * 14:
                    ghost = QtCore.QPoint(int(near[1]), int(near[2]))
            if ghost != self._vertex_ghost:
                self._vertex_ghost = ghost
                self.update()
            self._apply_hover_cursor(event.pos())
            return
        if self._resize_handle is not None:
            self._update_rect_from_resize(event.pos())
            self.update()
            return
        if self._marquee_phase == 'resizing':
            self._update_marquee_from_resize(event.pos())
            self._recompute_marquee_selection()
            self.update()
            return
        if self._marquee_phase == 'drawing' and self._marquee_start is not None:
            self._marquee_cur = event.pos()
            self.update()
            return
        if self.draw_mode and self._drag_start is not None:
            self._drag_cur = event.pos()
            self.update()
            return
        # Idle hover: update the cursor so the user sees a diagonal-resize
        # shape when over a marquee corner or a selected rect's handle.
        self._apply_hover_cursor(event.pos())

    def mouseReleaseEvent(self, event):
        if self._pan_drag_last is not None \
                and event.button() == QtCore.Qt.RightButton:
            moved = self._pan_drag_moved
            self._pan_drag_last  = None
            self._pan_drag_moved = False
            if not moved:
                self._right_click_action(event.pos())
            return
        if self.semiauto_edit_mode and self._semiauto_drag_pt is not None \
                and event.button() == QtCore.Qt.LeftButton:
            self._semiauto_drag_pt = None
            self.mask_point_added.emit()   # re-run SAM with the moved point
            self.update()
            return
        if self.semiauto_edit_mode and self._vertex_drag_idx is not None \
                and event.button() == QtCore.Qt.LeftButton:
            self._vertex_drag_idx = None   # vertex edits are already live in ann['data']
            self.update()
            return
        if self._resize_handle is not None and event.button() == QtCore.Qt.LeftButton:
            self._resize_handle  = None
            self._resize_ann_idx = None
            self._apply_hover_cursor(event.pos())
            self.update()
            self.boxes_changed.emit()
            return
        if self._marquee_phase == 'resizing' and event.button() == QtCore.Qt.LeftButton:
            self._marquee_handle = None
            self._marquee_phase  = None
            self._apply_hover_cursor(event.pos())
            self.update()
            return
        if self._marquee_phase == 'drawing' and event.button() == QtCore.Qt.LeftButton:
            mx1 = min(self._marquee_start.x(), event.pos().x())
            my1 = min(self._marquee_start.y(), event.pos().y())
            mx2 = max(self._marquee_start.x(), event.pos().x())
            my2 = max(self._marquee_start.y(), event.pos().y())
            self._marquee_start = None
            self._marquee_cur   = None
            self._marquee_phase = None
            # Tiny rects = click without drag, so leave selection cleared
            # (we already cleared in mousePressEvent for this path).
            if mx2 - mx1 >= 4 and my2 - my1 >= 4:
                img_box = self._widget_xyxy_to_image(mx1, my1, mx2, my2)
                if img_box is not None:
                    self._persistent_marquee = list(img_box)
                    self._recompute_marquee_selection()
            self.update()
            return
        if self.draw_mode and event.button() == QtCore.Qt.LeftButton and self._drag_start:
            x1 = min(self._drag_start.x(), event.pos().x())
            y1 = min(self._drag_start.y(), event.pos().y())
            x2 = max(self._drag_start.x(), event.pos().x())
            y2 = max(self._drag_start.y(), event.pos().y())
            added = False
            # Reject drags entirely in the letterbox area + tiny accidental clicks.
            if x2 - x1 > 8 and y2 - y1 > 8 and self._rect_intersects_image(x1, y1, x2, y2):
                cx1, cy1, cx2, cy2 = self._clip_rect_to_image(x1, y1, x2, y2)
                # Convert clipped widget coords -> absolute image coords. We
                # store boxes in image space (not widget space) so they don't
                # drift onto the gray letterbox when the window is resized.
                img_box = self._widget_xyxy_to_image(cx1, cy1, cx2, cy2)
                if img_box is not None:
                    # Store drag directly in self.annotations as a manual
                    # rect so it's editable (resize/delete) the moment the
                    # user toggles Edit Boxes, no separate bucket.
                    ix1, iy1, ix2, iy2 = img_box
                    cxn = (ix1 + ix2) / 2 / self._orig_w
                    cyn = (iy1 + iy2) / 2 / self._orig_h
                    wn  = (ix2 - ix1) / self._orig_w
                    hn  = (iy2 - iy1) / self._orig_h
                    _subj = getattr(self, 'draw_subject', 'annotation')
                    _src = _subj if _subj in ('prompt', 'neg_prompt') else 'manual'
                    self.annotations.append({
                        'type': 'rect',
                        'data': [cxn, cyn, wn, hn],
                        'deleted': False,
                        'source': _src,
                        'cls': int(getattr(self, 'active_draw_cls', 0) or 0),
                    })
                    added = True
            self._drag_start = None
            self._drag_cur   = None
            self.update()
            if added:
                self.boxes_changed.emit()

    def _image_area_widget_rect(self):
        """Return (x1, y1, x2, y2) of the displayed image inside the widget."""
        scale, off_x, off_y = self._get_scale_offset()
        return (off_x, off_y,
                off_x + self._orig_w * scale,
                off_y + self._orig_h * scale)

    def _rect_intersects_image(self, x1, y1, x2, y2):
        if not self._orig_w:
            return False
        ix1, iy1, ix2, iy2 = self._image_area_widget_rect()
        return not (x2 <= ix1 or y2 <= iy1 or x1 >= ix2 or y1 >= iy2)

    def _clip_rect_to_image(self, x1, y1, x2, y2):
        if not self._orig_w:
            return x1, y1, x2, y2
        ix1, iy1, ix2, iy2 = self._image_area_widget_rect()
        return (max(x1, ix1), max(y1, iy1), min(x2, ix2), min(y2, iy2))

    def keyPressEvent(self, event):
        # SAM mask drawing intercepts keys while active. Esc cancels;
        # Backspace/Delete drops the last point. Enter: semi-auto CLOSES the
        # outline (>=3 points) to segment + commit; autodraw commits the live
        # single-point preview. Falls through to normal handling otherwise.
        # Not gated on _resize_mode: drawing and editing stay live while zoomed
        # (see set_resize_mode), so the keys that commit or cancel that drawing
        # have to stay live with it, or a zoomed-in outline can be started and
        # then neither closed nor escaped.
        if self.mask_draw_mode:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if self._mask_draw_kind == "semiauto":
                    if len(self._mask_points) >= 3:
                        self.mask_close_requested.emit()
                else:
                    self.mask_commit_requested.emit()
                return
            if event.key() == QtCore.Qt.Key_Escape and self._mask_points:
                self.clear_mask_session()
                return
            if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace) \
                    and self._mask_points:
                self._mask_points.pop()
                if self._mask_draw_kind == "autodraw":
                    self.mask_point_added.emit()   # refresh live preview
                self.update()
                return
        # Edit-Semi-Auto-Segments: Enter applies the edit to the selected mask;
        # Esc deselects (reverting any vertex edits); S opens per-mask settings;
        # Backspace drops the last SAM point and re-runs (points target only).
        if self.semiauto_edit_mode and self._semiauto_sel_idx is not None:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.semiauto_apply_requested.emit()
                return
            if event.key() == QtCore.Qt.Key_S:
                self.semiauto_settings_requested.emit()
                return
            if event.key() == QtCore.Qt.Key_Escape:
                # Revert in-place vertex edits before deselecting.
                if self._semiauto_orig_data is not None \
                        and self._semiauto_sel_idx < len(self.annotations):
                    self.annotations[self._semiauto_sel_idx]['data'] = \
                        [list(p) for p in self._semiauto_orig_data]
                self.clear_semiauto_selection()
                return
            if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                # Points target with points placed: drop the last point.
                if self._semiauto_edit_target == "points" and self._mask_points:
                    self._mask_points.pop()
                    self.mask_point_added.emit()
                    self.update()
                    return
                # Otherwise Delete removes the whole selected mask.
                self.semiauto_delete_requested.emit()
                return
        if event.key() == QtCore.Qt.Key_Escape:
            if (self.selected_index is not None
                    or self.selected_indices
                    or self._persistent_marquee is not None):
                self.selected_index = None
                self.selected_indices = set()
                self._persistent_marquee = None
                self._marquee_handle = None
                self._marquee_phase  = None
                self.update()
                return
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            if self._delete_selected_ann():
                return
            # No selection: soft-delete the most recently added manual rect.
            for ann in reversed(self.annotations):
                if (ann.get('source') == 'manual' and ann['type'] == 'rect'
                        and not ann['deleted']):
                    ann['deleted'] = True
                    self.update()
                    self.boxes_changed.emit()
                    return
        super().keyPressEvent(event)

    # painting
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        pix = self._clean_pixmap if self.edit_mode else (self._baked_pixmap or self._clean_pixmap)
        if pix is not None:
            # Route the image blit through the same scale/offset as every
            # overlay so zoom/pan move the picture and the annotations together.
            scale, off_x, off_y = self._get_scale_offset()
            painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            target = QtCore.QRectF(off_x, off_y,
                                   self._orig_w * scale, self._orig_h * scale)
            painter.drawPixmap(target, pix, QtCore.QRectF(pix.rect()))
            # Darken Tint (view-only, Roboflow-style): dim the whole image but
            # keep every detection: boxes, model masks, manual + semi-auto
            # segments, kept bright by punching them out of the overlay. The
            # annotation outlines/handles drawn below still paint on top.
            if self._dark_tint and self._orig_w and self._orig_h:
                tint_path = QtGui.QPainterPath()
                tint_path.addRect(target)
                holes = QtGui.QPainterPath()
                for ann in self.get_active_annotations():
                    if ann['type'] == 'poly':
                        pts = self._to_label(ann['data'])
                        if len(pts) >= 3:
                            holes.addPolygon(QtGui.QPolygonF(
                                [QtCore.QPointF(px, py) for px, py in pts]))
                    elif ann['type'] == 'rect':
                        rx1, ry1, rx2, ry2 = self._rect_to_label(*ann['data'])
                        holes.addRect(QtCore.QRectF(rx1, ry1,
                                                    rx2 - rx1, ry2 - ry1))
                if not holes.isEmpty():
                    tint_path = tint_path.subtracted(holes)
                painter.save()
                painter.setClipRect(target)
                painter.fillPath(tint_path, QtGui.QColor(0, 0, 0, 140))
                painter.restore()
        # Drawn boxes: manual rect anns rendered yellow outside edit mode.
        # Inside edit mode, the annotation paint pass below renders them
        # with source-aware coloring (green=manual, magenta=detector,
        # cyan=selected) plus resize handles and the delete-X badge.
        if not self.edit_mode:
            painter.setBrush(QtCore.Qt.NoBrush)
            # Positive prompt boxes: dashed, colored by their class so multi-class
            # box prompts are distinguishable at a glance. The class count is
            # capped by the palette (style.MAX_BOX_CLASSES), not by this widget.
            pboxes, pcls = self.get_prompt_boxes_with_cls_in_image_coords()
            for box, c in zip(pboxes, pcls):
                wb = self._image_xyxy_to_widget(*box)
                if wb is None:
                    continue
                wx1, wy1, wx2, wy2 = wb
                painter.setPen(QtGui.QPen(class_color_qt(c), 2, QtCore.Qt.DashLine))
                painter.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))
            # Negative prompt boxes: dashed red (one type, suppresses matches).
            nboxes = self.get_neg_prompt_boxes_in_image_coords()
            if nboxes:
                painter.setPen(QtGui.QPen(QtGui.QColor(200, 60, 60), 2, QtCore.Qt.DashLine))
                for box in nboxes:
                    wb = self._image_xyxy_to_widget(*box)
                    if wb is None:
                        continue
                    wx1, wy1, wx2, wy2 = wb
                    painter.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))
        if self.draw_mode and self._drag_start and self._drag_cur:
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2, QtCore.Qt.DashLine))
            painter.setBrush(QtCore.Qt.NoBrush)
            x1 = min(self._drag_start.x(), self._drag_cur.x())
            y1 = min(self._drag_start.y(), self._drag_cur.y())
            x2 = max(self._drag_start.x(), self._drag_cur.x())
            y2 = max(self._drag_start.y(), self._drag_cur.y())
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
        # Drawing-phase marquee (live drag).
        if (self._marquee_phase == 'drawing' and self._marquee_start
                and self._marquee_cur):
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 1, QtCore.Qt.DashLine))
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 220, 255, 40)))
            mx1 = min(self._marquee_start.x(), self._marquee_cur.x())
            my1 = min(self._marquee_start.y(), self._marquee_cur.y())
            mx2 = max(self._marquee_start.x(), self._marquee_cur.x())
            my2 = max(self._marquee_start.y(), self._marquee_cur.y())
            painter.drawRect(mx1, my1, mx2 - mx1, my2 - my1)
        # Persistent marquee with corner resize handles.
        elif (self.multi_select_mode and self._persistent_marquee is not None):
            mwidget = self._image_xyxy_to_widget(*self._persistent_marquee)
            if mwidget is not None:
                wx1, wy1, wx2, wy2 = mwidget
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 1, QtCore.Qt.DashLine))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 220, 255, 30)))
                painter.drawRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
                r = self._HANDLE_R
                for hx, hy in ((wx1, wy1), (wx2, wy1), (wx1, wy2), (wx2, wy2)):
                    painter.drawRect(int(hx) - r, int(hy) - r, 2 * r, 2 * r)
        if self.edit_mode and self.annotations:
            r = self._X_R
            for idx, ann in enumerate(self.annotations):
                if ann['deleted']:
                    continue
                is_selected = (idx == self.selected_index
                               or idx in self.selected_indices)
                is_primary  = (idx == self.selected_index)
                # Selected ann = cyan; manual draws = green; detector output =
                # its class color (class 0 = the historical magenta). Color
                # encodes provenance so the user can tell at a glance which
                # boxes the model produced vs. which they added by hand; with
                # multi-class prompts each extra class gets its own hue.
                if is_selected:
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 3))
                elif ann.get('source') == 'neg_prompt':
                    painter.setPen(QtGui.QPen(QtGui.QColor(200, 60, 60), 2, QtCore.Qt.DashLine))
                elif ann.get('source') == 'prompt':
                    painter.setPen(QtGui.QPen(class_color_qt(ann.get('cls', 0)), 2, QtCore.Qt.DashLine))
                elif ann.get('source') == 'manual':
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 100), 2))
                else:
                    painter.setPen(QtGui.QPen(class_color_qt(ann.get('cls', 0)), 2))
                painter.setBrush(QtCore.Qt.NoBrush)
                if ann['type'] == 'poly':
                    pts = self._to_label(ann['data'])
                    if len(pts) < 3:
                        continue
                    painter.drawPolygon(
                        QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in pts])
                    )
                elif ann['type'] == 'rect':
                    x1, y1, x2, y2 = self._rect_to_label(*ann['data'])
                    painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                # Corner resize handles, RECT ONLY. Polygons stay
                # delete-only (per supervisor: only rects are resizable).
                if is_primary and ann['type'] == 'rect' and not self.multi_select_mode:
                    centers = self._rect_handle_centers(idx)
                    if centers is not None:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
                        painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
                        r = self._HANDLE_R
                        for (hx, hy) in centers.values():
                            painter.drawRect(int(hx) - r, int(hy) - r, 2 * r, 2 * r)
                # Render the red X delete badge on EVERY selected ann.
                # Each X is independently clickable in mousePressEvent so
                # the user can prune the selection one item at a time.
                if is_selected:
                    cx, cy = self._x_center_for_ann(ann)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 30, 30)))
                    painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
                    painter.drawEllipse(QtCore.QPoint(cx, cy), r, r)
                    painter.drawLine(cx - 6, cy - 6, cx + 6, cy + 6)
                    painter.drawLine(cx + 6, cy - 6, cx - 6, cy + 6)
        # Semi-auto SAM mask drawing / editing: live preview polygon (cyan) +
        # the prompt points (green=foreground, red=background). Shared by the
        # draw mode and the Edit-Semi-Auto-Segments mode.
        if (self.mask_draw_mode or getattr(self, "semiauto_edit_mode", False)) \
                and self._orig_w:
            edit_sel = (self.semiauto_edit_mode and self._semiauto_sel_idx is not None)
            vtx_mode = (edit_sel and self._semiauto_edit_target == "vertices")
            drawing_semiauto = (self.mask_draw_mode and self._mask_draw_kind == "semiauto")
            scale, off_x, off_y = self._get_scale_offset()
            # Red X delete badge on the selected mask (click or Del removes it).
            if edit_sel and self._semiauto_sel_idx < len(self.annotations):
                _asel = self.annotations[self._semiauto_sel_idx]
                if not _asel['deleted']:
                    bx, by = self._x_center_for_ann(_asel)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 30, 30)))
                    painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
                    painter.drawEllipse(QtCore.QPoint(bx, by), 9, 9)
                    painter.drawLine(bx - 5, by - 5, bx + 5, by + 5)
                    painter.drawLine(bx + 5, by - 5, bx - 5, by + 5)
            # Ghost "+" on the outline marks where a click adds a vertex.
            if vtx_mode and self._vertex_ghost is not None:
                gx, gy = self._vertex_ghost.x(), self._vertex_ghost.y()
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 220, 255)))
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
                painter.drawEllipse(QtCore.QPoint(gx, gy), 7, 7)
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 2))
                painter.drawLine(gx - 4, gy, gx + 4, gy)
                painter.drawLine(gx, gy - 4, gx, gy + 4)
            # SAM mask preview (cyan). Shown for autodraw (live) and the edit
            # mode. Semi-auto DRAWING shows NO mask until the outline closes;
            # closing segments + commits in one step.
            if self._mask_preview_poly and len(self._mask_preview_poly) >= 3 \
                    and not drawing_semiauto:
                pts = self._to_label(self._mask_preview_poly)
                if len(pts) >= 3:
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 220, 255, 50)))
                    painter.drawPolygon(
                        QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in pts])
                    )
                    if vtx_mode:
                        # Square draggable handles on every polygon vertex.
                        painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
                        painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
                        for x, y in pts:
                            painter.drawRect(int(x) - 4, int(y) - 4, 8, 8)
            # Extra SAM preview blobs (autodraw multi-object), same cyan fill,
            # NO connector between them (each commits as its own separate mask).
            if not drawing_semiauto and self._mask_preview_extra:
                painter.setPen(QtGui.QPen(QtGui.QColor(0, 220, 255), 2))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 220, 255, 50)))
                for epoly in self._mask_preview_extra:
                    epts = self._to_label(epoly)
                    if len(epts) >= 3:
                        painter.drawPolygon(
                            QtGui.QPolygon([QtCore.QPoint(int(x), int(y)) for x, y in epts])
                        )
            if not vtx_mode:
                wpts = [(int(ix * scale + off_x), int(iy * scale + off_y))
                        for ix, iy, _lab in self._mask_points]
                wlabs = [m[2] for m in self._mask_points]
                painter.setBrush(QtCore.Qt.NoBrush)
                if drawing_semiauto:
                    # OPEN connected outline (solid) + rubber-band line to the
                    # cursor. The curve grows as you click; it only fills once
                    # you close it (click the first point again / double-click).
                    if len(wpts) >= 2:
                        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 100), 2))
                        painter.drawPolyline(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in wpts]))
                    if wpts and self._mask_cursor is not None:
                        painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 100), 1, QtCore.Qt.DashLine))
                        painter.drawLine(wpts[-1][0], wpts[-1][1],
                                         self._mask_cursor.x(), self._mask_cursor.y())
                elif len(wpts) >= 2 and not (self.mask_draw_mode
                                             and self._mask_draw_kind == "autodraw"):
                    # SAM-point edit: closed dashed interlink around the points.
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 200, 100), 1, QtCore.Qt.DashLine))
                    painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(x, y) for x, y in wpts]))
                # Foreground prompt points. In semi-auto DRAWING the very first
                # point is the "close anchor", drawn AMBER (distinct from the
                # green others) so the user knows clicking it links the outline
                # closed; the rest are green.
                painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
                for i, (x, y) in enumerate(wpts):
                    if drawing_semiauto and i == 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 180, 0)))  # amber anchor
                        painter.drawEllipse(QtCore.QPoint(x, y), 6, 6)
                    elif i < len(wlabs) and wlabs[i] == 0:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(220, 40, 40)))   # red = negative
                        painter.drawEllipse(QtCore.QPoint(x, y), 5, 5)
                    else:
                        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 200, 100)))   # green = positive
                        painter.drawEllipse(QtCore.QPoint(x, y), 5, 5)
                # Once the outline can actually be closed (>=3 points), ring the
                # amber anchor to reinforce "click here to close".
                if drawing_semiauto and len(wpts) >= 3:
                    painter.setPen(QtGui.QPen(QtGui.QColor(255, 180, 0), 2))
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(QtCore.QPoint(*wpts[0]), 10, 10)
        painter.end()

    @staticmethod
    def _hit_box(point, box, tol=6):
        x1, y1, x2, y2 = box
        return x1 - tol < point.x() < x2 + tol and y1 - tol < point.y() < y2 + tol

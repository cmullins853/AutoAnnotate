"""A plain image view with the same zoom / pan behavior as the annotation canvas.

AnnotationCanvas owns drawing, editing, hit-testing and a dozen annotation
buckets; the side-by-side comparer needs none of that, only the view transform.
This widget carries just that transform, so a zoom gesture feels identical in
both screens without the comparer inheriting the annotation machinery.
"""
from PyQt5 import QtWidgets, QtGui, QtCore

from . import session_state

# Same limits and feel as AnnotationCanvas: fit-to-window is 1.0, you cannot
# zoom out past it, and 8x is as close as it goes.
MIN_ZOOM = 1.0
MAX_ZOOM = 8.0
_WHEEL_STEP = 1.0015          # per wheel unit, direction-aware


class ZoomPanImageView(QtWidgets.QLabel):
    """Shows one pixmap, fit to the widget, with scroll/pinch zoom and drag pan.

    Zoom and pan PERSIST until reset_view(), matching the Image Resize toggle in
    the annotation window: turning the mode off leaves you where you were.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._resize_mode = False
        self._pan_last = None
        self.setAlignment(QtCore.Qt.AlignCenter)

    # -- content ----------------------------------------------------------
    def set_pixmap(self, pixmap, keep_view=False):
        """Show `pixmap`. Resets the view unless keep_view, which is what the
        Prev/Next pair walk wants: the user zoomed in to inspect a detail and
        expects the next pair to stay at that zoom."""
        self._pixmap = pixmap
        if self.has_image():
            self.setText("")   # placeholder must not show through the image
        if not keep_view:
            self.reset_view()
        else:
            self._clamp_view()
        self.update()

    def set_placeholder(self, text):
        """Text shown when there is no pixmap. paintEvent falls back to QLabel's
        own text rendering in that case."""
        self.setText(text)

    def has_image(self):
        return self._pixmap is not None and not self._pixmap.isNull()

    # -- view state -------------------------------------------------------
    def set_resize_mode(self, enabled):
        self._resize_mode = bool(enabled)
        self._pan_last = None
        self.setCursor(QtCore.Qt.OpenHandCursor if enabled else QtCore.Qt.ArrowCursor)

    def reset_view(self):
        """Back to fit-to-window."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def view_state(self):
        """The full view configuration, for carrying across a side swap."""
        return {"zoom": self._zoom, "pan_x": self._pan_x, "pan_y": self._pan_y,
                "resize_mode": self._resize_mode}

    def apply_view_state(self, state):
        """Restore a dict from view_state(). Unknown/missing keys keep the
        current value, so an older saved state (e.g. one that still carries a
        dark_tint flag from the removed tint feature) cannot break the widget."""
        if not state:
            return
        self._zoom = float(state.get("zoom", self._zoom))
        self._pan_x = float(state.get("pan_x", self._pan_x))
        self._pan_y = float(state.get("pan_y", self._pan_y))
        self.set_resize_mode(state.get("resize_mode", self._resize_mode))
        self._clamp_view()
        self.update()

    # -- transform --------------------------------------------------------
    def _image_size(self):
        if not self.has_image():
            return 0, 0
        return self._pixmap.width(), self._pixmap.height()

    def _get_scale_offset(self):
        iw, ih = self._image_size()
        if not iw or not ih:
            return 1.0, 0.0, 0.0
        lw, lh = self.width(), self.height()
        base = min(lw / iw, lh / ih)       # fit-to-window
        scale = base * self._zoom
        off_x = (lw - iw * scale) / 2 + self._pan_x
        off_y = (lh - ih * scale) / 2 + self._pan_y
        return scale, off_x, off_y

    def _clamp_view(self):
        """Keep zoom in range and stop the image being panned off-screen."""
        self._zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom))
        iw, ih = self._image_size()
        if self._zoom <= MIN_ZOOM or not iw or not ih:
            self._pan_x = 0.0
            self._pan_y = 0.0
            return
        lw, lh = self.width(), self.height()
        scale = min(lw / iw, lh / ih) * self._zoom
        max_x = max(0.0, (iw * scale - lw) / 2)
        max_y = max(0.0, (ih * scale - lh) / 2)
        self._pan_x = max(-max_x, min(max_x, self._pan_x))
        self._pan_y = max(-max_y, min(max_y, self._pan_y))

    def _zoom_at(self, factor, wx, wy):
        """Multiply zoom by `factor`, keeping the image point under widget pixel
        (wx, wy) fixed, so the image grows toward the cursor."""
        iw, ih = self._image_size()
        if not iw or not ih:
            return
        scale, off_x, off_y = self._get_scale_offset()
        ix = (wx - off_x) / scale
        iy = (wy - off_y) / scale
        self._zoom *= factor
        self._clamp_view()
        lw, lh = self.width(), self.height()
        nscale = min(lw / iw, lh / ih) * self._zoom
        self._pan_x = (wx - ix * nscale) - (lw - iw * nscale) / 2
        self._pan_y = (wy - iy * nscale) - (lh - ih * nscale) / 2
        self._clamp_view()
        self.update()

    # -- input ------------------------------------------------------------
    def wheelEvent(self, event):
        if not self._resize_mode or not self.has_image():
            super().wheelEvent(event)
            return
        # Same wheel scheme as the annotation canvas: scroll pans, Ctrl/Cmd +
        # scroll zooms (pinch also zooms, in event()). Left-drag still pans
        # here too, since this view has no drawing to reserve the button for.
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
            self._zoom_at(_WHEEL_STEP ** action[1], pos.x(), pos.y())
        else:
            self._pan_x += action[1]
            self._pan_y += action[2]
            self._clamp_view()
            self.update()
        event.accept()

    def event(self, e):
        # macOS trackpad pinch arrives as a native gesture, not a wheel.
        if (self._resize_mode and self.has_image()
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

    def mousePressEvent(self, event):
        if (self._resize_mode and self.has_image()
                and event.button() == QtCore.Qt.LeftButton):
            self._pan_last = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resize_mode and self._pan_last is not None:
            d = event.pos() - self._pan_last
            self._pan_x += d.x()
            self._pan_y += d.y()
            self._pan_last = event.pos()
            self._clamp_view()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resize_mode and self._pan_last is not None:
            self._pan_last = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):
        # A zoomed, panned image must not get stranded off-screen when the
        # window changes size. The fit scale is recomputed every paint; only
        # the pan offset needs re-clamping.
        self._clamp_view()
        super().resizeEvent(event)

    # -- paint ------------------------------------------------------------
    def paintEvent(self, event):
        # QLabel first: it paints the stylesheet background/border, and the
        # placeholder text when there is nothing to show. The image goes on top.
        super().paintEvent(event)
        if not self.has_image():
            return
        painter = QtGui.QPainter(self)
        iw, ih = self._image_size()
        scale, off_x, off_y = self._get_scale_offset()
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        # Clip to the content area so a zoomed image cannot paint over the
        # panel's own border, which is what tells the two panes apart.
        painter.setClipRect(self.contentsRect())
        target = QtCore.QRectF(off_x, off_y, iw * scale, ih * scale)
        painter.drawPixmap(target, self._pixmap, QtCore.QRectF(self._pixmap.rect()))

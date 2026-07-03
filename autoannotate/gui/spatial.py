"""Uniform spatial hash grid for fast bbox/point hit-testing on the canvas."""
class SpatialGrid:
    """Uniform grid over the normalized [0,1]x[0,1] image space for fast
    bbox-overlap / point candidate lookups. Turns "compare against every mask"
    (O(n) per query, O(n^2) overall) into "compare against the handful of masks
    in the same neighbourhood". Pure-python, no deps.

    Items are inserted with a normalized xyxy bbox + a payload (any object).
    `query_bbox` / `query_point` return the payloads whose cells overlap the
    query, de-duplicated by identity. False positives (cell overlap without true
    geometry overlap) are fine; the caller still runs the exact test on the
    returned candidates; the grid only prunes the far-away ones."""

    __slots__ = ("cells", "buckets")

    def __init__(self, cells=32):
        self.cells = cells
        self.buckets = {}

    def _cell_range(self, x1, y1, x2, y2):
        n = self.cells
        cx1 = min(n - 1, max(0, int(x1 * n))); cx2 = min(n - 1, max(0, int(x2 * n)))
        cy1 = min(n - 1, max(0, int(y1 * n))); cy2 = min(n - 1, max(0, int(y2 * n)))
        if cx2 < cx1: cx1, cx2 = cx2, cx1
        if cy2 < cy1: cy1, cy2 = cy2, cy1
        return cx1, cy1, cx2, cy2

    def insert(self, bbox, payload):
        cx1, cy1, cx2, cy2 = self._cell_range(*bbox)
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                self.buckets.setdefault((cx, cy), []).append(payload)

    def query_bbox(self, bbox):
        cx1, cy1, cx2, cy2 = self._cell_range(*bbox)
        seen = set(); out = []
        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                for p in self.buckets.get((cx, cy), ()):
                    if id(p) not in seen:
                        seen.add(id(p)); out.append(p)
        return out

    def query_point(self, x, y):
        return self.query_bbox((x, y, x, y))

    @classmethod
    def build(cls, items, bbox_of, cells=32):
        """items -> grid, using bbox_of(item) for each item's normalized bbox."""
        g = cls(cells)
        for it in items:
            g.insert(bbox_of(it), it)
        return g

"""Per-class annotation colors, shared by the Qt canvas and the cv2 overlays.

Lives outside both `gui` and `pipeline` because both need it: `gui.style` wraps
these in QColor for the canvas, `pipeline.overlay` uses the BGR form for images
baked with cv2. Keeping the numbers here means a class is the same color in the
app and in the saved review images, and it keeps PyQt out of the pipeline.
"""

# Class 0 keeps the historical detector magenta so single-class sessions look
# exactly as they always have; classes 1+ cycle through hues chosen to stay
# clear of the provenance colors: gold (prompt boxes), green (manual work),
# cyan (selected), red (delete / negative).
#
# The canvas and the cv2 overlays have always used slightly different magentas
# for class 0 -- QColor(200,0,200) vs BGR (255,0,255). Both are kept verbatim
# rather than unified, so this refactor cannot change a single saved pixel of a
# single-class run.
CLASS_0_RGB = (200, 0, 200)         # canvas (Qt)
CLASS_0_BGR = (255, 0, 255)         # baked cv2 overlays
CLASS_0_NAME = "magenta"

CLASS_RGB = [
    (255, 140, 0),    # class 1
    (30, 144, 255),   # class 2
    (170, 90, 255),   # class 3
    (0, 170, 155),    # class 4
    (255, 105, 180),  # class 5
    (200, 200, 90),   # class 6
    (150, 220, 255),  # class 7
]

# Plain-English name per CLASS_RGB entry, in the same order. Data rather than a
# trailing comment, so class_colors.txt cannot drift from the actual hues.
CLASS_COLOR_NAMES = [
    "orange", "blue", "violet", "teal", "pink", "khaki", "light blue",
]
assert len(CLASS_COLOR_NAMES) == len(CLASS_RGB)

# How many box-prompt classes the Box Classes dialog offers: ids 0..4.
#
# This is a DELIBERATE product limit, not the palette's limit. The palette holds
# more hues than this, and each extra class costs one more SAM3 pass per image,
# so the cap is set to the number of classes a run is actually expected to need.
# Raising it means raising this number and nothing else, as long as it stays
# within len(CLASS_RGB) + 1 or two classes start sharing a color.
MAX_BOX_CLASSES = 5
assert MAX_BOX_CLASSES <= len(CLASS_RGB) + 1, "not enough distinct class colors"

# Provenance colors that are not per-class. Exposed so the legend image and the
# canvas agree on what green and red mean.
MANUAL_RGB = (0, 200, 100)
NEGATIVE_RGB = (200, 60, 60)


def class_color_name(idx):
    """Plain-English color name for a class index, matching class_color_rgb."""
    idx = int(idx)
    if idx <= 0:
        return CLASS_0_NAME
    return CLASS_COLOR_NAMES[(idx - 1) % len(CLASS_COLOR_NAMES)]


def rgb_to_hex(rgb):
    """(r, g, b) -> '#RRGGBB', uppercase."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def class_color_rgb(idx):
    """(r, g, b) for a class index. Wraps around past the end of the palette."""
    idx = int(idx)
    if idx <= 0:
        return CLASS_0_RGB
    return CLASS_RGB[(idx - 1) % len(CLASS_RGB)]


def class_color_bgr(idx):
    """BGR tuple for a class index (cv2 overlays)."""
    idx = int(idx)
    if idx <= 0:
        return CLASS_0_BGR
    r, g, b = class_color_rgb(idx)
    return (b, g, r)


def class_color_image_rgb(idx):
    """(r, g, b) as the class actually appears in a SAVED review image.

    Identical to class_color_rgb for every class but 0, which the cv2 overlays
    draw in a slightly brighter magenta than the canvas. class_colors.txt and
    the legend image both describe files on disk, so they must quote this."""
    b, g, r = class_color_bgr(idx)
    return (r, g, b)

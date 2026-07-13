"""Drawing helpers: boxes and mask borders baked onto cv2 images."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..imageio import imwrite_unicode
from ..palette import class_color_bgr, MANUAL_RGB, NEGATIVE_RGB

def draw_boxes_on_image(image, boxes, colors=None):
    """Draw boxes on the image. `colors[i]` overrides the default magenta for
    box i (used to flag manual draws in green so they're visually distinct
    from detector output in the baked overlay). Drawing happens through PIL
    on an RGB canvas, so entries in `colors` are RGB tuples, not cv2 BGR;
    class hues come from palette.class_color_image_rgb."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    img_width, img_height = pil_image.size
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        if x2 > x1 and y2 > y1:
            color = colors[i] if (colors is not None and i < len(colors)) else (255, 0, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def adjust_masks(sam_results):
    if not sam_results or getattr(sam_results[0], "masks", None) is None:
        return []
    result = sam_results[0]

    masks = result.masks.data.cpu().numpy()     # masks, (N, H, W)
    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    return masks

def overlay_with_borders(image, mask, color, thickness=2):
    # Convert mask to uint8 type
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image, contours, -1, color, thickness)
    return image


# Legend geometry, in pixels.
_LEG_PAD = 16
_LEG_ROW = 34
_LEG_SWATCH = 24
_LEG_TEXT_X = _LEG_PAD + _LEG_SWATCH + 12
_LEG_WIDTH = 420


def save_class_legend_image(names, out_path):
    """Write a standalone PNG keying each class id to the color it is drawn in.

    Saved NEXT TO the annotated boxes/ and masks/ folders rather than inside
    them: a reviewer opening the folder later can read the color key, but no
    pixel of a labelled review image is ever painted over, so the overlays stay
    faithful to what the model actually produced.

    `names` is the class list in id order (index == class id), the same list
    written to classes.txt. Returns the path written, or None on failure.
    """
    names = list(names or [])
    if not names:
        return None
    rows = [(i, f"{i}: {name}", class_color_bgr(i)) for i, name in enumerate(names)]
    # The two provenance colors that are not classes. Users see them on the
    # canvas, so the key would be misleading without them.
    r, g, b = MANUAL_RGB
    rows.append((None, "manual annotation (drawn by hand)", (b, g, r)))
    r, g, b = NEGATIVE_RGB
    rows.append((None, "negative box (found, then suppressed)", (b, g, r)))

    height = _LEG_PAD * 2 + _LEG_ROW * (len(rows) + 1)
    img = np.full((height, _LEG_WIDTH, 3), 32, dtype=np.uint8)
    cv2.putText(img, "Class colors", (_LEG_PAD, _LEG_PAD + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    y = _LEG_PAD + _LEG_ROW
    for _cls, label, color in rows:
        cv2.rectangle(img, (_LEG_PAD, y + 4), (_LEG_PAD + _LEG_SWATCH, y + 4 + _LEG_SWATCH),
                      color, -1)
        cv2.rectangle(img, (_LEG_PAD, y + 4), (_LEG_PAD + _LEG_SWATCH, y + 4 + _LEG_SWATCH),
                      (255, 255, 255), 1)
        cv2.putText(img, label, (_LEG_TEXT_X, y + 4 + _LEG_SWATCH - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += _LEG_ROW

    if not imwrite_unicode(out_path, img):
        return None
    return out_path

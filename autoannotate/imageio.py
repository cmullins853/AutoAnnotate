"""Unicode-safe cv2 image reading and writing.

OpenCV's `imread` / `imwrite` pass the filename to the C runtime as a byte
string. On POSIX that is UTF-8 and any path works. On Windows they go through
the ANSI API, so a path containing a character outside the active code page
(`C:\\Users\\Bäiley\\...`, an accented output folder, a Windows temp dir under a
non-ASCII username) fails: imread returns None and imwrite returns False, both
silently.

Routing every read/write through `numpy.fromfile` / `tofile` sidesteps the C
runtime entirely: Python opens the file with the correct wide-character API and
OpenCV only ever sees an in-memory buffer.
"""
import os

import cv2
import numpy as np


def imread_unicode(path, flags=None):
    """cv2.imread that works with any path. Returns None on failure, like imread.

    `flags` defaults to cv2.IMREAD_COLOR, resolved at CALL time rather than as a
    default argument: the headless test suite stubs cv2 with an empty module, and
    a default of `cv2.IMREAD_COLOR` would be evaluated at import and blow up."""
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img if img is not None else None


def imwrite_unicode(path, img, params=None):
    """cv2.imwrite that works with any path. Returns True on success.

    Creates the parent directory when the path has one. Encoding is chosen from
    the file extension, exactly as imwrite does.
    """
    if img is None:
        return False
    parent = os.path.dirname(path)
    if parent:
        # dirname is '' for a bare filename, and os.makedirs('') raises.
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            return False
    ext = os.path.splitext(path)[1]
    if not ext:
        return False
    try:
        ok, buf = cv2.imencode(ext, img, params if params is not None else [])
        if not ok:
            return False
        buf.tofile(path)
    except Exception:
        # cv2.error cannot be named in the except clause: the headless suite
        # stubs cv2, and resolving the attribute would raise while handling.
        return False
    return True

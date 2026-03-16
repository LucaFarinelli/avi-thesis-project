import cv2
import numpy as np


def detect_a4_corners(image, min_area_ratio=0.05):
    """Rileva i 4 angoli del foglio A4 nell'immagine.

    Args:
        image: Immagine BGR in input.
        min_area_ratio: Frazione minima dell'area totale che il quadrilatero
                        deve occupare per essere considerato un foglio A4.
                        Serve a escludere oggetti piccoli (es. telefono, scarpa)
                        che potrebbero essere rilevati erroneamente come A4.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * min_area_ratio

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        # Salta contorni troppo piccoli per essere un foglio A4
        if cv2.contourArea(contour) < min_area:
            break

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            pts_ordered = order_points(pts)
            return pts_ordered.astype(np.float32)

    return None


def order_points(pts):
    """Ordina 4 punti in senso orario: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def warp_image_to_a4(image, pts_src):
    """Raddrizza l'immagine applicando una trasformazione prospettica verso il piano A4."""
    if pts_src is None:
        return image

    A4_WIDTH = 700
    A4_HEIGHT = 1000

    pts_dst = np.array([
        [0, 0],
        [A4_WIDTH, 0],
        [A4_WIDTH, A4_HEIGHT],
        [0, A4_HEIGHT]
    ], dtype=np.float32)

    h, status = cv2.findHomography(pts_src, pts_dst)

    if h is None:
        return image

    warped = cv2.warpPerspective(image, h, (A4_WIDTH, A4_HEIGHT))
    return warped


def ensure_standard_size(image, target_size=(700, 1000)):
    """Ridimensiona l'immagine alle dimensioni standard mantenendo il rapporto d'aspetto."""
    h, w = image.shape[:2]
    target_h, target_w = target_size

    if h == target_h and w == target_w:
        return image

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded
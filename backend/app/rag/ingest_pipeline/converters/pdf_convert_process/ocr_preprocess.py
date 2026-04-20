# [bolt-004 PDF OCR] Tesseract 向け前処理
from __future__ import annotations

import numpy as np
from PIL import Image

import cv2

# -----------------------------------------------------------------------------
# 役割: OCR 前に PIL 画像をグレー化・コントラスト強化・傾き補正・二値化し、認識精度を上げる。
# 流れ: RGB→グレー → _clahe → medianBlur → _deskew_gray → adaptiveThreshold → L モード画像。
# -----------------------------------------------------------------------------


def preprocess_for_tesseract(img: Image.Image) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError("PIL.Image.Image が必要です")
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    if arr.size == 0 or arr.ndim != 3:
        return img

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = _clahe(gray)
    gray = cv2.medianBlur(gray, 3)
    gray = _deskew_gray(gray)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(binary, mode="L")


# -----------------------------------------------------------------------------
# 補助関数（OpenCV）
# -----------------------------------------------------------------------------


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _deskew_gray(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 20:
        return gray
    rect = cv2.minAreaRect(coords)
    angle = float(rect[-1])
    if angle < -45:
        angle = 90.0 + angle
    elif angle > 45:
        angle = angle - 90.0
    if abs(angle) < 0.15:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        gray,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

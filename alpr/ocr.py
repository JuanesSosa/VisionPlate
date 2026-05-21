"""
ocr.py - OCR sincrono directo (sin hilos)
El hilo separado causaba que los resultados se perdieran.
"""

import cv2
import numpy as np
import pytesseract
import re
import logging
from typing import Optional

log = logging.getLogger("ALPR.ocr")


def extract_plate_by_color(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Recorta la region naranja/amarilla de la placa dentro del crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([3,  80,  80], dtype=np.uint8),
                       np.array([38, 255, 255], dtype=np.uint8))
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h_c, w_c  = crop_bgr.shape[:2]
    best, best_score = None, 0.0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < h_c * w_c * 0.08:
            continue
        ratio = w / max(h, 1)
        if not (1.5 < ratio < 5.5):
            continue
        score = w * h * min(ratio / 2.5, 1.0)
        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        return None
    x, y, w, h = best
    pad = 4
    return crop_bgr[max(0,y-pad):min(h_c,y+h+pad),
                    max(0,x-pad):min(w_c,x+w+pad)]


class Preprocessor:
    def __init__(self, cfg):
        self.clahe    = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP,
                                         tileGridSize=cfg.CLAHE_GRID)
        self.denoise_h = cfg.DENOISE_H
        self.morph_k   = cv2.getStructuringElement(cv2.MORPH_RECT, cfg.MORPH_KERNEL)
        self.target_h  = cfg.OCR_TARGET_H

    def process(self, crop: np.ndarray) -> Optional[np.ndarray]:
        region = extract_plate_by_color(crop)
        src    = region if region is not None else crop
        return self._bin(src)

    def _bin(self, src: np.ndarray) -> Optional[np.ndarray]:
        h, w = src.shape[:2]
        if h < 8 or w < 15:
            return None
        scale   = self.target_h / h
        resized = cv2.resize(src, (max(1, int(w*scale)), self.target_h),
                             interpolation=cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.divide(gray, cv2.GaussianBlur(gray,(51,51),0), scale=255)
        gray = self.clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=self.denoise_h)
        k    = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        gray = np.clip(cv2.filter2D(gray,-1,k),0,255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(gray,255,
                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_k)

    def invert(self, b): return cv2.bitwise_not(b)
    def upscale(self, b, s=2.0):
        h,w = b.shape[:2]
        return cv2.resize(b,(int(w*s),int(h*s)),interpolation=cv2.INTER_LANCZOS4)


class OCREngine:
    _PSM8 = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _PSM7 = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, cfg):
        self.min_conf = cfg.MIN_OCR_CONF
        self.patterns = [re.compile(p) for p in cfg.PLATE_PATTERNS]
        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_CMD
        log.info(f"Tesseract: {cfg.TESSERACT_CMD}")

    def read(self, binary: np.ndarray, preprocessor: Preprocessor) -> tuple:
        """Ejecuta OCR sincrono. Retorna (texto, confianza)."""
        inv  = preprocessor.invert(binary)
        up   = preprocessor.upscale(binary)
        uinv = preprocessor.invert(up)

        for img, cfg in [(binary, self._PSM8),
                         (inv,    self._PSM8),
                         (up,     self._PSM8),
                         (uinv,   self._PSM7)]:
            text, conf = self._run(img, cfg)
            log.debug(f"OCR intento '{text}' conf={conf:.0f}")
            if self.is_valid(text):
                log.info(f"OCR exito: '{text}' conf={conf:.0f}")
                return text, conf

        # Ultimo recurso: image_to_string rapido
        for img in (binary, inv, up):
            try:
                raw  = pytesseract.image_to_string(img, config=self._PSM8)
                text = self._clean(raw)
                if self.is_valid(text):
                    log.info(f"OCR string exito: '{text}'")
                    return text, 65.0
            except Exception:
                pass

        return "", 0.0

    # Alias para compatibilidad
    def read_plate(self, binary, preprocessor): return self.read(binary, preprocessor)
    def submit(self, *a, **k): pass
    def get_results(self): return []

    def _run(self, img, config):
        try:
            data  = pytesseract.image_to_data(img, config=config,
                        output_type=pytesseract.Output.DICT)
            words, confs = [], []
            for i, w in enumerate(data["text"]):
                c = int(data["conf"][i])
                if c >= self.min_conf and w.strip():
                    words.append(w.strip().upper())
                    confs.append(c)
            if not words:
                return "", 0.0
            return self._clean("".join(words)), sum(confs)/len(confs)
        except Exception as e:
            log.debug(f"Tesseract err: {e}")
            return "", 0.0

    def _clean(self, text):
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        if len(text) < 5:
            return text
        r = list(text)
        for i, ch in enumerate(r):
            if i < 3:
                r[i] = {"0":"O","1":"I","5":"S","8":"B"}.get(ch,ch)
            else:
                r[i] = {"O":"0","I":"1","S":"5","B":"8",
                         "G":"6","Z":"2","T":"1"}.get(ch,ch)
        return "".join(r)

    def is_valid(self, text):
        return bool(text) and any(p.match(text) for p in self.patterns)

    def infer_vehicle_type(self, plate):
        if re.match(r"^[A-Z]{3}[0-9]{2}$", plate): return "motorcycle"
        if re.match(r"^CD", plate): return "diplomatic"
        return "car"

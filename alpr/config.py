"""
config.py - Configuracion ALPR - Universidad de Ibague
"""

import os
import platform


def _detect_tesseract() -> str:
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0]
    return "tesseract"


class Config:
    # -- Camara ---------------------------------------------------------------
    VIDEO_SOURCE   = 0        # 0 = camara del portatil
    FRAME_WIDTH    = 1280
    FRAME_HEIGHT   = 720
    TARGET_FPS     = 25

    # -- Modelo YOLO ----------------------------------------------------------
    YOLO_MODEL_PATH = "yolov8n.pt"
    YOLO_CONF       = 0.40
    YOLO_IOU        = 0.45
    YOLO_CLASSES    = None

    # -- OCR ------------------------------------------------------------------
    MIN_OCR_CONF = 35

    # -- Formatos placas colombianas ------------------------------------------
    PLATE_PATTERNS = [
        r'^[A-Z]{3}[0-9]{3}$',   # carro: ABC123
        r'^[A-Z]{3}[0-9]{2}$',   # moto:  ABC12
        r'^CD[0-9]{3,4}$',       # diplomatico
    ]

    # -- Tracking -------------------------------------------------------------
    TRACK_COOLDOWN_SEC = 60   # 1 min entre ENTRADA y SALIDA (produccion)
    CONFIRM_READS      = 3    # lecturas iguales para confirmar
    TRACK_HISTORY      = 50

    # -- Tarifas Alcaldia de Ibague 2025 (COP) --------------------------------
    # Cobro por hora o fraccion (minimo 1 hora)
    FEE_PER_HOUR_CAR  = 3_500
    FEE_PER_HOUR_MOTO = 1_950
    BILLING_MODE      = "hourly_fraction"
    MAX_FEE_CAR       = 25_000
    MAX_FEE_MOTO      = 15_000

    # -- Base de datos --------------------------------------------------------
    DB_PATH = "parking_unibague.db"

    # -- Preprocesamiento -----------------------------------------------------
    CLAHE_CLIP    = 3.0
    CLAHE_GRID    = (8, 8)
    DENOISE_H     = 8
    MORPH_KERNEL  = (2, 2)
    OCR_TARGET_H  = 128

    # -- Display --------------------------------------------------------------
    DISPLAY_SCALE = 1.0
    SHOW_DEBUG    = False   # True para ver ventanas de debug

    # -- Tesseract ------------------------------------------------------------
    TESSERACT_CMD: str = _detect_tesseract()

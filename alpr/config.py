"""
config.py - Configuracion ALPR - Tarifas Universidad de Ibague 2026
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
    # -- Video ----------------------------------------------------------------
    VIDEO_SOURCE   = 0
    FRAME_WIDTH    = 1280
    FRAME_HEIGHT   = 720
    TARGET_FPS     = 30

    # -- Modelo YOLO ----------------------------------------------------------
    YOLO_MODEL_PATH = "yolov8n.pt"
    YOLO_CONF       = 0.35
    YOLO_IOU        = 0.45
    YOLO_CLASSES    = None

    # -- OCR ------------------------------------------------------------------
    MIN_OCR_CONF = 35

    # -- Formatos de placas colombianas ---------------------------------------
    PLATE_PATTERNS = [
        r'^[A-Z]{3}[0-9]{3}$',   # carro/camion: ABC123
        r'^[A-Z]{3}[0-9]{2}$',   # moto: ABC12
        r'^CD[0-9]{3,4}$',       # diplomatico
    ]

    # -- Tracking -------------------------------------------------------------
    TRACK_COOLDOWN_SEC = 5    # subir a 60 en produccion
    CONFIRM_READS      = 1
    TRACK_HISTORY      = 50

    # -- Tarifas Alcaldia de Ibague 2025 (COP) --------------------------------
    # Fuente: Resolucion Alcaldia Ibague - jornada diurna
    #
    # Carro:  entre $2.650 y $4.250 por hora o fraccion
    # Moto:   $1.950 por hora o fraccion
    #
    # Se aplica COBRO MINIMO de 1 hora (fraccion se cobra como hora completa)
    # Ejemplo: 20 minutos = se cobra 1 hora completa
    #          75 minutos = se cobran 2 horas

    FEE_PER_HOUR_CAR  = 3_500   # promedio entre $2.650 y $4.250
    FEE_PER_HOUR_MOTO = 1_950   # tarifa fija Alcaldia

    # Fraccion: cada hora o fraccion se cobra completa
    # (no se cobra por minuto sino por hora iniciada)
    BILLING_MODE = "hourly_fraction"  # opciones: "hourly_fraction" | "per_minute"

    # Maximo diario
    MAX_FEE_CAR  = 25_000
    MAX_FEE_MOTO = 15_000

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
    SHOW_DEBUG    = True

    # -- Tesseract ------------------------------------------------------------
    TESSERACT_CMD: str = _detect_tesseract()

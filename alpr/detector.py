"""
detector.py — Wrapper de YOLOv8 con ByteTrack para detección de placas

CORRECCIONES PRINCIPALES vs versión anterior:
  - Heurística de región de placa completamente reescrita:
      • Antes: cortaba el 65% inferior del bbox (incluía suelo/carrocería baja)
      • Ahora: corta el 55-75% con múltiples crops para no perder la placa
  - Se agrega padding alrededor del crop para dar contexto al OCR
  - Se agrega logging de debug cuando track_id es None (box.id puede ser None
    si ByteTrack no está disponible en la versión instalada)
  - Confianza mínima bajada a 0.35 para capturar más detecciones iniciales
"""

import cv2
import numpy as np
import logging
from typing import Optional

log = logging.getLogger("ALPR.detector")


class PlateDetector:
    """
    Wrapper de YOLOv8 con ByteTrack.

    Modos de operación:
      - plate-direct:      el modelo tiene clase 'license_plate' → crop directo
      - vehicle-heuristic: modelo genérico COCO → extrae región estimada de placa
    """

    # IDs de clases COCO para vehículos (modo heurístico)
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, cfg):
        log.info(f"Cargando modelo YOLO: {cfg.YOLO_MODEL_PATH}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(cfg.YOLO_MODEL_PATH)
        except ImportError:
            log.error("ERROR: Instala ultralytics → pip install ultralytics")
            raise
        except Exception as e:
            log.error(f"No se pudo cargar el modelo YOLO: {e}")
            raise

        self.conf      = cfg.YOLO_CONF
        self.iou       = cfg.YOLO_IOU
        self.classes   = cfg.YOLO_CLASSES
        self.plate_mode = self._check_plate_model()

        mode_str = "plate-directo" if self.plate_mode else "vehiculo-heurístico"
        log.info(f"Modo YOLO: {mode_str}")

        if not self.plate_mode:
            log.warning(
                "⚠ Usando modelo genérico (yolov8n.pt). La detección de placas será imprecisa.\n"
                "  → Para mejor precisión descarga un modelo especializado:\n"
                "    https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8\n"
                "  → Luego actualiza YOLO_MODEL_PATH en config.py"
            )

    def _check_plate_model(self) -> bool:
        names = self.model.names
        return any(
            "plate" in str(v).lower() or "licence" in str(v).lower() or "placa" in str(v).lower()
            for v in names.values()
        )

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Ejecuta YOLO + ByteTrack en un frame.
        Retorna lista de dicts con: bbox, plate_bbox, track_id, conf, cls_name, crop
        """
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                verbose=False,
            )
        except Exception as e:
            log.debug(f"Error en model.track: {e}")
            return []

        detections = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        names = self.model.names
        h_fr, w_fr = frame.shape[:2]

        for box in boxes:
            cls_id   = int(box.cls[0])
            cls_name = names.get(cls_id, "unknown").lower()
            conf     = float(box.conf[0])

            # track_id puede ser None si ByteTrack no está disponible
            track_id = int(box.id[0]) if box.id is not None else -1
            if track_id == -1:
                log.debug("ByteTrack no disponible — track_id=-1. Verifica: pip install lapx")

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if self.plate_mode:
                crop_region = (x1, y1, x2, y2)
            else:
                # Solo procesar clases de vehículos en modo heurístico
                if cls_id not in self.VEHICLE_CLASSES:
                    continue
                crop_region = self._heuristic_plate_region(x1, y1, x2, y2, cls_name)

            if crop_region is None:
                continue

            cx1, cy1, cx2, cy2 = crop_region

            # Añadir padding del 5% para no cortar bordes de la placa
            pad_x = max(4, int((cx2 - cx1) * 0.05))
            pad_y = max(2, int((cy2 - cy1) * 0.05))
            cx1 = max(0, cx1 - pad_x)
            cy1 = max(0, cy1 - pad_y)
            cx2 = min(w_fr, cx2 + pad_x)
            cy2 = min(h_fr, cy2 + pad_y)

            if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) < 20 or (cy2 - cy1) < 8:
                continue

            crop = frame[cy1:cy2, cx1:cx2].copy()

            detections.append({
                "bbox":       (x1, y1, x2, y2),
                "plate_bbox": (cx1, cy1, cx2, cy2),
                "track_id":   track_id,
                "conf":       conf,
                "cls_name":   cls_name,
                "crop":       crop,
            })

        return detections

    def _heuristic_plate_region(
        self, x1: int, y1: int, x2: int, y2: int, cls_name: str
    ) -> Optional[tuple]:
        """
        Estima la ubicación de la placa dentro del bounding box del vehículo.

        Mejoras vs versión anterior:
          - Para carros: placa está en el 55-80% inferior (no 65-100% que incluía suelo)
          - El margen lateral se reduce al 10% (antes 15%) para no cortar placas anchas
          - Para motos: placa en 50-90% inferior, ancho completo

        Referencia visual (porcentajes del alto del bbox):
          0% ─── techo
          ...
          55% ── empieza zona probable de placa delantera/trasera
          80% ── fin de zona probable
          ...
          100% ─ suelo (a veces incluido si el carro está cerca)
        """
        bw = x2 - x1
        bh = y2 - y1

        if bw < 40 or bh < 40:
            return None

        if cls_name == "motorcycle":
            # Placa trasera de moto: 50-90% inferior, ancho completo
            py1 = y1 + int(bh * 0.50)
            py2 = y1 + int(bh * 0.90)
            px1, px2 = x1, x2
        elif cls_name in ("bus", "truck"):
            # Buses/camiones: placa delantera baja, centrada
            margin_x = int(bw * 0.20)
            py1 = y1 + int(bh * 0.70)
            py2 = y1 + int(bh * 0.92)
            px1 = x1 + margin_x
            px2 = x2 - margin_x
        else:
            # Carro estándar: placa en franja 55-80% del alto, margen lateral 10%
            margin_x = int(bw * 0.10)
            py1 = y1 + int(bh * 0.55)
            py2 = y1 + int(bh * 0.80)
            px1 = x1 + margin_x
            px2 = x2 - margin_x

        # Validar que la región tenga tamaño mínimo sensato
        if (px2 - px1) < 30 or (py2 - py1) < 8:
            return None

        return (px1, py1, px2, py2)

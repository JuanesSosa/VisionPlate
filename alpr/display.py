"""
display.py - Utilidades de visualizacion (bounding boxes, HUD, overlays)
Nota: OpenCV en Windows no soporta tildes en putText. ASCII puro.
"""

import cv2
import numpy as np
from datetime import datetime


class Display:
    COLORS = {
        "car":        (0, 200, 50),
        "motorcycle": (0, 150, 255),
        "diplomatic": (200, 0, 200),
        "unknown":    (100, 100, 100),
        "entry":      (0, 255, 120),
        "exit":       (0, 120, 255),
        "pending":    (255, 200, 0),
    }

    STATUS_LABELS = {
        "entry":   "ENTRADA",
        "exit":    "SALIDA",
        "pending": "LEYENDO...",
    }

    @staticmethod
    def draw_detection(frame, det, plate_text, status, fee_info=""):
        x1, y1, x2, y2     = det["bbox"]
        px1, py1, px2, py2 = det["plate_bbox"]
        color        = Display.COLORS.get(status, Display.COLORS["unknown"])
        status_label = Display.STATUS_LABELS.get(status, status.upper())

        # Bbox vehiculo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Bbox placa (amarillo)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 3)

        label = f"{plate_text or '?'}  [{det['conf']:.2f}]  {status_label}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - lh - 12), (x1 + lw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Tarifa debajo del bbox si hay salida
        if fee_info:
            (fw, fh), _ = cv2.getTextSize(fee_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + fw + 6, y2 + fh + 10), (0, 0, 0), -1)
            cv2.putText(frame, fee_info, (x1 + 3, y2 + fh + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    @staticmethod
    def draw_hud(frame, fps, total_inside, revenue):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame,
                    f"Unibague ALPR  |  {ts}  |  FPS: {fps:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        cv2.putText(frame,
                    f"Vehiculos dentro: {total_inside}  |  Recaudo hoy: COP {revenue:,.0f}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 255, 150), 1)

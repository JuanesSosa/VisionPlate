"""
system.py - Controlador principal ALPR v10
- Usa find_similar_active para evitar doble registro por confusion OCR
"""

import cv2
import time
import logging
from collections import deque
from datetime import datetime

from alpr.config import Config
from alpr.database import Database
from alpr.ocr import Preprocessor, OCREngine, extract_plate_by_color
from alpr.tracker import PlateTracker, FeeCalculator
from alpr.detector import PlateDetector
from alpr.display import Display

log = logging.getLogger("ALPR.system")


class ALPRParkingSystem:

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.db           = Database(cfg.DB_PATH)
        self.preprocessor = Preprocessor(cfg)
        self.ocr          = OCREngine(cfg)
        self.tracker      = PlateTracker(cfg)
        self.detector     = PlateDetector(cfg)
        self.fee_calc     = FeeCalculator(cfg)
        self.display      = Display()

        self._fps_counter   = deque(maxlen=30)
        self._daily_revenue = 0.0
        self._state: dict   = {}  # track_id -> {text, status, fee}

        log.info("Sistema listo.")

    def run(self):
        src = self.cfg.VIDEO_SOURCE
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f"No se puede abrir camara: {src}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          self.cfg.TARGET_FPS)
        log.info("Teclas: Q=salir  R=reporte  D=debug")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    if isinstance(src, str): break
                    continue

                out = self._process(frame)

                if self.cfg.DISPLAY_SCALE != 1.0:
                    s   = self.cfg.DISPLAY_SCALE
                    out = cv2.resize(out, (int(out.shape[1]*s),
                                          int(out.shape[0]*s)))

                cv2.imshow("ALPR - Universidad de Ibague", out)

                key = cv2.waitKey(1) & 0xFF
                if   key == ord("q"): break
                elif key == ord("r"): self._print_report()
                elif key == ord("d"):
                    self.cfg.SHOW_DEBUG = not self.cfg.SHOW_DEBUG
                    if not self.cfg.SHOW_DEBUG:
                        for w in ("Placa Debug", "Region Color"):
                            cv2.destroyWindow(w)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            log.info("Sistema cerrado.")

    def _process(self, frame):
        t0         = time.perf_counter()
        detections = self.detector.detect(frame)

        for det in detections:
            tid  = det["track_id"]
            crop = det["crop"]

            if self.cfg.SHOW_DEBUG:
                region = extract_plate_by_color(crop)
                if region is not None:
                    cv2.imshow("Region Color", cv2.resize(region, (300, 80)))

            binary = self.preprocessor.process(crop)
            if binary is None:
                self._draw(frame, det, tid)
                continue

            if self.cfg.SHOW_DEBUG:
                cv2.imshow("Placa Debug", cv2.resize(binary, (300, 100)))

            text, conf = self.ocr.read(binary, self.preprocessor)

            if text:
                log.debug(f"OCR: '{text}' conf={conf:.0f} "
                          f"valido={self.ocr.is_valid(text)}")

            if self.ocr.is_valid(text):
                confirmed = self.tracker.update(tid, text)
                if not confirmed:
                    confirmed = self.tracker.update(-1, text)
                if confirmed:
                    self._register(confirmed, conf, tid)

            self._draw(frame, det, tid)

        elapsed = time.perf_counter() - t0
        self._fps_counter.append(1.0 / max(elapsed, 1e-9))
        fps = sum(self._fps_counter) / len(self._fps_counter)
        self.display.draw_hud(frame, fps,
                              self.db.count_inside(), self._daily_revenue)
        return frame

    def _register(self, plate: str, conf: float, tid: int):
        """
        Registra entrada o salida.
        Antes de crear una entrada nueva, verifica si hay una sesion abierta
        con una placa similar (diferencia de 1 caracter por confusion OCR).
        """
        vtype = self.ocr.infer_vehicle_type(plate)

        # Buscar sesion abierta exacta
        session = self.db.get_open_session(plate)

        # Si no hay sesion exacta, buscar una similar (ELR084 ~ ELR984)
        if session is None:
            active_plates = self.db.get_active_plates()
            similar = self.tracker.find_similar_active(plate, active_plates)
            if similar:
                session = self.db.get_open_session(similar)
                if session:
                    log.info(f"Usando sesion de placa similar: "
                             f"'{plate}' → '{similar}'")
                    plate = similar  # usar la placa correcta

        if session is None:
            # Nueva entrada
            self.db.register_entry(plate, vtype)
            self.db.log_detection(plate, conf)
            self._state[tid] = {"text": plate, "status": "entry", "fee": ""}
            self._state[-1]  = self._state[tid]
            log.info(f"*** ENTRADA: {plate} ({vtype}) ***")
        else:
            # Salida
            dur, fee = self.fee_calc.calculate(
                session["entry_time"], time.time(), session["vehicle_type"])
            self.db.register_exit(session["id"], fee, dur)
            self._daily_revenue += fee
            fee_str = f"COP {fee:,.0f} ({dur:.0f} min)"
            self._state[tid] = {"text": plate, "status": "exit", "fee": fee_str}
            self._state[-1]  = self._state[tid]
            log.info(f"*** SALIDA: {plate} | {dur:.1f} min | COP {fee:,.0f} ***")

    def _draw(self, frame, det, tid):
        s = (self._state.get(tid) or self._state.get(-1) or
             {"text": "?", "status": "pending", "fee": ""})
        self.display.draw_detection(frame, det, s["text"], s["status"], s["fee"])

    def _print_report(self):
        rows = self.db.get_daily_report()
        sep  = "=" * 72
        print(f"\n{sep}\n  REPORTE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(sep)
        print(f"  {'PLACA':<12} {'ENTRADA':<22} {'SALIDA':<22} "
              f"{'MIN':>6} {'COP':>10}")
        print("-" * 72)
        total = 0.0
        for plate, entry_dt, exit_dt, dur, fee in rows:
            salida  = exit_dt[:19] if exit_dt else "ADENTRO"
            print(f"  {plate:<12} {entry_dt[:19]:<22} {salida:<22} "
                  f"{str(round(dur)) if dur else '-':>6} "
                  f"{f'{fee:,.0f}' if fee else '-':>10}")
            if fee: total += fee
        print("-" * 72)
        print(f"  TOTAL HOY: COP {total:,.0f}\n{sep}\n")

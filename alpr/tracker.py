"""
tracker.py - Tracker con deteccion de placas similares

FIX principal: antes de registrar una placa nueva, verifica si ya existe
una placa "similar" activa (difiere en 1 caracter). Si existe, usa la
placa ya registrada en lugar de crear una nueva entrada duplicada.

Esto resuelve casos como ELR984 vs ELR084 donde Tesseract confunde
0/9, 8/B, 1/I, etc. segun el angulo o la iluminacion.
"""

import math
import time
import logging
from collections import defaultdict, deque, Counter
from typing import Optional

log = logging.getLogger("ALPR.tracker")


def plate_similarity(a: str, b: str) -> int:
    """
    Retorna el numero de caracteres diferentes entre dos placas.
    Si tienen distinto largo, retorna un numero grande.
    Ejemplo: ELR984 vs ELR084 → 1 diferencia (pos 3: 9 vs 0)
    """
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(c1 != c2 for c1, c2 in zip(a, b))


# Pares de caracteres que Tesseract confunde frecuentemente
COMMON_CONFUSIONS = {
    ('0', '9'), ('0', 'O'), ('1', 'I'), ('1', 'L'),
    ('5', 'S'), ('8', 'B'), ('6', 'G'), ('2', 'Z'),
}

def is_ocr_confusion(a: str, b: str) -> bool:
    """
    Retorna True si las dos placas difieren solo en confusiones OCR conocidas.
    ELR984 vs ELR084: posicion 3 tiene '9' vs '0' → confusion conocida → True
    """
    if len(a) != len(b):
        return False
    diffs = [(c1, c2) for c1, c2 in zip(a, b) if c1 != c2]
    if len(diffs) == 0:
        return True
    if len(diffs) > 2:
        return False
    return all(
        (c1, c2) in COMMON_CONFUSIONS or (c2, c1) in COMMON_CONFUSIONS
        for c1, c2 in diffs
    )


class PlateTracker:
    def __init__(self, cfg):
        self.confirm_reads = cfg.CONFIRM_READS
        self.cooldown_sec  = cfg.TRACK_COOLDOWN_SEC
        self._candidates: dict = defaultdict(
            lambda: deque(maxlen=max(self.confirm_reads, 3)))
        self._last_confirmed: dict = {}

    def update(self, track_id: int, text: str) -> Optional[str]:
        if not text:
            return None

        q = self._candidates[track_id]
        q.append(text)

        if len(q) < self.confirm_reads:
            return None

        counts = Counter(q)
        top, n = counts.most_common(1)[0]

        if n / len(q) < 0.5:
            return None

        now     = time.time()
        elapsed = now - self._last_confirmed.get(top, 0)
        if elapsed < self.cooldown_sec:
            return None

        self._last_confirmed[top] = now
        log.info(f"Placa confirmada: {top} (tid={track_id})")
        return top

    def find_similar_active(self, plate: str,
                             active_plates: list[str]) -> Optional[str]:
        """
        Busca si alguna placa activa (con sesion abierta) es similar
        a la placa recien leida por confusion OCR.
        Retorna la placa activa si la encuentra, None si no.

        Ejemplo: se lee 'ELR084', hay sesion abierta de 'ELR984'
                 → retorna 'ELR984' (la real)
        """
        for active in active_plates:
            if is_ocr_confusion(plate, active):
                log.info(
                    f"Placa similar detectada: '{plate}' ~ '{active}' "
                    f"(probablemente la misma, usando '{active}')"
                )
                return active
        return None


class FeeCalculator:
    def __init__(self, cfg):
        self.rate_car   = cfg.FEE_PER_HOUR_CAR
        self.rate_moto  = cfg.FEE_PER_HOUR_MOTO
        self.max_car    = cfg.MAX_FEE_CAR
        self.max_moto   = cfg.MAX_FEE_MOTO
        self.mode       = cfg.BILLING_MODE

    def calculate(self, entry_time: float, exit_time: float,
                  vehicle_type: str) -> tuple:
        minutes = (exit_time - entry_time) / 60.0
        rate    = self.rate_moto if vehicle_type == "motorcycle" else self.rate_car
        max_fee = self.max_moto  if vehicle_type == "motorcycle" else self.max_car

        if self.mode == "hourly_fraction":
            hours = max(1, math.ceil(minutes / 60.0))
            fee   = min(hours * rate, max_fee)
        else:
            fee = min((minutes / 60.0) * rate, max_fee)

        return round(minutes, 2), round(fee, 2)

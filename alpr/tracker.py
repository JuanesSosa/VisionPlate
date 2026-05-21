"""
tracker.py - Tracker con similitud de placas mejorada
"""

import math
import time
import logging
from collections import defaultdict, deque, Counter
from typing import Optional

log = logging.getLogger("ALPR.tracker")


def plate_distance(a: str, b: str) -> int:
    """Numero de caracteres distintos entre dos placas del mismo largo."""
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def normalize_plate(plate: str) -> str:
    """
    Normaliza una placa reemplazando todos los caracteres ambiguos
    por una forma canonica, para comparacion fuzzy.

    Ejemplo: NYS401 → N_S_01  /  JNU540 → JN_5_0
    Esto permite detectar que son variaciones de la misma placa.
    """
    ambiguous = {
        # digitos que parecen letras
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'S': '5',
        'B': '8',
        'G': '6',
        'Z': '2',
        # letras que parecen digitos (inverso)
        '0': '0', '1': '1', '5': '5',
        '8': '8', '6': '6', '2': '2',
    }
    return "".join(ambiguous.get(c, c) for c in plate.upper())


def plates_are_same(a: str, b: str, max_diff: int = 2) -> bool:
    """
    Retorna True si dos placas probablemente son la misma
    considerando confusiones OCR.
    Normaliza ambas y compara — tolera hasta max_diff diferencias.
    """
    if len(a) != len(b):
        return False
    na, nb = normalize_plate(a), normalize_plate(b)
    return plate_distance(na, nb) <= max_diff


class PlateTracker:
    def __init__(self, cfg):
        self.confirm_reads = cfg.CONFIRM_READS
        self.cooldown_sec  = cfg.TRACK_COOLDOWN_SEC
        self._candidates: dict = defaultdict(
            lambda: deque(maxlen=max(self.confirm_reads, 5)))
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

        # Requiere mayoria clara (60%)
        if n / len(q) < 0.5:
            log.debug(f"Lecturas inconsistentes para tid={track_id}: {dict(counts)}")
            return None

        now     = time.time()
        elapsed = now - self._last_confirmed.get(top, 0)
        if elapsed < self.cooldown_sec:
            return None

        self._last_confirmed[top] = now
        log.info(f"Placa confirmada: {top} ({n}/{len(q)} lecturas, tid={track_id})")
        return top

    def find_similar_active(self, plate: str,
                            active_plates: list) -> Optional[str]:
        """
        Busca si alguna placa activa es la misma que 'plate'
        con confusiones OCR. Tolera hasta 2 caracteres distintos
        despues de normalizar.
        """
        for active in active_plates:
            if plates_are_same(plate, active, max_diff=2):
                log.info(f"Placa similar: '{plate}' ~ '{active}' → usando '{active}'")
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

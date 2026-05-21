"""
tracker.py - Seguimiento de placas y calculo de tarifas
Tarifas: Alcaldia de Ibague 2025 - cobro por hora o fraccion
"""

import math
import time
import logging
from collections import defaultdict, deque, Counter
from typing import Optional

log = logging.getLogger("ALPR.tracker")


class PlateTracker:
    def __init__(self, cfg):
        self.confirm_reads = cfg.CONFIRM_READS
        self.cooldown_sec  = cfg.TRACK_COOLDOWN_SEC
        self._candidates: dict = defaultdict(lambda: deque(maxlen=max(self.confirm_reads, 3)))
        self._last_confirmed: dict = {}

    def update(self, track_id: int, text: str) -> Optional[str]:
        if not text:
            return None

        q = self._candidates[track_id]
        q.append(text)

        if len(q) < self.confirm_reads:
            return None

        counts   = Counter(q)
        top, n   = counts.most_common(1)[0]

        if n / len(q) < 0.5:
            return None

        now     = time.time()
        elapsed = now - self._last_confirmed.get(top, 0)
        if elapsed < self.cooldown_sec:
            return None

        self._last_confirmed[top] = now
        log.info(f"Placa confirmada: {top} (tid={track_id})")
        return top


class FeeCalculator:
    """
    Calcula tarifas segun resolucion Alcaldia de Ibague 2025.
    Modo: por hora o fraccion (cada hora iniciada se cobra completa).
    """

    def __init__(self, cfg):
        self.rate_car   = cfg.FEE_PER_HOUR_CAR
        self.rate_moto  = cfg.FEE_PER_HOUR_MOTO
        self.max_car    = cfg.MAX_FEE_CAR
        self.max_moto   = cfg.MAX_FEE_MOTO
        self.mode       = cfg.BILLING_MODE

    def calculate(self, entry_time: float, exit_time: float,
                  vehicle_type: str) -> tuple:
        """
        Retorna (duracion_minutos, tarifa_COP).

        Modo hourly_fraction:
          - Minimo cobro: 1 hora
          - Cada hora iniciada se cobra completa
          - Ej: 20min -> 1h -> $3.500
                75min -> 2h -> $7.000
               130min -> 3h -> $10.500
        """
        minutes = (exit_time - entry_time) / 60.0
        rate    = self.rate_moto if vehicle_type == "motorcycle" else self.rate_car
        max_fee = self.max_moto  if vehicle_type == "motorcycle" else self.max_car

        if self.mode == "hourly_fraction":
            # Redondear hacia arriba al multiplo de hora mas cercano (minimo 1h)
            hours = max(1, math.ceil(minutes / 60.0))
            fee   = min(hours * rate, max_fee)
        else:
            # Modo por minuto (legacy)
            fee = min((minutes / 60.0) * rate, max_fee)

        return round(minutes, 2), round(fee, 2)

    def preview(self, minutes: float, vehicle_type: str) -> str:
        """Genera string de previsualizacion de tarifa."""
        _, fee = self.calculate(0, minutes * 60, vehicle_type)
        hours  = max(1, math.ceil(minutes / 60.0))
        return f"COP {fee:,.0f} ({hours}h cobradas)"

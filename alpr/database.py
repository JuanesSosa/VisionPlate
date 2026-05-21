"""
database.py — Capa de persistencia SQLite para el sistema ALPR
"""

import sqlite3
import time
import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger("ALPR.db")


class Database:
    """Interfaz SQLite para registros de parqueo."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        log.info(f"Base de datos lista: {db_path}")

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS parking_sessions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                plate         TEXT    NOT NULL,
                vehicle_type  TEXT    DEFAULT 'car',
                entry_time    REAL    NOT NULL,
                exit_time     REAL,
                duration_min  REAL,
                fee_cop       REAL,
                entry_dt      TEXT,
                exit_dt       TEXT
            );

            CREATE TABLE IF NOT EXISTS detections_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                plate       TEXT    NOT NULL,
                confidence  REAL,
                detected_at REAL,
                frame_ts    TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_plate ON parking_sessions(plate);
        """)
        self.conn.commit()

    def get_open_session(self, plate: str) -> Optional[dict]:
        """Retorna la sesión abierta (sin salida) de una placa, o None."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, entry_time, vehicle_type FROM parking_sessions "
            "WHERE plate=? AND exit_time IS NULL ORDER BY entry_time DESC LIMIT 1",
            (plate,)
        )
        row = cur.fetchone()
        if row:
            return {"id": row[0], "entry_time": row[1], "vehicle_type": row[2]}
        return None

    def register_entry(self, plate: str, vehicle_type: str) -> int:
        now = time.time()
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO parking_sessions (plate, vehicle_type, entry_time, entry_dt) VALUES (?,?,?,?)",
            (plate, vehicle_type, now, datetime.fromtimestamp(now).isoformat())
        )
        self.conn.commit()
        log.info(f"ENTRADA registrada: {plate} ({vehicle_type})")
        return cur.lastrowid

    def register_exit(self, session_id: int, fee_cop: float, duration_min: float):
        now = time.time()
        cur = self.conn.cursor()
        cur.execute(
            """UPDATE parking_sessions
               SET exit_time=?, exit_dt=?, duration_min=?, fee_cop=?
               WHERE id=?""",
            (now, datetime.fromtimestamp(now).isoformat(), duration_min, fee_cop, session_id)
        )
        self.conn.commit()
        log.info(f"SALIDA registrada: sesión {session_id} | {duration_min:.1f} min | COP {fee_cop:,.0f}")

    def log_detection(self, plate: str, confidence: float):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO detections_log (plate, confidence, detected_at, frame_ts) VALUES (?,?,?,?)",
            (plate, confidence, time.time(), datetime.now().isoformat())
        )
        self.conn.commit()

    def get_daily_report(self) -> list:
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT plate, entry_dt, exit_dt, duration_min, fee_cop "
            "FROM parking_sessions WHERE entry_time >= ? ORDER BY entry_time DESC",
            (today_start,)
        )
        return cur.fetchall()

    def count_inside(self) -> int:
        """Cuenta vehículos actualmente dentro del parqueadero."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM parking_sessions WHERE exit_time IS NULL"
        )
        row = cur.fetchone()
        return row[0] if row else 0

    def close(self):
        self.conn.close()

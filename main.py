"""
main.py - ALPR Universidad de Ibague

Uso:
    python main.py                  # camara del portatil
    python main.py --source 1       # segunda camara
    python main.py --source vid.mp4 # archivo de video
    python main.py --reset-db       # borrar BD y empezar de cero
    python main.py --debug          # mostrar ventanas de debug
    python main.py --debug-ocr      # logs detallados en consola
    python main.py --cooldown 5     # cooldown corto para pruebas
"""

import sys
import logging
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="ALPR - Universidad de Ibague")
    p.add_argument("--source",    default=None,
                   help="0=portatil, 1=segunda camara, o ruta a video")
    p.add_argument("--model",     default=None)
    p.add_argument("--conf",      type=float, default=None)
    p.add_argument("--scale",     type=float, default=None)
    p.add_argument("--cooldown",  type=int,   default=None,
                   help="Segundos entre ENTRADA y SALIDA (pruebas: 5)")
    p.add_argument("--debug",     action="store_true",
                   help="Mostrar ventanas Region Color y Placa Debug")
    p.add_argument("--debug-ocr", action="store_true",
                   help="Logs detallados de OCR y tracker")
    p.add_argument("--reset-db",  action="store_true",
                   help="Borra la base de datos")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug_ocr else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("alpr_system.log", encoding="utf-8"),
        ],
    )

    from alpr.config import Config
    from alpr.system import ALPRParkingSystem

    cfg = Config()

    if args.source is not None:
        cfg.VIDEO_SOURCE = (int(args.source)
                            if args.source.isdigit() else args.source)
    if args.model    is not None: cfg.YOLO_MODEL_PATH   = args.model
    if args.conf     is not None: cfg.YOLO_CONF         = args.conf
    if args.scale    is not None: cfg.DISPLAY_SCALE     = args.scale
    if args.cooldown is not None: cfg.TRACK_COOLDOWN_SEC = args.cooldown
    if args.debug:                cfg.SHOW_DEBUG         = True

    if args.reset_db:
        import os
        if os.path.exists(cfg.DB_PATH):
            os.remove(cfg.DB_PATH)
            print(f"Base de datos eliminada: {cfg.DB_PATH}")

    print(f"""
==============================================================
  ALPR - Universidad de Ibague
  Camara         : {cfg.VIDEO_SOURCE}
  Cooldown       : {cfg.TRACK_COOLDOWN_SEC}s entre ENTRADA y SALIDA
  Tarifa carro   : COP {cfg.FEE_PER_HOUR_CAR:,}/hora o fraccion
  Tarifa moto    : COP {cfg.FEE_PER_HOUR_MOTO:,}/hora o fraccion
  Debug          : {'ON' if cfg.SHOW_DEBUG else 'OFF'}
==============================================================
  Flujo:
    1. Carro entra  -> placa detectada -> ENTRADA (verde)
    2. Carro sale   -> placa detectada -> SALIDA + precio (azul)

  Teclas: Q=salir  R=reporte  D=toggle debug
==============================================================
""")

    ALPRParkingSystem(cfg).run()


if __name__ == "__main__":
    main()

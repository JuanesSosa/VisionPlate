"""
main.py - Punto de entrada del sistema ALPR

Uso:
    python main.py                        # webcam
    python main.py --source video.mp4     # archivo de video
    python main.py --conf 0.25            # umbral de confianza bajo
    python main.py --model placa_col.pt   # modelo especializado
    python main.py --debug-ocr            # logs detallados de OCR y tracker
    python main.py --reset-db             # borrar la BD y empezar de cero
"""

import sys
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="ALPR - Universidad de Ibague")
    parser.add_argument("--source",    default=None)
    parser.add_argument("--model",     default=None)
    parser.add_argument("--conf",      type=float, default=None)
    parser.add_argument("--scale",     type=float, default=None)
    parser.add_argument("--no-debug",  action="store_true")
    parser.add_argument("--debug-ocr", action="store_true",
                        help="Mostrar logs detallados de OCR y tracker en consola")
    parser.add_argument("--reset-db",  action="store_true",
                        help="Borrar la base de datos y empezar de cero")
    parser.add_argument("--cooldown",  type=int, default=None,
                        help="Segundos de cooldown entre lecturas (default: 5 en pruebas)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Nivel de log
    log_level = logging.DEBUG if args.debug_ocr else logging.INFO
    logging.basicConfig(
        level=log_level,
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
        cfg.VIDEO_SOURCE = int(args.source) if args.source.isdigit() else args.source
    if args.model    is not None: cfg.YOLO_MODEL_PATH = args.model
    if args.conf     is not None: cfg.YOLO_CONF       = args.conf
    if args.scale    is not None: cfg.DISPLAY_SCALE   = args.scale
    if args.cooldown is not None: cfg.TRACK_COOLDOWN_SEC = args.cooldown
    if args.no_debug:             cfg.SHOW_DEBUG       = False

    if args.reset_db:
        import os
        if os.path.exists(cfg.DB_PATH):
            os.remove(cfg.DB_PATH)
            print(f"Base de datos eliminada: {cfg.DB_PATH}")

    print(f"""
======================================================
  ALPR - Universidad de Ibague
  Cooldown entre lecturas : {cfg.TRACK_COOLDOWN_SEC}s
  Confirmaciones requeridas: {cfg.CONFIRM_READS}
  Modelo: {cfg.YOLO_MODEL_PATH}
======================================================
  Flujo de prueba:
  1. Muestra la placa -> aparece ENTRADA (verde)
  2. Espera {cfg.TRACK_COOLDOWN_SEC}s
  3. Vuelve a mostrar la placa -> aparece SALIDA + precio (azul)

  Teclas: Q=salir  R=reporte  D=toggle debug
======================================================
""")

    system = ALPRParkingSystem(cfg)
    system.run()


if __name__ == "__main__":
    main()

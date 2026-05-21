# ALPR Parking System — Universidad de Ibagué

Sistema de reconocimiento automático de placas (ALPR) con gestión de tarifas de parqueo, diseñado para el parqueadero exterior de la Universidad de Ibagué.

---

## Estructura del proyecto

```
alpr_project/
├── main.py                  # Punto de entrada
├── requirements.txt
├── parking_unibague.db      # Base de datos SQLite (se crea automáticamente)
├── alpr_system.log          # Log del sistema (se crea automáticamente)
└── alpr/
    ├── config.py            # Toda la configuración en un solo lugar
    ├── database.py          # Capa de persistencia SQLite
    ├── ocr.py               # Preprocesamiento de imagen + OCR Tesseract
    ├── detector.py          # Wrapper YOLOv8 + ByteTrack
    ├── tracker.py           # Seguimiento de placas + calculadora de tarifas
    ├── display.py           # Visualización (bounding boxes, HUD)
    └── system.py            # Controlador principal
```

---

## Instalación

### 1. Dependencias Python

```bash
pip install -r requirements.txt
```

### 2. Tesseract OCR

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

**Windows:**
Descarga e instala desde: https://github.com/UB-Mannheim/tesseract/wiki
La ruta se detecta automáticamente. Si falla, edita `TESSERACT_CMD` en `alpr/config.py`.

**macOS:**
```bash
brew install tesseract tesseract-lang
```

### 3. Modelo YOLO (importante para detección de placas)

El archivo `yolov8n.pt` incluido es un modelo **genérico** (80 clases COCO). Funciona en modo heurístico pero la detección de placas es imprecisa.

**Para mejor precisión**, descarga un modelo especializado en placas:
```bash
# Opción A — modelo de placas colombianas (recomendado)
# https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8

# Opción B — descargar con Python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # reemplazar con modelo de placas cuando esté disponible
```

Luego actualiza en `alpr/config.py`:
```python
YOLO_MODEL_PATH = "ruta/a/tu/modelo_placas.pt"
```

---

## Uso

```bash
# Webcam por defecto
python main.py

# Archivo de video
python main.py --source video.mp4

# Con modelo especializado en placas
python main.py --model modelo_placas.pt

# Umbral de confianza más bajo (detecta más, más falsos positivos)
python main.py --conf 0.25

# Ventana más pequeña
python main.py --scale 0.75

# Sin ventana de debug
python main.py --no-debug
```

### Teclas durante la ejecución

| Tecla | Acción |
|-------|--------|
| `Q`   | Salir |
| `R`   | Imprimir reporte del día en consola |
| `D`   | Toggle ventana de debug (crop preprocesado) |

---

## Cómo funciona

1. **Captura** → frame de webcam o video
2. **Detección** → YOLOv8 detecta vehículos (o placas si el modelo lo soporta)
3. **Extracción de región** → se recorta la zona probable de la placa
4. **Preprocesamiento** → CLAHE + denoising + umbralización adaptativa
5. **OCR** → Tesseract con estrategia multi-paso (PSM8, PSM7, invertida, multi-escala)
6. **Validación** → se verifica el formato de placa colombiana (ABC123, ABC12, CD1234)
7. **Confirmación** → se requieren 3 lecturas consecutivas coincidentes
8. **Lógica de parqueo** → si es entrada nueva → registra; si ya estaba adentro → calcula tarifa y registra salida
9. **Base de datos** → todo queda en SQLite para reportes

---

## Tarifas configuradas

| Vehículo    | Por minuto | Por hora  | Máximo/día |
|-------------|-----------|-----------|------------|
| Carro       | COP 150   | COP 9.000 | COP 25.000 |
| Moto        | COP 75    | COP 4.500 | COP 25.000 |

Edita `FEE_PER_MINUTE_CAR`, `FEE_PER_MINUTE_MOTO` y `MAX_FEE_PER_DAY` en `alpr/config.py`.

---

## Solución de problemas comunes

**El sistema detecta vehículos pero no lee placas:**
- El modelo `yolov8n.pt` es genérico. Descarga un modelo especializado en placas.
- Revisa que Tesseract esté instalado: `tesseract --version`
- Baja `MIN_OCR_CONF` a 40 en `config.py` para ser menos estricto.

**`track_id` siempre es -1:**
- Instala `lapx`: `pip install lapx` (necesario para ByteTrack)

**Ventana no abre (servidor sin pantalla):**
- Usa `--no-debug` y procesa un archivo de video guardando los resultados.

**Placas con formatos distintos no se detectan:**
- Agrega el patrón regex correspondiente en `PLATE_PATTERNS` dentro de `config.py`.

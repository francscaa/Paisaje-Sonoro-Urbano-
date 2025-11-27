¡Perfecto Fran!
Te preparo un **README profesional, claro, bonito y completo**, ideal para tu proyecto de análisis de paisaje sonoro urbano.
Incluye: descripción, instalación, uso, estructura del repo, ejemplos, y notas técnicas.

Si luego quieres, puedo agregar **logos, badges, GIFs, capturas, citaciones académicas, referencias** o lo que necesites para la memoria.

---

# 📄 **README.md — Paisaje Sonoro Urbano**

Aquí tienes la primera versión:

---

```markdown
# 🎧 Paisaje Sonoro Urbano — Sistema de Análisis Acústico

Proyecto de análisis del paisaje sonoro urbano que combina **clasificación automática de sonidos (YAMNet)**, **métricas psicoacústicas (Mosqito)** y **visualización perceptual (Soundscapy)**.  
Permite estudiar la composición acústica de distintos espacios mediante **segmentación temporal**, **identificación de fuentes sonoras** y **cálculo de indicadores perceptuales**.

Este repositorio corresponde al proyecto de memoria de **Francisca Barraza Escobar**.

---

## 📂 Estructura del Proyecto

```

Paisaje-Sonoro-Urbano-/
│
├── recordings/          # Audios de entrada (24–48 kHz)
├── results/             # CSV, gráficos y resultados exportados
├── scripts/             # Scripts principales de análisis
│   ├── YAMNet_Soundscapy.py
│   └── otros scripts de apoyo
├── models/              # Caché opcional para modelos de TensorFlow Hub
└── .gitignore

````

---

## ✨ Funcionalidades principales

### 🔍 1. Segmentación automática
- Divide el audio en ventanas de 3s (configurable).
- Procesa cada segmento de forma independiente.

### 🎵 2. Clasificación de fuentes sonoras (YAMNet)
Para cada segmento:
- Detecta la clase más probable (ej: *Vehicle*, *Speech*, *Engine*, *Bird*…)
- Guarda probabilidad y top-1 prediction.

### 🔊 3. Métricas psicoacústicas (Mosqito)
Por segmento calcula:
- **Loudness (sones)**
- **Sharpness**
- **Roughness**
- **Tonality (TNR / PR)**

### 📈 4. Análisis perceptual avanzado (Soundscapy)
Permite generar:
- Mapas perceptuales 2D por descriptor
- Gráficas comparativas por lugar
- Nubes de puntos con kernels perceptuales
- Modelos de *pleasantness*, *eventfulness*, *PAQ* y más

### 📊 5. Exportación a CSV + Gráficos automáticos
- Un CSV final con todos los segmentos
- Gráficos listos para tesis/artículos:
  - Fuentes sonoras vs tiempo  
  - Comparación de descriptores por audio  
  - Mapas perceptuales (si se activa Soundscapy)

---

## 🛠️ Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/francscaa/Paisaje-Sonoro-Urbano-.git
cd Paisaje-Sonoro-Urbano-
````

### 2️⃣ Crear entorno virtual (Python 3.10 recomendado)

#### macOS / Linux

```bash
python3 -m venv entorno
source entorno/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv entorno
entorno\Scripts\activate
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso

### Procesar un archivo de audio

```bash
python scripts/YAMNet_Soundscapy.py --file recordings/mi_audio.wav --window 3 --hop 3
```

El script:

* carga el audio
* lo segmenta
* aplica YAMNet
* calcula descriptores psicoacústicos
* genera resultados en:

```
results/yamnet_psico_segmentado.csv
results/plots/
```

---

## 📊 Resultados esperados

* **CSV completo** con:
  `Timestamp`, `Clase_YAMNet`, `Probabilidad`, `loudness_sones`, `sharpness_acum`, `roughness_asper`, `tonality_*`
* **Scatter plot** con fuentes vs probabilidad
* **Comparación psicoacústica por audio**

Ejemplo de columnas:

```csv
Timestamp,Recording,Clase_YAMNet,Probabilidad,loudness_sones,sharpness_acum,roughness_asper,tonality_tnr_db
0.0,mi_audio,Vehicle,0.81,22.5,1.34,0.05,0.0
3.0,mi_audio,Speech,0.67,18.1,1.12,0.04,0.0
...
```

---

## 📌 Notas importantes

* TensorFlow para Windows funciona **solo con Python 3.10**.
* Los entornos virtuales **no deben subirse a GitHub**.
* Soundscapy necesita **Seaborn**, **Plotly** y **Scipy** funcionando correctamente.
* Si usas Mac M1/M2/M3, TensorFlow puede requerir instalación específica.

---

## ✍️ Autora

**Francisca Barraza Escobar**
Diseño de Interacción Digital — UDD
2025

---

## 📜 Licencia

Este proyecto se distribuye bajo licencia MIT.
Puedes usar, modificar y citar este repositorio libremente.

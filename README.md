# Análisis Exploratorio de Datasets Públicos para Intérprete Multimodal de LSM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FAIR](https://img.shields.io/badge/FAIR-data-green.svg)](https://www.go-fair.org/fair-principles/)

Repositorio de soporte para la tarea académica **"Búsqueda, caracterización y análisis exploratorio de cinco datasets públicos"** del proyecto de tesis sobre el desarrollo de un intérprete multimodal de Lengua de Señas Mexicana (LSM) a voz y texto.

> **Autora:** Valeria Aguilar Meza  
> **Programa:** Ingeniería en Electrónica · Semestre 2026-1  
> **Director de tesis:** Dr. Everardo Inzunza Gonzalez  
> **Institución:** Facultad de Ingeniería, Arquitectura y Diseño (FIAD), Universidad Autónoma de Baja California  
> **Fecha:** 4 de mayo de 2026

---

## Resumen

Este repositorio contiene la caracterización y análisis exploratorio (EDA) de cinco conjuntos de datos públicos relacionados con la Lengua de Señas Mexicana, siguiendo los principios FAIR y el estándar *Datasheets for Datasets* (Gebru et al., 2021). Los datasets cubren tres modalidades complementarias:

- **Imágenes estáticas** (D1 Montesinos, D2 Chacon, D3 warcoder)
- **Video dinámico** (D4 Sánchez-Vicinaiz et al.)
- **Corpus textual paralelo** (D5 Lara-Ortiz et al.)

---

## Datasets caracterizados

| ID | Nombre | Autor | Año | Repositorio | DOI / URL | Licencia |
|---|---|---|---|---|---|---|
| **D1** | LSM (~100K fotos) | Montesinos | 2022 | Kaggle | [URL](https://www.kaggle.com/datasets/osvalmontesinos/lengua-de-seas-mexicana-100000-fotos) | Kaggle ToU |
| **D2** | LSM Lenguaje de señas mexicanas | Chacon Quintero et al. | 2022 | Zenodo | [10.5281/zenodo.6554337](https://doi.org/10.5281/zenodo.6554337) | CC-BY 4.0 |
| **D3** | Mexican Sign Language Dataset | warcoder | 2023 | Kaggle | [URL](https://www.kaggle.com/datasets/warcoder/mexican-sign-language-dataset) | Kaggle ToU |
| **D4** | MSL Alphabet (dynamic only) | Sánchez-Vicinaiz et al. | 2025 | Zenodo | [10.5281/zenodo.14689869](https://doi.org/10.5281/zenodo.14689869) | CC-BY 4.0 |
| **D5** | SPA→MSL glosses corpus | Lara-Ortiz et al. | 2025 | ScienceDB | [10.1038/s41597-025-04871-7](https://doi.org/10.1038/s41597-025-04871-7) | CC-BY 4.0 |

---

## Estructura del repositorio

```
lsm-datasets-tesis/
├── README.md                              ← este archivo
├── LICENSE                                ← licencia MIT
├── requirements.txt                       ← dependencias Python
├── .gitignore                             ← archivos ignorados por git
│
├── docs/
│   ├── Aguilar_Meza_datasets_tesis.pdf    ← documento técnico final (entregable 1)
│   └── Aguilar_Meza_comparativa.xlsx      ← tabla comparativa (entregable 4)
│
├── notebooks/                             ← scripts EDA reproducibles
│   ├── 01_eda_montesinos.py               ← EDA D1 (muestra de 200 imgs)
│   ├── 02_eda_chacon_zenodo.py            ← EDA D2 (72 imgs keypoints reales)
│   ├── 03_eda_warcoder.py                 ← EDA D3 (muestra de 25 imgs)
│   ├── 04_eda_dynamic.py                  ← EDA D4 (309 videos, 3 letras)
│   └── 05_eda_glosses_corpus.py           ← EDA D5 (3000 pares completo)
│
├── src/                                   ← módulos auxiliares reutilizables
│   ├── viz_style.py                       ← estilo unificado de figuras (seaborn)
│   ├── stats_utils.py                     ← descriptiva, IC95%, Shapiro-Wilk, IQR
│   ├── eda_utils.py                       ← inventario, metadatos imagen/video
│   └── nlp_utils.py                       ← tokenización, TTR, n-gramas
│
├── figures/                               ← visualizaciones generadas (300 dpi)
│   ├── d1_v1_balance.png
│   ├── d1_v2_brillo.png
│   ├── d1_v3_resolucion.png
│   ├── d2_chacon_keypoints_muestra.png
│   ├── d2_chacon_tsne.png
│   ├── d3_v1.png
│   ├── d3_v2.png
│   ├── d4_dinamico_v1.png
│   ├── d4_dinamico_v2.png
│   └── d5_v1_distribucion.png ... (5 figuras)
│
└── outputs/                               ← datos derivados
    ├── tablas/                            ← tablas LaTeX (booktabs)
    │   ├── desc_d2.tex, norm_d2.tex
    │   └── desc_d5.tex, norm_d5.tex, bigramas_d5.tex
    └── metadatos/                         ← CSVs con métricas calculadas
        ├── d2_chacon_metadatos.csv
        └── d5_glosses_metadatos.csv
```

---

## Requisitos e instalación

### Dependencias Python

```bash
python>=3.10
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
scipy>=1.10
opencv-python>=4.7
pillow>=9.4
openpyxl>=3.1
```

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/valeriaaguilarmeza05-hash/lsm-datasets-tesis.git
cd lsm-datasets-tesis

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Cómo reproducir los EDAs

Cada notebook es **autónomo y reproducible** con `random_state=42`. Para reejecutar:

```bash
# Procesar dataset individual
python notebooks/05_eda_glosses_corpus.py

# Procesar dataset con archivo local
python notebooks/02_eda_chacon_zenodo.py --input ./datos/ABECEDARIOIMAGENES.pickle
```

Los scripts generan automáticamente:
- Las figuras del EDA en `figures/`
- Las tablas estadísticas en `outputs/tablas/`
- Los metadatos en CSV en `outputs/metadatos/`

---

## Hallazgos principales del EDA

### D1 Montesinos
- **Bimodalidad fotométrica** ($p<10^{-4}$, Shapiro-Wilk): dos condiciones de iluminación distintas
- Resolución uniforme 224×224 px (preprocesamiento previo del autor para uso con redes ImageNet)
- Brillo medio: $\bar{X}=81.97$, $\sigma=39.73$, IC$_{95\%}=[76.46, 87.47]$

### D2 Chacon Quintero et al.
- **Hallazgo crítico:** las imágenes son visualizaciones renderizadas de *keypoints* MediaPipe Holistic, no fotografías RGB
- Esta caracterización **alinea favorablemente** el dataset con el pipeline embebido propuesto
- 72 imágenes 640×360 px, balance ordinal aceptable (entropía 0.937)

### D3 warcoder
- Bimodalidad de iluminación pese al control declarado por el autor
- Brillo: $\bar{X}=37.52$, $\sigma=3.31$, IC$_{95\%}=[36.22, 38.82]$ (muestra $n=25$)

### D4 Sánchez-Vicinaiz et al.
- Distribución bimodal de duración (1.5s y 2.1s) que refleja dos patrones articulatorios
- $\bar{X}=1.749$ s, IC$_{95\%}=[1.683, 1.815]$ s ($n=309$ videos)
- Compatible con ventanas temporales para LSTM/GRU

### D5 Lara-Ortiz et al.
- **Compresión SOV cuantificada:** $\Delta=+0.327$ palabras (IC$_{95\%}=[0.289, 0.364]$)
- **Marcación analítica del género femenino:** la palabra *mujer* es la más frecuente en LSM
- Vocabulario más cerrado en LSM (TTR 0.0569 vs 0.0685 español)

---

## Pipeline propuesto para la tesis

```
[Captura] → MediaPipe Holistic (543 keypoints) → [LSTM/GRU/Transformer]
              ↓                                            ↓
        Reducción dimensional                    Reconocimiento de seña
              ↓                                            ↓
       Despliegue en Raspberry Pi 4/5      Modelo seq2seq (D5) → Glosa LSM → Texto/Voz
```

---

## Cita

Si usas este trabajo en tu investigación, por favor cita:

```bibtex
@misc{aguilar2026lsmdatasets,
  author       = {Aguilar Meza, Valeria},
  title        = {Búsqueda, caracterización y análisis exploratorio de cinco datasets públicos para el desarrollo de un intérprete multimodal de Lengua de Señas Mexicana},
  year         = {2026},
  institution  = {Facultad de Ingeniería, Arquitectura y Diseño, Universidad Autónoma de Baja California},
  howpublished = {GitHub},
  url          = {https://github.com/valeriaaguilarmeza05-hash/lsm-datasets-tesis}
}
```

---

## Licencia

Este repositorio está bajo la licencia [MIT](LICENSE). Los datasets analizados conservan sus licencias originales (ver tabla arriba).

---

## Contacto

**Valeria Aguilar Meza** · valeria.aguilar79@uabc.edu.mx  
**Director:** Dr. Everardo Inzunza Gonzalez ·

Facultad de Ingeniería, Arquitectura y Diseño (FIAD)  
Universidad Autónoma de Baja California  
Ensenada, Baja California, México

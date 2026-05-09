"""
stats_utils.py
==============
Funciones de análisis estadístico descriptivo reproducible para el EDA
de los cinco datasets de Lengua de Señas Mexicana (LSM).

Cumple los requerimientos del apartado 5.3 de la rúbrica:
    - Medias, medianas, desviación estándar, rangos
    - Valores faltantes y outliers (regla IQR)
    - Balance de clases
    - Prueba de normalidad (Shapiro-Wilk o Kolmogorov-Smirnov)
    - Intervalos de confianza al 95 %

Autor: Valeria Aguilar Meza · FIAD-UABC · 2026-1
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1.  Estadística descriptiva univariada
# ---------------------------------------------------------------------------

def descriptiva(serie, nombre="variable"):
    """
    Resumen descriptivo completo de una serie numérica.
    Devuelve un DataFrame de una fila listo para concatenar y exportar a LaTeX.
    """
    s = pd.Series(serie).dropna()
    if len(s) == 0:
        return pd.DataFrame()

    resumen = {
        "variable": nombre,
        "n": len(s),
        "media": np.round(s.mean(), 4),
        "mediana": np.round(s.median(), 4),
        "std": np.round(s.std(ddof=1), 4),
        "min": np.round(s.min(), 4),
        "max": np.round(s.max(), 4),
        "rango": np.round(s.max() - s.min(), 4),
        "q1": np.round(s.quantile(0.25), 4),
        "q3": np.round(s.quantile(0.75), 4),
        "iqr": np.round(s.quantile(0.75) - s.quantile(0.25), 4),
        "cv_%": np.round(100 * s.std(ddof=1) / s.mean(), 2) if s.mean() != 0 else np.nan,
        "skew": np.round(s.skew(), 4),
        "kurtosis": np.round(s.kurtosis(), 4),
        "n_faltantes": int(pd.Series(serie).isna().sum()),
    }
    return pd.DataFrame([resumen])


# ---------------------------------------------------------------------------
# 2.  Intervalos de confianza al 95 %
# ---------------------------------------------------------------------------

def ic_95(serie, metodo="t"):
    """
    Intervalo de confianza al 95 % para la media.
        metodo='t'         -> t de Student (recomendado n<30 o sigma desconocida)
        metodo='bootstrap' -> bootstrap percentil (no paramétrico, robusto)
    """
    s = pd.Series(serie).dropna().to_numpy()
    if len(s) < 2:
        return (np.nan, np.nan)

    if metodo == "t":
        media = s.mean()
        sem = stats.sem(s)
        ic = stats.t.interval(confidence=0.95, df=len(s) - 1, loc=media, scale=sem)
        return (float(np.round(ic[0], 4)), float(np.round(ic[1], 4)))

    elif metodo == "bootstrap":
        rng = np.random.default_rng(seed=42)
        n_boot = 5000
        muestras = rng.choice(s, size=(n_boot, len(s)), replace=True)
        medias_boot = muestras.mean(axis=1)
        return (float(np.round(np.percentile(medias_boot, 2.5), 4)),
                float(np.round(np.percentile(medias_boot, 97.5), 4)))


# ---------------------------------------------------------------------------
# 3.  Pruebas de normalidad
# ---------------------------------------------------------------------------

def prueba_normalidad(serie, alpha=0.05):
    """
    Aplica Shapiro-Wilk si n<=5000, en caso contrario Kolmogorov-Smirnov.
    Devuelve dict con estadístico, p-valor, decisión y prueba aplicada.
    """
    s = pd.Series(serie).dropna().to_numpy()
    n = len(s)
    if n < 3:
        return {"prueba": "n<3, no aplica", "stat": np.nan, "p_valor": np.nan,
                "es_normal": np.nan, "n": n}

    if n <= 5000:
        stat, p = stats.shapiro(s)
        nombre = "Shapiro-Wilk"
    else:
        # KS contra normal estandarizada (con media y std muestrales)
        s_std = (s - s.mean()) / s.std(ddof=1)
        stat, p = stats.kstest(s_std, "norm")
        nombre = "Kolmogorov-Smirnov"

    return {
        "prueba": nombre,
        "n": n,
        "stat": float(np.round(stat, 6)),
        "p_valor": float(np.round(p, 6)),
        "alpha": alpha,
        "es_normal": bool(p > alpha),
    }


# ---------------------------------------------------------------------------
# 4.  Detección de outliers (regla IQR de Tukey)
# ---------------------------------------------------------------------------

def detectar_outliers_iqr(serie, k=1.5):
    """
    Devuelve número y proporción de outliers según la regla IQR de Tukey
    (valores fuera de [Q1 - k·IQR, Q3 + k·IQR]).
    """
    s = pd.Series(serie).dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    li, ls = q1 - k * iqr, q3 + k * iqr
    mask = (s < li) | (s > ls)
    return {
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(np.round(100 * mask.mean(), 3)),
        "limite_inf": float(np.round(li, 4)),
        "limite_sup": float(np.round(ls, 4)),
    }


# ---------------------------------------------------------------------------
# 5.  Balance de clases
# ---------------------------------------------------------------------------

def balance_clases(etiquetas):
    """
    Métricas de balance:
        - frecuencia absoluta y relativa por clase
        - razón mayoritaria/minoritaria
        - entropía normalizada (1 = perfectamente balanceado)
    """
    s = pd.Series(etiquetas).dropna()
    conteo = s.value_counts()
    n = len(s)
    proporciones = conteo / n

    razon_imb = float(conteo.max() / conteo.min()) if conteo.min() > 0 else np.inf
    entropia_norm = float(stats.entropy(proporciones) / np.log(len(conteo))) \
        if len(conteo) > 1 else 1.0

    return {
        "n_clases": int(len(conteo)),
        "n_muestras": int(n),
        "razon_imbalance": np.round(razon_imb, 3),
        "entropia_normalizada": np.round(entropia_norm, 4),
        "clase_max": str(conteo.idxmax()),
        "clase_min": str(conteo.idxmin()),
        "n_max": int(conteo.max()),
        "n_min": int(conteo.min()),
        "conteo": conteo,
    }


# ---------------------------------------------------------------------------
# 6.  Pipeline integral por dataset
# ---------------------------------------------------------------------------

def reporte_estadistico(df_numerico, etiquetas=None, nombres_vars=None):
    """
    Aplica el pipeline completo a un DataFrame de variables numéricas y a un
    vector de etiquetas. Devuelve tres DataFrames: descriptiva, normalidad,
    outliers; y un dict con balance de clases.
    """
    if nombres_vars is None:
        nombres_vars = list(df_numerico.columns)

    desc, norm, out = [], [], []
    for v in nombres_vars:
        desc.append(descriptiva(df_numerico[v], nombre=v))
        ic = ic_95(df_numerico[v])
        d = desc[-1].copy()
        d["ic95_inf"], d["ic95_sup"] = ic
        desc[-1] = d

        n = prueba_normalidad(df_numerico[v])
        n["variable"] = v
        norm.append(pd.DataFrame([n]))

        o = detectar_outliers_iqr(df_numerico[v])
        o["variable"] = v
        out.append(pd.DataFrame([o]))

    desc_df = pd.concat(desc, ignore_index=True)
    norm_df = pd.concat(norm, ignore_index=True)
    out_df  = pd.concat(out, ignore_index=True)

    bal = balance_clases(etiquetas) if etiquetas is not None else None
    return desc_df, norm_df, out_df, bal


# ---------------------------------------------------------------------------
# 7.  Exportación a LaTeX
# ---------------------------------------------------------------------------

def df_a_latex(df, label, caption, columnas=None, float_format="%.3f"):
    """Convierte un DataFrame en una tabla booktabs lista para insertar."""
    if columnas:
        df = df[columnas]
    return df.to_latex(
        index=False, escape=True, label=label, caption=caption,
        float_format=float_format, column_format="l" + "r" * (df.shape[1] - 1),
    )

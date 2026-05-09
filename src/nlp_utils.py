"""
nlp_utils.py
============
Utilidades para EDA de corpus textuales (D5: Spanish-to-MSL glosses corpus).

Cumple los requerimientos del 5.3 aplicados a NLP:
    - Distribución de longitudes de oración (tokens, caracteres)
    - Frecuencia y diversidad de vocabulario (type-token ratio)
    - Distribución por categorías gramaticales / dominios
    - Análisis de paralelismo SPA↔MSL (correlación de longitudes)
    - Embeddings TF-IDF para t-SNE
    - Métricas de complejidad lingüística

Autor: Valeria Aguilar Meza · FIAD-UABC · 2026-1
"""

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.  Tokenización ligera (no requiere NLTK)
# ---------------------------------------------------------------------------

def normalizar_texto(s):
    """Normalización Unicode + minúsculas + colapso de espacios."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenizar(s, mantener_puntuacion=False):
    """
    Tokenizador ligero por espacios y signos de puntuación.
    Devuelve una lista de tokens. Para texto en español y glosas LSM.
    """
    if not isinstance(s, str):
        return []
    if mantener_puntuacion:
        return re.findall(r"\w+|[^\w\s]", s, flags=re.UNICODE)
    return re.findall(r"\w+", s, flags=re.UNICODE)


# ---------------------------------------------------------------------------
# 2.  Métricas por oración
# ---------------------------------------------------------------------------

def metricas_oracion(s):
    """Devuelve dict con n_caracteres, n_palabras, n_tokens_unicos, longitud_media_palabra."""
    s_norm = normalizar_texto(s)
    tokens = tokenizar(s_norm)
    if not tokens:
        return {"n_caracteres": 0, "n_palabras": 0, "n_tokens_unicos": 0,
                "longitud_media_palabra": 0.0, "ratio_diversidad": 0.0}
    return {
        "n_caracteres": len(s_norm),
        "n_palabras": len(tokens),
        "n_tokens_unicos": len(set(tokens)),
        "longitud_media_palabra": float(np.mean([len(t) for t in tokens])),
        "ratio_diversidad": len(set(tokens)) / len(tokens),
    }


# ---------------------------------------------------------------------------
# 3.  Vocabulario y type-token ratio del corpus
# ---------------------------------------------------------------------------

def metricas_corpus(serie_textos):
    """Métricas globales: vocabulario, TTR, frecuencias, hapax legomena."""
    todos_tokens = []
    for s in serie_textos.dropna():
        todos_tokens.extend(tokenizar(normalizar_texto(s)))
    if not todos_tokens:
        return {}
    cnt = Counter(todos_tokens)
    n_tokens = len(todos_tokens)
    n_types = len(cnt)
    hapax = sum(1 for v in cnt.values() if v == 1)
    return {
        "n_oraciones": len(serie_textos.dropna()),
        "n_tokens_total": n_tokens,
        "n_tipos_unicos": n_types,
        "type_token_ratio": n_types / n_tokens,  # diversidad léxica
        "n_hapax_legomena": hapax,                # palabras con frec 1
        "pct_hapax": 100 * hapax / n_types,
        "frecuencia": cnt,
    }


# ---------------------------------------------------------------------------
# 4.  Detección automática de columnas SPA y MSL
# ---------------------------------------------------------------------------

def detectar_columnas_paralelas(df):
    """
    Heurística para identificar cuál columna es español y cuál es la glosa MSL.
    Indicios usados:
        - español tiene tildes y artículos (la, el, los, las)
        - glosa MSL típicamente está en MAYÚSCULAS y sin artículos
        - longitud media de la glosa MSL es menor (compresión SOV)
    """
    indicios = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        texto_total = " ".join(sample)
        pct_mayus = sum(1 for c in texto_total if c.isupper()) / max(1, len(texto_total))
        pct_articulos = sum(1 for s in sample if any(
            f" {a} " in f" {s.lower()} " for a in ["la", "el", "los", "las", "un", "una"])) / len(sample)
        long_media = np.mean([len(tokenizar(s)) for s in sample])
        indicios.append({
            "columna": col,
            "pct_mayusculas": pct_mayus,
            "pct_articulos_es": pct_articulos,
            "long_media_tokens": long_media,
            "n_no_nulos": int(df[col].notna().sum()),
        })
    return pd.DataFrame(indicios)


# ---------------------------------------------------------------------------
# 5.  Análisis n-gramas (bigramas, trigramas)
# ---------------------------------------------------------------------------

def n_gramas(textos, n=2, top_k=20):
    """Top-k n-gramas más frecuentes en el corpus."""
    cnt = Counter()
    for s in textos.dropna():
        toks = tokenizar(normalizar_texto(s))
        for i in range(len(toks) - n + 1):
            cnt[tuple(toks[i:i + n])] += 1
    return cnt.most_common(top_k)


# ---------------------------------------------------------------------------
# 6.  Embedding TF-IDF para t-SNE
# ---------------------------------------------------------------------------

def embedding_tfidf(textos, max_features=500):
    """Devuelve matriz TF-IDF densa, lista para t-SNE."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer(max_features=max_features, lowercase=True,
                           token_pattern=r"\b\w+\b")
    X = vect.fit_transform([normalizar_texto(s) for s in textos.fillna("")])
    return X.toarray(), vect.get_feature_names_out()

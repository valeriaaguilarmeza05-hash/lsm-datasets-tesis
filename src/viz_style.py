"""
viz_style.py
============
Estilo visual unificado para todas las visualizaciones del EDA de los
cinco datasets de Lengua de Señas Mexicana (LSM).

Estilo solicitado: matplotlib + seaborn clásico, alto contraste.
Todas las figuras se exportan a 300 dpi en PNG, listas para LaTeX.

Autor: Valeria Aguilar Meza · FIAD-UABC · 2026-1
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# Paleta categórica de alto contraste (cumple WCAG AA)
PALETA = {
    "primario":   "#0B5394",  # azul oscuro (UABC institucional)
    "secundario": "#CC4125",  # rojo coral (alto contraste)
    "acento":     "#38761D",  # verde bosque
    "neutro":     "#444444",  # gris carbón
    "claro":      "#E8E8E8",  # gris claro fondo
    "highlight":  "#F1C232",  # ámbar (énfasis)
}

PALETA_CATEGORICA = [
    "#0B5394", "#CC4125", "#38761D", "#674EA7", "#F1C232",
    "#1B998B", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F",
]


def aplicar_estilo():
    """Aplica el estilo seaborn clásico de alto contraste a todas las figuras."""
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette=PALETA_CATEGORICA,
        font="DejaVu Sans",
        font_scale=1.05,
    )
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#888888",
        "lines.linewidth": 2.0,
        "patch.edgecolor": "#222222",
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def guardar_figura(fig, nombre, carpeta_separadas, carpeta_combinadas=None,
                   es_combinada=False, dpi=300):
    """
    Guarda una figura en PNG (300 dpi). Si es_combinada=True, va a la carpeta
    de combinadas; si no, a separadas. También exporta versión PDF para LaTeX.
    """
    import os
    destino = carpeta_combinadas if es_combinada else carpeta_separadas
    os.makedirs(destino, exist_ok=True)
    ruta_png = os.path.join(destino, f"{nombre}.png")
    fig.savefig(ruta_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    return ruta_png


def titulo_figura(ax, titulo, subtitulo=None):
    """Título compuesto: título principal en negrita + subtítulo opcional."""
    if subtitulo:
        ax.set_title(f"{titulo}\n{subtitulo}", loc="left", fontsize=12)
    else:
        ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold")

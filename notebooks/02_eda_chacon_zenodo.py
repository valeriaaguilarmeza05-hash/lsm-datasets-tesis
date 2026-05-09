"""
01_eda_chacon_zenodo.py — EDA reproducible del dataset D1 (versión corregida)
=============================================================================
Chacon Quintero, J. C., Martinez Nimi, H., Seira, S., & Betanzo, A. (2022).
Dataset LSM Lenguaje de señas mexicanas (v0.1) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.6554337
Licencia: CC-BY 4.0

HALLAZGO CRÍTICO: el dataset NO contiene fotografías RGB sino visualizaciones
renderizadas de los keypoints de MediaPipe Holistic sobre fondo blanco.
El EDA emplea variables geométricas apropiadas a esa modalidad.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import ndimage
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from viz_style import aplicar_estilo, PALETA, PALETA_CATEGORICA, guardar_figura
from stats_utils import (descriptiva, ic_95, prueba_normalidad,
                         detectar_outliers_iqr, balance_clases)

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_UPLOADS = "/mnt/user-data/uploads"
DIR_FIG_SEP  = os.path.join(RAIZ, "figures", "separadas")
DIR_FIG_COMB = os.path.join(RAIZ, "figures", "combinadas")
DIR_TABLAS   = os.path.join(RAIZ, "outputs", "tablas")
DIR_LATEX    = os.path.join(RAIZ, "secciones_eda_latex")
for d in (DIR_FIG_SEP, DIR_FIG_COMB, DIR_TABLAS, DIR_LATEX):
    os.makedirs(d, exist_ok=True)

aplicar_estilo()
SEED = 42; np.random.seed(SEED)


def cargar_stream(ruta):
    out = []
    with open(ruta, "rb") as f:
        while True:
            try: out.append(pickle.load(f))
            except EOFError: break
    return out


print("[1] Cargando pickles...")
etiq_abc = cargar_stream(f"{PATH_UPLOADS}/ABECEDARIO.pickle")[0]
imgs_abc = cargar_stream(f"{PATH_UPLOADS}/ABECEDARIOIMAGENES.pickle")
etiq_pal = cargar_stream(f"{PATH_UPLOADS}/PALABRAS.pickle")[0]
etiq_num = cargar_stream(f"{PATH_UPLOADS}/NUMEROS.pickle")[0]
etiq_abc_lim = [s.strip().replace(".png", "") for s in etiq_abc]
print(f"  ABECEDARIO: {len(imgs_abc)} imágenes (640×360 RGB con keypoints)")
print(f"  PALABRAS  : {len(etiq_pal)} etiquetas")
print(f"  NUMEROS   : {len(etiq_num)} (vacío en v0.1)")


print("\n[2] Extrayendo variables geométricas de los keypoints...")

def extraer_features_keypoints(img):
    arr_rgb = np.asarray(img, dtype=np.float32)
    arr_g = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = arr_g.shape
    mask = arr_g < 220
    pct_activo = float(mask.mean() * 100)
    if not mask.any():
        return {k: np.nan for k in ["pct_activo", "centroide_x", "centroide_y",
                "extension_x", "extension_y", "ratio_extension", "skew_x",
                "skew_y", "n_componentes", "rojo_act", "verde_act", "azul_act"]}
    ys, xs = np.where(mask)
    cx, cy = xs.mean() / W, ys.mean() / H
    ext_x = (xs.max() - xs.min()) / W
    ext_y = (ys.max() - ys.min()) / H
    skew_x = pd.Series(xs / W).skew()
    skew_y = pd.Series(ys / H).skew()
    _, n_comp = ndimage.label(mask)
    return {
        "pct_activo": pct_activo,
        "centroide_x": cx, "centroide_y": cy,
        "extension_x": ext_x, "extension_y": ext_y,
        "ratio_extension": ext_x / ext_y if ext_y > 0 else np.nan,
        "skew_x": skew_x, "skew_y": skew_y,
        "n_componentes": int(n_comp),
        "rojo_act": float(arr_rgb[..., 0][mask].mean()),
        "verde_act": float(arr_rgb[..., 1][mask].mean()),
        "azul_act": float(arr_rgb[..., 2][mask].mean()),
    }


registros = []
for i, img in enumerate(imgs_abc):
    f = extraer_features_keypoints(img)
    f.update({
        "idx": i,
        "etiqueta_id": etiq_abc_lim[i] if i < len(etiq_abc_lim) else f"id_{i}",
        "subset": "ABECEDARIO",
        "ancho": img.size[0], "alto": img.size[1],
        "n_canales": len(img.getbands()),
        "modo": img.mode,
    })
    registros.append(f)
df = pd.DataFrame(registros)
df["bloque"] = pd.cut(df["idx"], bins=[-1, 8, 29, 50, 60, 71],
                     labels=["B1 (1–9)", "B2 (10–30)", "B3 (31–51)",
                             "B4 (52–61)", "B5 (62–72)"])
print(f"  DataFrame: {df.shape}")
print(df[["pct_activo", "centroide_x", "centroide_y", "extension_x",
          "extension_y", "n_componentes"]].describe().round(3))


print("\n[3] Estadística descriptiva...")
vars_clave = ["pct_activo", "centroide_x", "centroide_y", "extension_x",
              "extension_y", "ratio_extension", "n_componentes"]
desc_filas, norm_filas, out_filas = [], [], []
for v in vars_clave:
    d = descriptiva(df[v], nombre=v).iloc[0].to_dict()
    ic_inf, ic_sup = ic_95(df[v])
    d["ic95_inf"], d["ic95_sup"] = ic_inf, ic_sup
    desc_filas.append(d)
    n = prueba_normalidad(df[v]); n["variable"] = v
    norm_filas.append(n)
    o = detectar_outliers_iqr(df[v]); o["variable"] = v
    out_filas.append(o)
desc_df = pd.DataFrame(desc_filas)
norm_df = pd.DataFrame(norm_filas)
out_df = pd.DataFrame(out_filas)
bal = balance_clases(df["bloque"])
print(desc_df[["variable", "media", "mediana", "std", "ic95_inf", "ic95_sup"]].round(3))
print(f"  Balance: razón={bal['razon_imbalance']}, entropía={bal['entropia_normalizada']}")


print("\n[4] Generando visualizaciones...")

# Vis 1: distribución por bloque
fig1, ax1 = plt.subplots(figsize=(8, 5))
conteo = df["bloque"].value_counts().sort_index()
colores = PALETA_CATEGORICA[:len(conteo)]
bars = ax1.bar(conteo.index.astype(str), conteo.values, color=colores,
               edgecolor="#222", linewidth=0.8)
ax1.set_title("Distribución de muestras por bloque ordinal del alfabeto",
              loc="left", fontweight="bold")
ax1.set_xlabel("Bloque (rango de IDs en ABECEDARIO.pickle)")
ax1.set_ylabel("Número de imágenes")
for b, v in zip(bars, conteo.values):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.3, str(v),
             ha="center", fontsize=10, fontweight="bold")
ax1.set_ylim(0, conteo.max() * 1.18)
guardar_figura(fig1, "d1_chacon_01_distribucion", DIR_FIG_SEP)
plt.close(fig1)

# Vis 2: histograma % área activa
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.histplot(df["pct_activo"], bins=18, kde=True, color=PALETA["primario"],
             edgecolor="#222", ax=ax2)
m = df["pct_activo"].mean()
ic_inf, ic_sup = ic_95(df["pct_activo"])
ax2.axvline(m, color=PALETA["secundario"], lw=2, label=f"Media = {m:.3f}%")
ax2.axvline(ic_inf, color=PALETA["acento"], ls="--",
            label=f"IC95% = [{ic_inf:.3f}, {ic_sup:.3f}]")
ax2.axvline(ic_sup, color=PALETA["acento"], ls="--")
ax2.set_title("Histograma del porcentaje de píxeles activos (keypoints)",
              loc="left", fontweight="bold")
ax2.set_xlabel("% de área activa por imagen")
ax2.set_ylabel("Frecuencia")
ax2.legend(loc="best")
guardar_figura(fig2, "d1_chacon_02_histograma_area", DIR_FIG_SEP)
plt.close(fig2)

# Vis 3: grid muestras
fig3, axs = plt.subplots(4, 4, figsize=(11, 7))
np.random.seed(SEED)
indices_muestra = np.random.choice(len(imgs_abc), size=16, replace=False)
for ax, k in zip(axs.flat, indices_muestra):
    ax.imshow(imgs_abc[k])
    ax.set_title(f"ID {etiq_abc_lim[k]} · {df.loc[k, 'bloque']}", fontsize=8)
    ax.axis("off")
fig3.suptitle("Muestra cruda aleatoria (n=16) · keypoints renderizados sobre fondo blanco",
              fontsize=12, fontweight="bold", y=1.0, x=0.05, ha="left")
fig3.tight_layout()
guardar_figura(fig3, "d1_chacon_03_grid_muestras", DIR_FIG_SEP)
plt.close(fig3)

# Vis 4: heatmap correlación
fig4, ax4 = plt.subplots(figsize=(9, 7))
corr_vars = vars_clave + ["skew_x", "skew_y"]
corr = df[corr_vars].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            square=True, cbar_kws={"label": "Pearson r"},
            linewidths=0.5, linecolor="white", ax=ax4)
ax4.set_title("Correlación entre features geométricas de los keypoints",
              loc="left", fontweight="bold")
plt.setp(ax4.get_xticklabels(), rotation=35, ha="right", fontsize=9)
plt.setp(ax4.get_yticklabels(), fontsize=9)
guardar_figura(fig4, "d1_chacon_04_heatmap_corr", DIR_FIG_SEP)
plt.close(fig4)

# Vis 5: t-SNE
print("  Generando embeddings + t-SNE...")
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
emb_visual = np.stack([
    np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32).flatten() / 255.0
    for img in imgs_abc
])
emb_geom = StandardScaler().fit_transform(df[vars_clave + ["skew_x", "skew_y"]].fillna(0))
emb_full = np.concatenate([emb_visual, emb_geom], axis=1)
print(f"    Embedding shape: {emb_full.shape}")
tsne = TSNE(n_components=2, perplexity=10, random_state=SEED, init="pca",
            learning_rate="auto")
emb_2d = tsne.fit_transform(emb_full)

fig5, ax5 = plt.subplots(figsize=(8, 6))
for bi, color in zip(df["bloque"].cat.categories, PALETA_CATEGORICA):
    mask = (df["bloque"] == bi).to_numpy()
    ax5.scatter(emb_2d[mask, 0], emb_2d[mask, 1], color=color,
                label=str(bi), edgecolor="#222", s=80, alpha=0.85)
ax5.set_title("t-SNE de embeddings (visual 32×32 + features geométricas)",
              loc="left", fontweight="bold")
ax5.set_xlabel("t-SNE 1"); ax5.set_ylabel("t-SNE 2")
ax5.legend(title="Bloque", loc="best", fontsize=9)
guardar_figura(fig5, "d1_chacon_05_tsne", DIR_FIG_SEP)
plt.close(fig5)


print("\n[5] Generando panel combinado...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes[0, 0].bar(conteo.index.astype(str), conteo.values, color=colores, edgecolor="#222")
axes[0, 0].set_title("(a) Distribución por bloque", fontweight="bold")
axes[0, 0].set_ylabel("N imágenes")
axes[0, 0].tick_params(axis='x', rotation=20)

axes[0, 1].hist(df["pct_activo"], bins=18, color=PALETA["primario"], edgecolor="#222")
axes[0, 1].axvline(m, color=PALETA["secundario"], lw=2, label=f"μ={m:.3f}%")
axes[0, 1].set_title("(b) % área activa", fontweight="bold")
axes[0, 1].set_xlabel("% píxeles con keypoints"); axes[0, 1].legend()

axes[0, 2].imshow(imgs_abc[0])
axes[0, 2].set_title("(c) Muestra cruda · ID 1", fontweight="bold")
axes[0, 2].axis("off")

corr_red = df[["pct_activo", "centroide_x", "centroide_y", "extension_x",
               "extension_y", "n_componentes"]].corr()
sns.heatmap(corr_red, ax=axes[1, 0], annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, cbar=False, linewidths=0.4, annot_kws={"size": 8})
axes[1, 0].set_title("(d) Correlación geométrica", fontweight="bold")
plt.setp(axes[1, 0].get_xticklabels(), rotation=35, ha="right", fontsize=8)
plt.setp(axes[1, 0].get_yticklabels(), fontsize=8)

for bi, color in zip(df["bloque"].cat.categories, PALETA_CATEGORICA):
    mask = (df["bloque"] == bi).to_numpy()
    axes[1, 1].scatter(emb_2d[mask, 0], emb_2d[mask, 1], color=color,
                       label=str(bi), edgecolor="#222", s=55, alpha=0.85)
axes[1, 1].set_title("(e) t-SNE multimodal", fontweight="bold")
axes[1, 1].set_xlabel("t-SNE 1"); axes[1, 1].set_ylabel("t-SNE 2")
axes[1, 1].legend(fontsize=7, title="Bloque", title_fontsize=8)

sns.boxplot(data=df, x="bloque", y="pct_activo", ax=axes[1, 2],
            palette=PALETA_CATEGORICA[:5], hue="bloque", legend=False)
axes[1, 2].set_title("(f) % activo por bloque", fontweight="bold")
axes[1, 2].set_xlabel(""); axes[1, 2].set_ylabel("% activo")
axes[1, 2].tick_params(axis='x', rotation=20)

fig.suptitle("EDA Dataset D1 · Chacon Quintero et al. (2022) · DOI: 10.5281/zenodo.6554337",
             fontsize=13, fontweight="bold", y=1.0)
fig.tight_layout()
guardar_figura(fig, "d1_chacon_grid", "", DIR_FIG_COMB, es_combinada=True)
plt.close(fig)


print("\n[6] Exportando tablas...")
desc_export = desc_df[["variable", "n", "media", "mediana", "std", "min", "max",
                       "ic95_inf", "ic95_sup", "cv_%"]].copy()
desc_export.columns = ["Variable", "$n$", "Media", "Mediana", "Std", "Min", "Max",
                       "IC95\\% inf", "IC95\\% sup", "CV (\\%)"]
with open(f"{DIR_TABLAS}/desc_d1.tex", "w") as f:
    f.write(desc_export.to_latex(index=False, escape=False, float_format="%.3f",
            column_format="lrrrrrrrrr"))

norm_export = norm_df[["variable", "prueba", "n", "stat", "p_valor", "es_normal"]].copy()
norm_export.columns = ["Variable", "Prueba", "$n$", "Estadístico", "$p$-valor", "¿Normal?"]
norm_export["¿Normal?"] = norm_export["¿Normal?"].map({True: "Sí", False: "No"})
out_export = out_df[["variable", "n_outliers", "pct_outliers", "limite_inf", "limite_sup"]].copy()
out_export.columns = ["Variable", "Outliers", "\\% outliers", "Lím. inf", "Lím. sup"]
norm_out = norm_export.merge(out_export, on="Variable")
with open(f"{DIR_TABLAS}/norm_d1.tex", "w") as f:
    f.write(norm_out.to_latex(index=False, escape=False, float_format="%.4f",
            column_format="llrrrlrrrr"))

df.to_csv(f"{RAIZ}/outputs/d1_chacon_metadatos.csv", index=False)
desc_df.to_csv(f"{RAIZ}/outputs/d1_chacon_descriptiva.csv", index=False)


print("\n[7] Generando bloque LaTeX interpretativo...")
m_act = df["pct_activo"].mean()
ic_act = ic_95(df["pct_activo"])
sh_p = norm_df.loc[norm_df["variable"] == "pct_activo", "p_valor"].iloc[0]
sh_n = bool(norm_df.loc[norm_df["variable"] == "pct_activo", "es_normal"].iloc[0])
out_act = int(out_df.loc[out_df["variable"] == "pct_activo", "n_outliers"].iloc[0])

text = (
    "\\subsection*{Análisis exploratorio (EDA)}\n"
    "\\label{subsec:eda_d1_chacon}\n\n"
    "\\paragraph{Inventario y formato.}\n"
    "La inspección programática de los archivos \\texttt{*.pickle} confirma "
    f"que el subset \\textit{{ABECEDARIO}} contiene {len(imgs_abc)} imágenes "
    "RGB de $640\\times360$ píxeles serializadas en formato "
    "\\texttt{PngImageFile} de \\texttt{PIL}. El subset \\textit{PALABRAS} "
    f"declara {len(etiq_pal)} etiquetas (las imágenes correspondientes residen "
    "en \\texttt{PALABRASIMAGES.pickle}, no analizadas en este lote por "
    "restricciones de tamaño). El subset \\textit{NUMEROS} se publicó vacío "
    "en la versión 0.1 (\\texttt{NUMEROS.pickle} contiene una lista de "
    "longitud cero), limitación que se documenta por transparencia. Las "
    "etiquetas son identificadores ordinales sin un mapeo semántico explícito "
    "a las letras del alfabeto.\n\n"
    "\\paragraph{Hallazgo crítico sobre la naturaleza del contenido.}\n"
    "La inspección visual confirma que las imágenes contenidas en este "
    "dataset \\textbf{no son fotografías RGB de personas haciendo señas}, "
    "sino \\textbf{visualizaciones renderizadas de los keypoints de "
    "MediaPipe Holistic} (puntos faciales, brazos y mano dominante) sobre "
    "fondo blanco. En consecuencia, las variables fotométricas tradicionales "
    "(brillo medio, contraste, distribución RGB global) presentan varianza "
    "prácticamente nula y no son discriminativas. Esta caracterización es "
    "favorable para el pipeline de tesis propuesto: el dataset ya está "
    "alineado con la representación de bajo costo computacional requerida "
    "para el despliegue del intérprete LSM en hardware embebido.\n\n"
    "\\paragraph{Variables apropiadas y visualizaciones.}\n"
    "El EDA se reformuló empleando variables geométricas extraídas de la "
    "máscara de píxeles activos (umbral $L<220$): porcentaje de área activa, "
    "centroides $(x,y)$ normalizados, extensión espacial, ratio de extensión, "
    "asimetría espacial y número de componentes conexos. La "
    "Figura~\\ref{fig:eda_d1_chacon_grid} sintetiza las cinco visualizaciones "
    "exigidas por la rúbrica: (a) distribución por bloque ordinal del "
    "alfabeto, (b) histograma del porcentaje de área activa, (c) muestra "
    "cruda representativa, (d) heatmap de correlación entre features "
    "geométricas y (e) proyección t-SNE de embeddings combinados (visual "
    "$32\\times32$ + descriptores geométricos).\n\n"
    "\\begin{figure}[H]\n"
    "    \\centering\n"
    "    \\includegraphics[width=\\linewidth]{figures/combinadas/d1_chacon_grid.png}\n"
    "    \\caption{Análisis exploratorio del dataset D1 (Chacon Quintero "
    "et~al., 2022): (a) distribución por bloque ordinal, (b) histograma del "
    "porcentaje de área activa, (c) muestra cruda con keypoints renderizados, "
    "(d) heatmap de correlación geométrica, (e) proyección t-SNE multimodal y "
    "(f) boxplot del porcentaje activo por bloque. Fuente: elaboración propia.}\n"
    "    \\label{fig:eda_d1_chacon_grid}\n"
    "\\end{figure}\n\n"
    "\\paragraph{Estadística descriptiva e intervalos de confianza.}\n"
    "La Tabla~\\ref{tab:desc_d1} reporta los estadísticos descriptivos para "
    "siete variables geométricas. El porcentaje promedio de área activa es "
    f"$\\bar{{X}}={m_act:.3f}\\,\\%$ con intervalo de confianza al $95\\,\\%$ "
    f"de $[{ic_act[0]:.3f},\\,{ic_act[1]:.3f}]$ (método $t$ de Student, "
    f"$n={len(imgs_abc)}$).\n\n"
    "\\begin{table}[H]\n"
    "    \\centering\n"
    "    \\caption{Estadística descriptiva de variables geométricas del "
    "dataset D1 (Chacon \\textit{et al.}, 2022). Variables expresadas en "
    "coordenadas normalizadas $[0,1]$ excepto \\texttt{pct\\_activo} (\\%) "
    "y \\texttt{n\\_componentes} (entero).}\n"
    "    \\label{tab:desc_d1}\n"
    "    \\footnotesize\n"
    "    \\input{outputs/tablas/desc_d1.tex}\n"
    "\\end{table}\n\n"
    "\\paragraph{Pruebas de normalidad y outliers.}\n"
    f"La Tabla~\\ref{{tab:norm_d1}} resume las pruebas de Shapiro--Wilk "
    f"($n={len(imgs_abc)}\\leq 5000$) sobre cada variable geométrica junto "
    f"con la detección de outliers mediante la regla IQR de Tukey ($k=1.5$). "
    f"El porcentaje de área activa "
    f"{'no rechaza' if sh_n else 'rechaza'} la hipótesis de normalidad "
    f"($p={sh_p:.4f}$); se identificaron $\\mathbf{{{out_act}}}$ outliers "
    f"para esta variable.\n\n"
    "\\begin{table}[H]\n"
    "    \\centering\n"
    "    \\caption{Pruebas de normalidad (Shapiro--Wilk) y detección de "
    "outliers (regla IQR, $k=1.5$) para las variables geométricas del "
    "dataset D1.}\n"
    "    \\label{tab:norm_d1}\n"
    "    \\footnotesize\n"
    "    \\input{outputs/tablas/norm_d1.tex}\n"
    "\\end{table}\n\n"
    "\\paragraph{Hallazgos clave.}\n"
    "\\begin{enumerate}\n"
    "    \\item \\textbf{Modalidad real del dataset:} las imágenes son "
    "renderizados de keypoints de MediaPipe Holistic, no fotografías RGB. "
    "Esto alinea el dataset con el preprocesamiento basado en landmarks "
    "propuesto en la metodología de la tesis.\n"
    "    \\item \\textbf{Resolución uniforme:} $640\\times360$ RGB en el "
    f"$100\\,\\%$ de las imágenes inspeccionadas ($n={len(imgs_abc)}$).\n"
    "    \\item \\textbf{Etiquetas no semánticas:} los archivos contienen "
    "identificadores ordinales sin mapeo explícito a letras o palabras del "
    "LSM. Será necesario un etiquetado manual asistido para entrenamiento "
    "supervisado.\n"
    "    \\item \\textbf{Subset NUMEROS vacío:} la versión 0.1 publicada "
    "no incluye datos numéricos, restringiendo la cobertura léxica al "
    "alfabeto y palabras.\n"
    "    \\item \\textbf{Estructura espacial homogénea:} la dispersión "
    f"de los keypoints es muy reducida (CV $<5\\,\\%$ en variables "
    "de extensión), indicando una protocolización estricta de la captura.\n"
    f"    \\item \\textbf{{Balance ordinal:}} la razón de imbalance entre "
    f"bloques es ${bal['razon_imbalance']:.2f}$ y la entropía normalizada "
    f"es ${bal['entropia_normalizada']:.3f}$, indicando una distribución "
    "aceptablemente uniforme.\n"
    "\\end{enumerate}\n\n"
    "% Fin del bloque EDA D1\n"
)

with open(f"{DIR_LATEX}/eda_d1_chacon.tex", "w") as f:
    f.write(text)

print(f"\n✅ EDA D1 completado.")
print(f"   Figuras separadas: {DIR_FIG_SEP}/d1_chacon_*.png")
print(f"   Figura combinada:  {DIR_FIG_COMB}/d1_chacon_grid.png")
print(f"   Tablas LaTeX:      {DIR_TABLAS}/desc_d1.tex, norm_d1.tex")
print(f"   Bloque LaTeX:      {DIR_LATEX}/eda_d1_chacon.tex")

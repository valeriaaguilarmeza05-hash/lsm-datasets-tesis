import cv2
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import zipfile

# --- DIAGNÓSTICO DE RUTA ---
ruta_donde_estoy = os.path.dirname(os.path.abspath(__file__))
print(f"1. El script se está ejecutando en: {ruta_donde_estoy}")
print(f"2. Contenido de esta carpeta: {os.listdir(ruta_donde_estoy)}")

datos = []

# --- BÚSQUEDA PROFUNDA ---
print("\n3. Iniciando búsqueda de videos .mp4...")
for raiz, carpetas, archivos in os.walk(ruta_donde_estoy):
    for archivo in archivos:
        if archivo.lower().endswith('.mp4'):
            ruta_video = os.path.join(raiz, archivo)
            print(f"   -> ¡Encontrado!: {archivo} en {os.path.basename(raiz)}")
            
            cap = cv2.VideoCapture(ruta_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            if fps > 0 and frames > 0:
                duracion = frames / fps
                letra_carpeta = os.path.basename(raiz).upper()
                datos.append({
                    'Letra': letra_carpeta,
                    'FPS': fps,
                    'Duracion': duracion
                })
            cap.release()

# --- PROCESAMIENTO DE RESULTADOS ---
if not datos:
    print("\nERROR: Python recorrió las carpetas pero NO leyó ningún archivo .mp4.")
    print("Verifica que los archivos no pesen 0 KB y que terminen en .mp4")
else:
    df = pd.DataFrame(datos)
    print(f"\nÉXITO: Se analizaron {len(df)} videos.")
    
    # Gráficas
    plt.figure(figsize=(8,5))
    sns.histplot(df['Duracion'], kde=True, color='green')
    plt.title('EDA: Duración de Señas Dinámicas (J, K, Ñ)')
    plt.xlabel('Segundos')
    plt.savefig('d4_dinamico_v1.png')
    
    # Estadística
    p_val = stats.shapiro(df['Duracion'])[1]
    print(f"P-value Shapiro (Duración): {p_val:.8f}")
    print(f"Duración promedio: {df['Duracion'].mean():.2f} seg")
    
    with zipfile.ZipFile('EDA_Dinamico_Final.zip', 'w') as z:
        z.write('d4_dinamico_v1.png')
    print("\nZIP 'EDA_Dinamico_Final.zip' creado en tu carpeta.")
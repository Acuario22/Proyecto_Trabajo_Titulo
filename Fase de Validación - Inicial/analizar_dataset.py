# 03_analizar_dataset.py
# OBJETIVO
# Diagnosticar la calidad y consistencia del dataset sintético generado:
#   1) Balance de clases
#   2) Auditoría léxica (fuga de vocabulario entre categorías)
#   3) Entrenamiento de un clasificador simple (TF-IDF + Regresión Logística)
#   4) Matriz de confusión (absoluta y normalizada)
#   5) Inspección de errores de clasificación

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_entrada = BASE_DIR / "1.Fase de generación" / "dataset_reportes_sinteticos.csv"
total_esperado = 2000

# Vocabulario que NO debería aparecer en cada categoría
# (si aparece, es señal de "fuga léxica" entre categorías)
vocabulario_prohibido = {
    "Pavimentación": ["poste de luz", "farol", "tapa de alcantarillado", "cañería",
                       "fuga de agua", "rejilla", "alcantarilla"],
    "Alcantarillado": ["bache", "hoyo en la calzada", "poste de luz", "farol",
                        "vereda dañada", "acera dañada"],
    "Iluminación": ["bache", "hoyo en la calzada", "tapa de alcantarillado", "cañería",
                     "vereda dañada", "acera dañada", "alcantarilla"]
}


# ---------------------
# FUNCIÓN: AUDITORÍA LÉXICA
# ---------------------
def auditar_fugas_lexicas(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    filas_problema = []
    for _, row in df.iterrows():
        texto = str(row[columna_texto]).lower()
        categoria = row["Categoria"]
        prohibidas = vocabulario_prohibido.get(categoria, [])
        encontradas = [p for p in prohibidas if p in texto]
        if encontradas:
            filas_problema.append({
                "ID_Reporte": row.get("ID_Reporte", ""),
                "Categoria": categoria,
                "Palabras_filtradas": ", ".join(encontradas),
                "Texto_Reporte": row[columna_texto]
            })
    return pd.DataFrame(filas_problema)


# ---------------------
# CARGA DEL DATASET
# ---------------------
df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")

# Usamos Texto_Limpio si existe (post Fase 2), si no Texto_Reporte crudo
columna_texto = "Texto_Limpio" if "Texto_Limpio" in df.columns else "Texto_Reporte"

X = df[columna_texto]
y = df["Categoria"]

# ---------------------
# 1) BALANCE DE CLASES
# ---------------------
print("\n" + "=" * 60)
print("BALANCE DE CLASES")
print("=" * 60)
print(y.value_counts())
print(f"\nTotal de reportes en el dataset: {len(df)}")
if len(df) < total_esperado:
    print(f"⚠ Advertencia: el dataset tiene menos de los {total_esperado} reportes esperados. "
          f"El diagnóstico puede no ser representativo aún.")

# ---------------------
# 2) AUDITORÍA LÉXICA
# ---------------------
print("\n" + "=" * 60)
print("AUDITORÍA LÉXICA (fuga de vocabulario entre categorías)")
print("=" * 60)

df_fugas = auditar_fugas_lexicas(df, columna_texto)
total = len(df)
n_fugas = len(df_fugas)
porcentaje_fugas = (n_fugas / total * 100) if total > 0 else 0

print(f"Reportes con fuga léxica: {n_fugas}/{total} ({porcentaje_fugas:.1f}%)")

if n_fugas > 0:
    print("\nDistribución de fugas por categoría:")
    print(df_fugas["Categoria"].value_counts())
    df_fugas.to_csv("reportes_con_fuga_lexica.csv", index=False, encoding="utf-8-sig")
    print("\nGuardado detalle en 'reportes_con_fuga_lexica.csv'")
else:
    print("✅ No se encontraron fugas léxicas evidentes.")

# ---------------------
# 3) ENTRENAMIENTO DE CLASIFICADOR DIAGNÓSTICO
# ---------------------
# Dividimos en entrenamiento (80%) y prueba (20%)
# random_state=42 asegura que el split sea reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF: convierte texto a números (vectorización simple)
# Es suficiente para este diagnóstico
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenamos un clasificador simple
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_vec, y_train)

# Predicciones
y_pred = modelo.predict(X_test_vec)

# ---------------------
# REPORTE DE CLASIFICACIÓN
# ---------------------
print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACIÓN")
print("=" * 60)
print(classification_report(y_test, y_pred))

# ---------------------
# 4) MATRIZ DE CONFUSIÓN (ABSOLUTA)
# ---------------------
categorias_orden = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=categorias_orden)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categorias_orden,
    yticklabels=categorias_orden
)
plt.title("Matriz de Confusión — Diagnóstico de Calidad del Dataset")
plt.ylabel("Categoría Real")
plt.xlabel("Categoría Predicha")
plt.tight_layout()
plt.savefig("matriz_confusion_diagnostico.png", dpi=150)
plt.show()
print("\n✅ Imagen guardada como 'matriz_confusion_diagnostico.png'")

# ---------------------
# MATRIZ DE CONFUSIÓN NORMALIZADA (% por categoría real)
# ---------------------
cm_norm = confusion_matrix(y_test, y_pred, labels=categorias_orden, normalize="true")

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1%",
    cmap="Blues",
    xticklabels=categorias_orden,
    yticklabels=categorias_orden
)
plt.title("Matriz de Confusión Normalizada (% por categoría real)")
plt.ylabel("Categoría Real")
plt.xlabel("Categoría Predicha")
plt.tight_layout()
plt.savefig("matriz_confusion_normalizada.png", dpi=150)
plt.show()
print("✅ Imagen guardada como 'matriz_confusion_normalizada.png'")

# ---------------------
# 5) INSPECCIÓN DE ERRORES DE CLASIFICACIÓN
# ---------------------
df_test = pd.DataFrame({
    "Texto": X_test.values,
    "Real": y_test.values,
    "Predicho": y_pred
})
df_errores = df_test[df_test["Real"] != df_test["Predicho"]]

print("\n" + "=" * 60)
print("INSPECCIÓN DE ERRORES DE CLASIFICACIÓN")
print("=" * 60)
print(f"Errores: {len(df_errores)}/{len(df_test)} ({len(df_errores)/len(df_test)*100:.1f}%)")

df_errores.to_csv("errores_clasificador.csv", index=False, encoding="utf-8-sig")
print("Guardado detalle en 'errores_clasificador.csv'")
print("Revísalo para distinguir errores por fuga léxica real vs. ambigüedad natural del lenguaje.")

# ---------------------
# INTERPRETACIÓN FINAL
# ---------------------
print("\n" + "=" * 60)
print("INTERPRETACIÓN")
print("=" * 60)
print("  - Diagonal alta en la matriz = dataset consistente ✅")
print("  - Valores altos fuera de la diagonal = confusión entre categorías ⚠️")
print("  - % de fuga léxica alto (>5-10%) = conviene reforzar el prompt y/o filtrar ⚠️")
print("  - Cruza 'reportes_con_fuga_lexica.csv' y 'errores_clasificador.csv':")
print("    si los mismos ID_Reporte aparecen en ambos, el problema es del dataset,")
print("    no ambigüedad natural del lenguaje.")


if os.path.exists("reportes_con_fuga_lexica.csv"):
    fugas = pd.read_csv("reportes_con_fuga_lexica.csv", encoding="utf-8-sig")
    errores = pd.read_csv("errores_clasificador.csv", encoding="utf-8-sig")
    interseccion = fugas[fugas["Texto_Reporte"].isin(errores["Texto"])]
    print(f"Reportes con fuga léxica que también fueron mal clasificados: {len(interseccion)}")
else:
    print("No hay archivo de fugas léxicas (0% de fuga detectada) — no aplica el cruce.")
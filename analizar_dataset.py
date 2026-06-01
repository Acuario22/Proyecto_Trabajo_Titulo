# 03_analizar_dataset.py
# Entrena un clasificador simple y genera matriz de confusión.
# Esto mide la consistencia interna del dataset, que es
# mucho más informativo que la auditoría por palabras clave.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carga tu dataset
df = pd.read_csv("dataset_reportes_preprocesado.csv", encoding="utf-8-sig")

# Usamos Texto_Limpio si existe, si no Texto_Reporte
columna_texto = "Texto_Limpio" if "Texto_Limpio" in df.columns else "Texto_Reporte"

X = df[columna_texto]
y = df["Categoria"]

# Dividimos en entrenamiento (80%) y prueba (20%)
# random_state=42 asegura que el split sea reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF: convierte texto a números (vectorización simple)
# Es suficiente para este diagnóstico
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Entrenamos un clasificador simple
modelo = LogisticRegression(max_iter=1000, random_state=42)
modelo.fit(X_train_vec, y_train)

# Predicciones
y_pred = modelo.predict(X_test_vec)

# ---- REPORTE DE CLASIFICACIÓN ----
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN")
print("="*60)
print(classification_report(y_test, y_pred))

# ---- MATRIZ DE CONFUSIÓN ----
# Esta es la clave: muestra dónde se confunde el modelo.
# Si una categoría se confunde mucho con otra, hay ruido real.
categorias = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=categorias)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=categorias,
    yticklabels=categorias
)
plt.title("Matriz de Confusión — Diagnóstico de Calidad del Dataset")
plt.ylabel("Categoría Real")
plt.xlabel("Categoría Predicha")
plt.tight_layout()
plt.savefig("matriz_confusion_diagnostico.png", dpi=150)
plt.show()

print("\n✅ Imagen guardada como 'matriz_confusion_diagnostico.png'")
print("\nInterpretación:")
print("  - Diagonal alta = dataset consistente ✅")
print("  - Valores altos fuera de la diagonal = confusión entre categorías ⚠️")
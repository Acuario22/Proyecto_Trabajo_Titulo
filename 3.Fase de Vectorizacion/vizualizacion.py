from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent

archivo = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"

matriz = joblib.load(archivo)

# Imprime el tipo y la forma de la matriz cargada
print(type(matriz))
print(matriz.shape)



# Grafica de dispersión de la matriz TF-IDF
plt.figure(figsize=(8, 10))
# plt.spy dibuja solo los elementos no nulos de una matriz dispersa
plt.spy(matriz, markersize=0.5, color="navy")
plt.title(
    "Estructura de Dispersión de la Matriz TF-IDF (2000 x 3333)\n99.50% de ceros"
)
plt.xlabel("Índice de Términos (Vocabulario)")
plt.ylabel("Índice de Documentos (Reportes)")
plt.tight_layout()
plt.show()

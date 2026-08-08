from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

archivo = BASE_DIR / "3.Fase de Vectorizacion" / "matriz_tf_idf.joblib"

matriz = joblib.load(archivo)

print(type(matriz))
print(matriz.shape)
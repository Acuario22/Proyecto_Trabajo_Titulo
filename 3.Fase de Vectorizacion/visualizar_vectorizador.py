# Archivo con el objetivo de visualizar el vectorizador TF-IDF generado

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

archivo = BASE_DIR / "3.Fase de Vectorizacion" / "vectorizador_tf_idf.joblib"

vectorizador = joblib.load(archivo)

print(type(vectorizador))
print(vectorizador)

print(vectorizador.get_feature_names_out()[:20])
print(len(vectorizador.get_feature_names_out()))

idf = vectorizador.idf_

for termino, peso in zip(vectorizador.get_feature_names_out()[:20], idf[:20]):
    print(f"{termino:25} {peso:.3f}")
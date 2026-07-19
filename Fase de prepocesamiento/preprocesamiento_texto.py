# 02_preprocesamiento_texto.py
import pandas as pd
import re
import emoji
import unicodedata
from pathlib import Path 
BASE_DIR = Path(__file__).resolve().parent.parent

#ELIMINAR EMOJIS
def eliminar_emojis(texto):
    return emoji.replace_emoji(texto, replace="")


#LIMPIEZA DE TEXTO


def limpiar_texto(texto):

    if pd.isna(texto):
        return ""

    # Convertir a string y minúsculas
    texto = str(texto).lower()

    # Normalizar Unicode (evita acentos "rotos" o descompuestos)
    texto = unicodedata.normalize("NFC", texto)     # <-- agregar esta línea

    # Eliminar emojis
    texto = eliminar_emojis(texto)

    # Eliminar saltos de línea
    texto = texto.replace("\n", " ")

    # Eliminar caracteres de acentuación sueltos
    texto = texto.replace("´", "")
    texto = texto.replace("`", "")
    texto = texto.replace("¨", "")

    # Eliminar números
    texto = re.sub(r"\d+", "", texto)

    # Mantener solo letras españolas y espacios
    texto = re.sub(
        r"[^a-záéíóúüñ\s]",
        "",
        texto
    )

    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto)

    # Eliminar espacios al inicio y final
    texto = texto.strip()

    return texto



#MAIN
def main():

    archivo_entrada = archivo_entrada = BASE_DIR / "Fase de generación" / "dataset_reportes_sinteticos.csv"

    archivo_salida = "dataset_reportes_preprocesado.csv"

    print("=" * 60)
    print("PREPROCESAMIENTO DE TEXTO")
    print("=" * 60)

    print("\nCargando dataset...")

    df = pd.read_csv(
        archivo_entrada,
        encoding="utf-8-sig"
    )

    print(f"Total de reportes encontrados: {len(df)}")

    # Crear columna limpia
    df["Texto_Limpio"] = df["Texto_Reporte"].apply(
        limpiar_texto
    )

    # Guardar dataset
    df.to_csv(
        archivo_salida,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nDataset preprocesado correctamente.")
    print(f"Archivo generado: {archivo_salida}")

    # Mostrar algunos ejemplos
    print("\n" + "=" * 60)
    print("EJEMPLOS")
    print("=" * 60)

    for i in range(min(5, len(df))):

        print(f"\nEjemplo {i+1}")

        print("\nORIGINAL:")
        print(df.iloc[i]["Texto_Reporte"])

        print("\nLIMPIO:")
        print(df.iloc[i]["Texto_Limpio"])

        print("\n" + "-" * 60)

    # Validación final
    texto_completo = " ".join(
        df["Texto_Limpio"].astype(str)
    )

    caracteres_raros = sorted(
        set(
            re.sub(
                r"[a-záéíóúüñ\s]",
                "",
                texto_completo
            )
        )
    )

    print("\n" + "=" * 60)
    print("VALIDACIÓN")
    print("=" * 60)

    if len(caracteres_raros) == 0:
        print("✓ No se detectaron caracteres extraños.")
    else:
        print("⚠ Caracteres encontrados:")
        print(caracteres_raros)


if __name__ == "__main__":
    main()
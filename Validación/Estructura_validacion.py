# 04_validar_estructura_dataset.py
#
# OBJETIVO
# Validación automatizada de estructura y distribución de clases del dataset
# final, previa al entrenamiento de los modelos predictivos. Corresponde al
# primer filtro de la estrategia de auditoría y control de calidad (2.7.2 - A).
#
# Verifica:
#   1) Presencia de las columnas estructurales obligatorias.
#   2) Ausencia de valores nulos en Texto_Reporte.
#   3) Balance de clases en la variable Categoria, dentro de un margen de
#      tolerancia del 1% respecto al valor esperado por clase (N/3).

import pandas as pd


# ---------------------
# CONFIGURACIÓN
# ---------------------
archivo_entrada = "dataset_reportes_sinteticos.csv"

columnas_obligatorias = [
    "ID_Reporte",
    "Texto_Reporte",
    "Palabras_Clave_Sugeridas",
    "Categoria",
    "Urgencia",
    "Sentimiento"
]

TOLERANCIA_BALANCE = 0.01  # 1% de desviación máxima permitida respecto al valor esperado por clase


# ---------------------
# FUNCIÓN A1: VALIDAR ESTRUCTURA DE COLUMNAS
# ---------------------
def validar_columnas(df: pd.DataFrame) -> bool:
    columnas_faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if columnas_faltantes:
        print(f"✗ Columnas faltantes: {columnas_faltantes}")
        return False
    print("✓ Todas las columnas estructurales obligatorias están presentes.")
    return True


# ---------------------
# FUNCIÓN A2: VALIDAR VALORES NULOS EN Texto_Reporte
# ---------------------
def validar_nulos(df: pd.DataFrame) -> bool:
    nulos = df["Texto_Reporte"].isnull().sum()
    if nulos > 0:
        print(f"✗ Se encontraron {nulos} valores nulos en 'Texto_Reporte'.")
        return False
    print("✓ No se encontraron valores nulos en 'Texto_Reporte'.")
    return True


# ---------------------
# FUNCIÓN A3: VALIDAR BALANCE DE CLASES (con margen de tolerancia)
# ---------------------
def validar_balance_clases(df: pd.DataFrame, tolerancia: float = TOLERANCIA_BALANCE) -> bool:
    conteo = df["Categoria"].value_counts()
    total = len(df)
    n_clases = df["Categoria"].nunique()
    valor_esperado = total / n_clases

    print(f"\nDistribución observada (esperado ≈ {valor_esperado:.1f} por clase):")
    print(conteo)

    balance_ok = True
    for categoria, cantidad in conteo.items():
        desviacion = abs(cantidad - valor_esperado) / valor_esperado
        estado = "✓" if desviacion <= tolerancia else "✗"
        print(f"  {estado} {categoria}: {cantidad} reportes (desviación {desviacion*100:.2f}%)")
        if desviacion > tolerancia:
            balance_ok = False

    if balance_ok:
        print(f"✓ El balance de clases se encuentra dentro del margen de tolerancia (±{tolerancia*100:.0f}%).")
    else:
        print(f"✗ Se detectó un desbalance que excede el margen de tolerancia (±{tolerancia*100:.0f}%).")

    return balance_ok


# ---------------------
# MAIN
# ---------------------
def main():
    print("=" * 60)
    print("VALIDACIÓN AUTOMATIZADA DE ESTRUCTURA Y DISTRIBUCIÓN DE CLASES")
    print("=" * 60)

    print("\nCargando dataset...")
    df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")
    print(f"Total de reportes encontrados: {len(df)}")

    print("\n" + "-" * 60)
    print("A.1 - VALIDACIÓN DE ESTRUCTURA DE COLUMNAS")
    print("-" * 60)
    ok_columnas = validar_columnas(df)

    print("\n" + "-" * 60)
    print("A.2 - VALIDACIÓN DE VALORES NULOS")
    print("-" * 60)
    ok_nulos = validar_nulos(df) if ok_columnas else False

    print("\n" + "-" * 60)
    print("A.3 - VALIDACIÓN DE BALANCE DE CLASES")
    print("-" * 60)
    ok_balance = validar_balance_clases(df) if ok_columnas else False

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    if ok_columnas and ok_nulos and ok_balance:
        print("✅ El dataset pasó todas las validaciones estructurales. Apto para continuar el flujo.")
    else:
        print("⚠ El dataset presenta observaciones. Revisar los detalles impresos arriba antes de continuar.")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:41:08 2025

@author: Pablo D
"""

import pandas as pd

# Cargar sin cabecera
archivo_poblacion = "poblacion_total_municipios.xlsx"
raw = pd.read_excel(archivo_poblacion, sheet_name=0, header=None)

# Buscar columnas válidas donde:
# - Fila 11 (índice 11) tiene el año
# - Fila 12 (índice 12) tiene '(POBLACION) Population'
col_indices = []
años = []

for i in range(2, raw.shape[1]):
    campo = str(raw.iloc[12, i]).strip()
    año_raw = str(raw.iloc[11, i]).strip()
    if "Population" in campo and "(" in año_raw:
        extraido = pd.Series(año_raw).str.extract(r"(\d{4})")[0]
        if not extraido.isna().all():
            año = int(extraido[0])
            col_indices.append(i)
            años.append(año)

# Inicializar lista para cada año
bloques = []

for col_idx, año in zip(col_indices, años):
    bloque = raw.iloc[13:, [0, 1, col_idx]].copy()
    bloque.columns = ["codigo_municipio", "sexo", "poblacion"]
    bloque["codigo_municipio"] = bloque["codigo_municipio"].ffill()
    bloque["label"] = bloque["codigo_municipio"].str.extract(r"\)\s*(.*)$")[0]
    bloque["time_period"] = año
    bloque["poblacion"] = pd.to_numeric(bloque["poblacion"], errors="coerce")
    bloques.append(bloque)

# Unir todos los bloques
poblacion_larga = pd.concat(bloques, ignore_index=True)

# Pivotear: una fila por municipio y año, columnas para cada sexo
poblacion_final = poblacion_larga.pivot_table(
    index=["label", "time_period"],
    columns="sexo",
    values="poblacion",
    aggfunc="first"
).reset_index()

# Renombrar columnas
poblacion_final.columns.name = None
poblacion_final.rename(columns={
    "(F) Female": "poblacion_femenina",
    "(M) Male": "poblacion_masculina",
    "(_T) Total": "poblacion_total"
}, inplace=True)

# Verificar
print(poblacion_final.head())



# Mapeo de municipios a islas (como ya tienes)
municipios_a_islas = {
    # El Hierro (3)
    "El Pinar de El Hierro": "El Hierro",
    "Frontera": "El Hierro",
    "Valverde": "El Hierro",

    # La Gomera (6)
    "Agulo": "La Gomera",
    "Alajeró": "La Gomera",
    "Hermigua": "La Gomera",
    "San Sebastián de La Gomera": "La Gomera",
    "Valle Gran Rey": "La Gomera",
    "Vallehermoso": "La Gomera",

    # La Palma (14)
    "Barlovento": "La Palma",
    "Breña Alta": "La Palma",
    "Breña Baja": "La Palma",
    "El Paso": "La Palma",
    "Fuencaliente de La Palma": "La Palma",
    "Garafía": "La Palma",
    "Los Llanos de Aridane": "La Palma",
    "Puntagorda": "La Palma",
    "Puntallana": "La Palma",
    "San Andrés y Sauces": "La Palma",
    "Santa Cruz de La Palma": "La Palma",
    "Tazacorte": "La Palma",
    "Tijarafe": "La Palma",
    "Villa de Mazo": "La Palma",

    # Lanzarote (7)
    "Arrecife": "Lanzarote",
    "Haría": "Lanzarote",
    "San Bartolomé": "Lanzarote",
    "Teguise": "Lanzarote",
    "Tías": "Lanzarote",
    "Tinajo": "Lanzarote",
    "Yaiza": "Lanzarote",

    # Fuerteventura (6)
    "Antigua": "Fuerteventura",
    "Betancuria": "Fuerteventura",
    "La Oliva": "Fuerteventura",
    "Pájara": "Fuerteventura",
    "Puerto del Rosario": "Fuerteventura",
    "Tuineje": "Fuerteventura",

    # Gran Canaria (21)
    "Agaete": "Gran Canaria",
    "Agüimes": "Gran Canaria",
    "Artenara": "Gran Canaria",
    "Arucas": "Gran Canaria",
    "Firgas": "Gran Canaria",
    "Gáldar": "Gran Canaria",
    "Ingenio": "Gran Canaria",
    "La Aldea de San Nicolás": "Gran Canaria",
    "Las Palmas de Gran Canaria": "Gran Canaria",
    "Mogán": "Gran Canaria",
    "Moya": "Gran Canaria",
    "San Bartolomé de Tirajana": "Gran Canaria",
    "Santa Brígida": "Gran Canaria",
    "Santa Lucía de Tirajana": "Gran Canaria",
    "Santa María de Guía de Gran Canaria": "Gran Canaria",
    "Tejeda": "Gran Canaria",
    "Telde": "Gran Canaria",
    "Teror": "Gran Canaria",
    "Valleseco": "Gran Canaria",
    "Valsequillo de Gran Canaria": "Gran Canaria",
    "Vega de San Mateo": "Gran Canaria",

    # Tenerife (28)
    "Adeje": "Tenerife",
    "Arafo": "Tenerife",
    "Arico": "Tenerife",
    "Arona": "Tenerife",
    "Buenavista del Norte": "Tenerife",
    "Candelaria": "Tenerife",
    "El Rosario": "Tenerife",
    "El Sauzal": "Tenerife",
    "El Tanque": "Tenerife",
    "Fasnia": "Tenerife",
    "Garachico": "Tenerife",
    "Granadilla de Abona": "Tenerife",
    "Guía de Isora": "Tenerife",
    "Güímar": "Tenerife",
    "Icod de los Vinos": "Tenerife",
    "La Guancha": "Tenerife",
    "La Matanza de Acentejo": "Tenerife",
    "La Orotava": "Tenerife",
    "La Victoria de Acentejo": "Tenerife",
    "Los Realejos": "Tenerife",
    "Los Silos": "Tenerife",
    "Puerto de la Cruz": "Tenerife",
    "San Cristóbal de La Laguna": "Tenerife",
    "San Juan de la Rambla": "Tenerife",
    "San Miguel de Abona": "Tenerife",
    "Santa Cruz de Tenerife": "Tenerife",
    "Santa Úrsula": "Tenerife",
    "Santiago del Teide": "Tenerife",
    "Tacoronte": "Tenerife",
    "Tegueste": "Tenerife",
    "Vilaflor de Chasna": "Tenerife"
}
# Asignamos las islas a los municipios en el dataframe
poblacion_final["isla"] = poblacion_final["label"].map(municipios_a_islas)

# Comprobamos si hay municipios sin isla asignada
municipios_sin_isla = poblacion_final[poblacion_final["isla"].isna()]
print(f"\nMunicipios sin isla asignada: {len(municipios_sin_isla)}")

# Comprobación por año
año_objetivo = 2023  # Cambiar por el año que quieras
df_año = poblacion_final[poblacion_final["time_period"] == año_objetivo]

# Sumar población por isla
poblacion_por_isla = df_año.groupby("isla")[["poblacion_total", "poblacion_masculina", "poblacion_femenina"]].sum()

# Comparar con el total de "Canary Islands"
canarias_total = df_año[df_año["label"] == "Canary Islands"]
print(f"\nPoblación total reportada para 'Canary Islands' en {año_objetivo}:")
print(canarias_total[["poblacion_total", "poblacion_masculina", "poblacion_femenina"]])


# Eliminar 'Canary Islands' si está en el grupo (evita duplicación)
poblacion_por_isla_sin_canarias = poblacion_por_isla.drop("Canary Islands", errors="ignore")

# Mostrar población total por isla (sin 'Canary Islands')
print(f"\nPoblación por isla (sin 'Canary Islands') en {año_objetivo}:")
print(poblacion_por_isla_sin_canarias)

# Sumar población total desde islas individuales
suma_islas = poblacion_por_isla_sin_canarias.sum()
print(f"\nSuma total por islas en {año_objetivo}:")
print(suma_islas)

# Comparar con el total reportado por 'Canary Islands'
print(f"\nComparación con 'Canary Islands' en {año_objetivo}:")
print(canarias_total[["poblacion_total", "poblacion_masculina", "poblacion_femenina"]])


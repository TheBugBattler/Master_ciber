# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:33:13 2025

@author: Pablo D
"""

#Cargamos modulos


import pandas as pd

# Cargar datos (omite si ya está cargado)
data = pd.read_csv("laboral_municipios.csv")

print("Columnas del dataset:")
print(data.columns.tolist())

print("Años disponibles (ordenados):")
print(sorted(data["time_period"].unique()))

municipios = sorted(data["label"].unique())
print(f"Número de municipios distintos: {len(municipios)}")
print("Ejemplo de municipios:")
print(municipios) 

print(f"Número de años distintos: {data['time_period'].nunique()}")

print("\nNúmero de registros por año:")
print(data["time_period"].value_counts().sort_index())


#Para añadir nueva columna de isla y poder organizar por isla

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

data["isla"] = data["label"].map(municipios_a_islas)
print("Nulos en 'isla':", data["isla"].isna().sum())


#--------------------------------------------------------------------------
#Merging

# Comparar MUNICIPIOS (labels) con los del archivo de población
municipios_laboral = set(data["label"].unique())

# Guardamos para comparar luego
print(f"Municipios en datos laborales: {len(municipios_laboral)}")
print("Ejemplo:", sorted(list(municipios_laboral))[:10])  # muestra algunos

# Comparar AÑOS (time_period) con los del archivo de población
años_laboral = set(data["time_period"].unique())

print(f"Años en datos laborales: {sorted(años_laboral)}")

poblacion_simplificada = pd.read_csv("poblacion_municipios_simplificada.csv")
data = data.merge(
    poblacion_simplificada,
    on=["label", "time_period"],
    how="left"
)


print("Columnas tras el merge:", data.columns.tolist())

# Ver si hay alguna fila sin datos de población
faltantes = data[data["poblacion_total"].isna()]
print(f"\nFilas sin datos de población:", len(faltantes))
if not faltantes.empty:
    print(faltantes[["label", "time_period"]].drop_duplicates())

#----------------------------------------------------------------------------



# === DEFINICIÓN CORRECTA DE TODAS LAS TASAS ===

# --- Tasa de paro (%): población parada / población activa ---
data["tasa_paro"] = 100 * data["ppar_t"] / data["pact_t"]
data["tasa_paro_m"] = 100 * data["ppar_m"] / data["pact_m"]
data["tasa_paro_f"] = 100 * data["ppar_f"] / data["pact_f"]

# --- Tasa de actividad (%): población activa / población total ---
data["tasa_actividad"] = 100 * data["pact_t"] / data["poblacion_total"]
data["tasa_actividad_m"] = 100 * data["pact_m"] / data["poblacion_masculina"]
data["tasa_actividad_f"] = 100 * data["pact_f"] / data["poblacion_femenina"]

# --- Tasa de empleo (%): población ocupada / población total ---
data["tasa_empleo"] = 100 * data["pocu_t"] / data["poblacion_total"]
data["tasa_empleo_m"] = 100 * data["pocu_m"] / data["poblacion_masculina"]
data["tasa_empleo_f"] = 100 * data["pocu_f"] / data["poblacion_femenina"]

# --- Redondear todas las tasas a 4 decimales ---
columnas_tasas = [col for col in data.columns if col.startswith("tasa_")]
data[columnas_tasas] = data[columnas_tasas].round(4)

# === COMPROBACIÓN CON LAS TASAS DE PARO OFICIALES ===

data["diferencia_tasa_paro"] = data["tasa_paro"] - data["tpar_t"]
data["diferencia_tasa_paro_m"] = data["tasa_paro_m"] - data["tpar_m"]
data["diferencia_tasa_paro_f"] = data["tasa_paro_f"] - data["tpar_f"]

# Ver resumen estadístico de las diferencias
print("Resumen de diferencias (nuestras tasas - tasas oficiales):")
print(data[[
    "diferencia_tasa_paro", 
    "diferencia_tasa_paro_m", 
    "diferencia_tasa_paro_f"
]].describe())

# Mostrar discrepancias si las hay
discrepancias = data[
    (data["diferencia_tasa_paro"].abs() > 0.0001) |
    (data["diferencia_tasa_paro_m"].abs() > 0.0001) |
    (data["diferencia_tasa_paro_f"].abs() > 0.0001)
]

print(f"\nNúmero de discrepancias detectadas: {len(discrepancias)}")
if not discrepancias.empty:
    print(discrepancias[[
        "label", "time_period",
        "tasa_paro", "tpar_t", "diferencia_tasa_paro",
        "tasa_paro_m", "tpar_m", "diferencia_tasa_paro_m",
        "tasa_paro_f", "tpar_f", "diferencia_tasa_paro_f"
    ]].head())


columnas_exportar = [
    "label", "isla", "time_period",
    "poblacion_total", "poblacion_masculina", "poblacion_femenina",
    "pact_t", "pact_m", "pact_f",
    "pocu_t", "pocu_m", "pocu_f",
    "ppar_t", "ppar_m", "ppar_f",
    "tpar_t", "tpar_m", "tpar_f",  # ✅ tasas oficiales de paro (únicas que nos quedamos)
    "tasa_actividad", "tasa_actividad_m", "tasa_actividad_f",
    "tasa_empleo", "tasa_empleo_m", "tasa_empleo_f",
    "geom"
]
# Exportar a CSV
data[columnas_exportar].to_csv("municipios_enriquecido2.csv", index=False)

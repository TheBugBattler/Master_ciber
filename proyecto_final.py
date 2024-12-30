# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 10:21:31 2024
@author: Pablo D
"""
# PROGRAMA PARA PROYECTO FINAL

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, wilcoxon
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Leemos los datos
data = pd.read_csv('peliculas_limpias.csv')  #Cargamos datos
print(data.columns) #Imprimimos nombre columnas

# Procesamiento inicial
#Leemos archivo. Nos quedamos con primer genero, convertimos fechas y calculamos ROI
#data['genres'] = data['genres'].str.split(',').str[0]  # Nos quedamos con primer valor de género

#Pintamos generos
# Limpiar espacios en blanco alrededor de cada género
generos_series = data['genres'].str.split(',').explode().str.strip()
# Recontar las frecuencias después de la limpieza
frecuencias_generos = generos_series.value_counts()
# Regenerar el histograma
plt.figure(figsize=(10, 6))
frecuencias_generos.plot(kind='bar', color=plt.cm.tab10.colors, alpha=0.8)
# Configurar título y etiquetas
plt.title('Frecuencia de Géneros en las Películas', fontsize=14)
plt.xlabel('Géneros', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)

# Mostrar el histograma
plt.tight_layout()
plt.show()

'''
data = data.drop(columns=['adult']) #Quitamos columna adult ya que no hay peliculas de este tipo
genres_encoded = data['genres'].str.get_dummies(sep=', ') #Convertimos columna genres en dummy
data = pd.concat([data, genres_encoded], axis=1)
data.drop(columns=['genres'], inplace=True) #Eliminamos columna genres original para evitar redundancia
data['release_date'] = pd.to_datetime(data['release_date'])  # Convertir formato fecha
data['año'] = data['release_date'].dt.year  #Añadimos columna al dataset con año
data['ROI'] = ((data['revenue'] - data['budget']) / data['budget']) * 100  # Return of Investment

print(data.describe())  # Valores rápidos estadísticos
idiomas = data['original_language'].value_counts()  # Conteo de idiomas
#generos = data["genres"].value_counts()  # Conteo de géneros

# Histograma de idiomas
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='original_language', order=data['original_language'].value_counts().index)
plt.title('Distribución de Idiomas')
plt.xticks(rotation=90)
plt.xlabel('Idioma')
plt.ylabel('Frecuencia')
plt.show()
'''


'''
#ESTUDIO DE READAPTACIONES


# Identificar duplicados por 'title'
duplicados = data[data.duplicated(subset='title', keep=False)] #Identificamos los titulos repetidos 

# Clasificar como 'original' o 'readaptacion' basado en el año más antiguo
duplicados['tipo'] = duplicados.groupby('title')['año'].transform(
    lambda x: ['original' if i == x.idxmin() else 'readaptacion' for i in x.index]) #Agrupamos peliculas
#Titulo mas antiguo lo identificamos como original y los demas como readaptacion

# Separar en originales y readaptaciones
originales = duplicados[duplicados['tipo'] == 'original']
readaptaciones = duplicados[duplicados['tipo'] == 'readaptacion']

metricas = ['revenue', 'ROI', 'vote_average', 'vote_count'] #Metricas para estudiar diferencia entre originales y readaptaciones
resultados = {}

# Calculamos puntuaje ponderado de votos 
originales['weighted_score'] = (originales['vote_average'] * originales['vote_count']) / originales['vote_count'].sum()
readaptaciones['weighted_score'] = (readaptaciones['vote_average'] * readaptaciones['vote_count']) / readaptaciones['vote_count'].sum()

# Añadir a métricas
metricas.append('weighted_score') #Añadimos a las metricas que vamos a estudiar media ponderada
for metrica in metricas:
    # Seleccionar los valores de la métrica actual
    mean_orig, mean_read = originales[metrica].mean(), readaptaciones[metrica].mean()
    t_stat, p_val = ttest_ind(originales[metrica], readaptaciones[metrica], equal_var=False)
    
    # Añadir resultados en una sola línea
    resultados[metrica] = [mean_orig, mean_read, t_stat, p_val]
    
#Para cada metrica (revenue, roi, vote average, vote count) calculamos media de originales, media readaptaciones
#y valor estadistico de la prueba t-stat y p-valor. 


# Convertir los resultados a un DataFrame para mostrar
resultados_df = pd.DataFrame(resultados, index=['Media Originales', 'Media Readaptaciones', 't-stat', 'p-valor']).T
print("Comparación de métricas con pruebas estadísticas:")
print(resultados_df)

# Visualización de las distribuciones para cada métrica
for metrica in metricas:
    plt.figure(figsize=(8, 6))
    plt.boxplot([originales[metrica], readaptaciones[metrica]], labels=['Originales', 'Readaptaciones'])
    plt.title(f'Distribución de {metrica} entre Originales y Readaptaciones')
    plt.ylabel(metrica)
    plt.show()
'''


'''
# VARIABLES NUMÉRICAS
numericas = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'año', "ROI"]

# TRATAMIENTO DE OUTLIERS: Gráfico de boxplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10)) #Figuras
axes = axes.flatten()

for i, column in enumerate(numericas): #Para cada columna numerica generamos boxplot
    sns.boxplot(x=data[column], ax=axes[i])
    axes[i].set_title(f'Boxplot de {column}')

# Ocultamos subplots vacios
for j in range(len(numericas), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
'''

'''
# DETECCIÓN DE OUTLIERS Rango intercuartilico 
outliers_info = {}
for column in numericas:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers_below = (data[column] < lower_limit).sum()
    outliers_above = (data[column] > upper_limit).sum()

    outliers_info[column] = {'outliers_below': outliers_below, 'outliers_above': outliers_above}

# Imprimir resultados de outliers
for column, outliers in outliers_info.items():
    print(f"Variable: {column}")
    print(f"Outliers por debajo del límite inferior: {outliers['outliers_below']}")
    print(f"Outliers por encima del límite superior: {outliers['outliers_above']}")
    print("-" * 30)
'''    

'''
#ANALISIS ESTUDIO PELICULAS POR MES

# Creamos una nueva columna para el mes de estreno
data['mes'] = data['release_date'].dt.month
# Calculamos metrica por mes
metricas_por_mes = data.groupby('mes').agg(revenue_promedio=('revenue', 'mean'),revenue_mediana=('revenue', 'median'), revenue_total=('revenue', 'sum'),num_peliculas=('revenue', 'size')).reset_index()
#Hemos agrupado por mes, calculamos media ingresos, mediana ingresos, total ingresos, numero peliculas


revenue_total_global = data['revenue'].sum() #Calculamos ingreso total de todas las peliculas
metricas_por_mes['proporcion_revenue'] = metricas_por_mes['revenue_total'] / revenue_total_global #Para cada mes, dividimos total de ingresos mensuales por ingreso total global
# Visualizar métricas clave
plt.figure(figsize=(12, 8))
# Gráfico de revenue promedio y mediana
plt.subplot(2, 1, 1)
plt.bar(metricas_por_mes['mes'], metricas_por_mes['revenue_promedio'], color='skyblue', label='Promedio')
plt.bar(metricas_por_mes['mes'], metricas_por_mes['revenue_mediana'], color='lightcoral', label='Mediana', alpha=0.7)
plt.title('Revenue promedio y mediana por mes de estreno (Todos los años)')
plt.xlabel('Mes')
plt.ylabel('Revenue')
plt.xticks(ticks=range(1, 13), labels=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
plt.legend()
# Gráfico del número de películas por mes
plt.subplot(2, 1, 2)
plt.bar(metricas_por_mes['mes'], metricas_por_mes['num_peliculas'], color='teal')
plt.title('Número de películas estrenadas por mes (Todos los años)')
plt.xlabel('Mes')
plt.ylabel('Cantidad de películas')
plt.xticks(ticks=range(1, 13), labels=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], rotation=45)
plt.tight_layout()
plt.show()
# Mostrar métricas calculadas por mes
print("Métricas calculadas por mes:")
print(metricas_por_mes)

'''

'''

#-----------------------------------------------------------------------------------
# CODIFICACIÓN LENGUAJE
#-----------------------------------------------------------------------------------
# Agrupación y codificación de 'original_language'
umbral = 5  # Idiomas con menos de 5 películas se agrupan como 'other'
frequent_languages = idiomas[idiomas >= umbral].index
data['original_language'] = data['original_language'].apply(
    lambda x: x if x in frequent_languages else 'other')
languages_encoded = pd.get_dummies(data['original_language'], prefix='lang') #Aplicamos dummies
data = pd.concat([data, languages_encoded], axis=1)


#-----------------------------------------------------------------------------------
#Nueva categoria categorica revenue para predicciones
#---------------------------------------------------
# Calculamos percentiles para hacer division de revenue
percentile_33 = data['revenue'].quantile(0.33)
percentile_66 = data['revenue'].quantile(0.66)
def categorize_revenue_by_percentiles(revenue):
    if revenue <= percentile_33:
        return 'bajo'
    elif revenue <= percentile_66:
        return 'medio'
    else:
        return 'alto'
data['revenue_category'] = data['revenue'].apply(categorize_revenue_by_percentiles) #Añadimos nueva columna


#-----------------------------------------------------------------------------------
#Normalizamos
#----------------------------------------------------------------------------------
# VARIABLES NUMÉRICAS: Normalización con Robust Scaler. Menos sensibles a outliers 
scaler = RobustScaler()
data[numericas] = scaler.fit_transform(data[numericas])


#PREDICCIONES DE REVENUE#
# CLASIFICADOR KNN 

# SELECCIÓN DE FEATURES (X) Y TARGET (y)
# -------------------------------------------------------------------
y = data['revenue_category']


# Codificamos la variable objetivo a valores numéricos
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
data['revenue_category_encoded'] = y_encoded #Añadimos columna codificada
#Con esto convertimos bajo 0 medio 1 alto 2. 

# Calcular la matriz de correlación Spearman incluyendo la variable objetivo
spearman_corr_matrix = data[numericas + ['revenue_category_encoded']].corr(method='spearman')
# Ordenar las correlaciones con respecto a la variable objetivo
correlation_with_target = spearman_corr_matrix['revenue_category_encoded'].drop('revenue_category_encoded').sort_values(ascending=False)

'''
'''
# Visualización opcional
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlaciones (Spearman) - Variables Numéricas")
plt.show()
'''

'''
# Mostrar las variables más correlacionadas con la variable objetivo
print("Correlaciones de las variables numéricas con la variable objetivo:")
print(correlation_with_target)

#Hemos guardado este dataset para ejecutar resto de programa en otro .py

'''



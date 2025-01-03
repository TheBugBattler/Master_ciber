# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:07:28 2024

@author: Pablo D
""" 
#IMPORTAMOS MODULOS

import numpy as np
from scipy.stats import spearmanr, chi2_contingency
import seaborn as sns
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
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV

data=pd.read_csv("prueba_dia_29.csv") #Leemos nueva base de datos
numericas = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget'] #Varibales numericas

'''

#-----------------------------------------------------------------------------------
#AGRUPAMIENTO KMEANS
#-----------------------------------------------------------------------------------


# Selección de variables relevantes
#Buscamos que generos tienen relacion significativa con variables numericas. Los que estén 
#poco correlacionados probablemente no aportarán información util al modelo de agrupamiento 

# Calcular las correlaciones entre las variables numéricas y los géneros
genre_columns = [col for col in data.columns if col in ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
                                                        'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
                                                        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 
                                                        'Thriller', 'War', 'Western']]

# Crear una matriz de correlación entre las variables numéricas y los géneros. Correlacion Spearman 
correlation_matrix = data[numericas + genre_columns].corr(method='spearman')

# Extraer las correlaciones entre los géneros y las variables numéricas
genre_correlations = correlation_matrix.loc[genre_columns, numericas]

# Ordenar géneros por su máxima correlación con alguna variable numérica
genre_relevance = genre_correlations.abs().max(axis=1).sort_values(ascending=False)

# Mostrar las correlaciones ordenadas
print(genre_relevance)
# Seleccionamos géneros con correlación absoluta mayor a 0.2
selected_genres = genre_relevance[genre_relevance > 0.2].index.tolist()

variables_estudio=["vote_average","vote_count","runtime","revenue","budget"]


#selected_features = ['vote_average', 'vote_count', 'runtime', 'revenue'] #Elegimos estas variables para agrupamiento

all_features = variables_estudio#  + selected_genres

clustering_data = data[all_features] #Dataset para agrupamiento

# Definir rango de valores de k
k_values = range(2, 20)

# Inicializamos listas para almacenar métricas
inertia = []
silhouette_scores = []

# Calculamos el método del codo y la puntuación de Silhouette
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(clustering_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(clustering_data, kmeans.labels_))

# Graficar el método del codo
plt.figure(figsize=(10, 5))
plt.plot(k_values, inertia, marker='o')
plt.title('Método del Codo para Determinar k Óptimo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.grid(True)
plt.show()

# Graficar las puntuaciones de Silhouette
plt.figure(figsize=(10, 5))
plt.plot(k_values, silhouette_scores, marker='o', color='orange')
plt.title('Puntuación de Silhouette para Determinar k Óptimo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Puntuación de Silhouette')
plt.grid(True)
plt.show()
from sklearn.decomposition import PCA

# Aplicamos K-Means con k=5
kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10, max_iter=300)
data['cluster'] = kmeans_final.fit_predict(clustering_data)

# Reducimos a 2 dimensiones para visualización mediante analisis PCA. Es exclusivo para visualizar clusters
#Las componentes principales pca1, pca2 no tiene significado directo como las variables originales, no podemos por ejemplo
#decir que pca1 representa revenue, etc.

pca = PCA(n_components=2)
pca_features = pca.fit_transform(clustering_data)

# Agregamos las componentes principales al dataset
data['pca1'] = pca_features[:, 0]
data['pca2'] = pca_features[:, 1]

# Graficamos los clusters en 2D
plt.figure(figsize=(12, 8))
for cluster in range(5):
    cluster_points = data[data['cluster'] == cluster]
    plt.scatter(cluster_points['pca1'], cluster_points['pca2'], label=f'Cluster {cluster}', alpha=0.6)

plt.title('Visualización de Clusters en 2D (PCA)', fontsize=14)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend(title="Clusters")
plt.grid(True)
plt.show()


#Numero de peliculas en cada cluster
cluster_counts = data['cluster'].value_counts()

print("Cantidad de películas en cada cluster:", cluster_counts)

#Seleccionamos peliculas mas relevantes de cada cluster respecto a vote_average
for cluster in range(5):  # Cambiar el número según los clusters
    print(f"\nPelículas representativas del Cluster {cluster}:")
    cluster_movies = data[data['cluster'] == cluster].sort_values(by='vote_average', ascending=False)
    print(cluster_movies[['title', 'vote_average', 'revenue', 'release_date']].head(10))
'''

'''

#------------------------------------------------------------------------------------------
#BUSQUEDA PELICULAS SIMILARES
#-----------------------------------------------------------------------------------------

#Usamos similitud coseno porque no depende de la magnitud de los vectores, sino de su orientacion. 

from sklearn.metrics.pairwise import cosine_similarity

# Selección de características relevantes para similitud
similarity_features = ['vote_average', 'vote_count', 'revenue'] + selected_genres

# Extraemos la matriz de características
similarity_matrix = data[similarity_features]

# Calcular la matriz de similitud coseno
cosine_sim_matrix = cosine_similarity(similarity_matrix)

# Función para recomendar películas similares
def recommend_movies(movie_title, data, sim_matrix, top_n=5):
    # Buscamos el índice de la película en el dataset
    movie_idx = data[data['title'] == movie_title].index[0]
    
    # Obtenemos las puntuaciones de similitud para esta película
    sim_scores = list(enumerate(sim_matrix[movie_idx]))
    
    # Ordenamos las películas por similitud (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Obtenemos los índices de las películas más similares
    similar_movies_idx = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Retornamos los títulos de las películas similares
    return data.iloc[similar_movies_idx][['title', 'release_date']]

# Probamos el recomendador
movie_to_recommend = "The Matrix"  
recommended_movies = recommend_movies(movie_to_recommend, data, cosine_sim_matrix, top_n=5)
print(f"Películas similares a '{movie_to_recommend}':")
print(recommended_movies)

'''



#--------------------------------------------------------------------
#PREDICCIONES DE REVENUE#
# CLASIFICADOR KNN 
# -------------------------------------------------------------------

# Calcular la matriz de correlación Spearman incluyendo la variable objetivo
spearman_corr_matrix = data[numericas + ['revenue_category_encoded']].corr(method='spearman')

# Ordenar las correlaciones con respecto a la variable objetivo
correlation_with_target = spearman_corr_matrix['revenue_category_encoded'].drop('revenue_category_encoded').sort_values(ascending=False)


# Visualización opcional
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de Correlaciones (Spearman) - Variables Numéricas")
plt.show()

# Mostrar las variables más correlacionadas con la variable objetivo
print("Correlaciones de las variables numéricas con la variable objetivo:")
print(correlation_with_target)

#No incluimos generos ni idiomas porque no mejora sustancialmente

X = data[['vote_average', 'vote_count', 'runtime', 'budget']] #Variables seleccionados despues de realizar estudio
y_encoded = data['revenue_category_encoded']

# -------------------------------------------------------------------
# CONFIGURACIÓN DE VALIDACIÓN CRUZADA E INSTANCIAS DE CLASIFICADORES
# -------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #5 folds, Stratified se asegura de que en cada fold haya aprox misma proporcion
#de clases (bajo, medio, alto) 

# -------------------------------------------------------------------
# 1. KNN
# -------------------------------------------------------------------
k_values = range(1, 21)
mean_scores = []
#Busqueda mejor K
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y_encoded, cv=skf, scoring='accuracy')
    mean_scores.append(scores.mean())

best_k = k_values[mean_scores.index(max(mean_scores))]
print(f"El mejor valor de k es: {best_k} con precisión promedio de {max(mean_scores):.4f}")

# Evaluar el modelo final con el mejor k
knn_final = KNeighborsClassifier(n_neighbors=best_k)
final_scores = cross_val_score(knn_final, X, y_encoded, cv=skf, scoring='accuracy')

print(f"Precisión promedio final con KNN (validación cruzada): {final_scores.mean():.4f} ± {final_scores.std():.4f}")

# Predicciones en validación cruzada
y_pred_knn = cross_val_predict(knn_final, X, y_encoded, cv=skf)


target_names = data['revenue_category'].unique()

print(classification_report(y_encoded, y_pred_knn, target_names=target_names))

cm_knn = confusion_matrix(y_encoded, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn, display_labels=target_names)
disp_knn.plot(cmap='Blues')
plt.title('Matriz de Confusión para KNN')
plt.show()


# -------------------------------------------------------------------
# 2. Árbol de Decisión
# -------------------------------------------------------------------


tree_params = {'max_depth': range(1, 21)}  # Probar profundidades de 1 a 20
tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), tree_params, cv=skf, scoring='accuracy')
tree_grid.fit(X, y_encoded)

# Mejor profundidad y su rendimiento
best_depth = tree_grid.best_params_['max_depth']
print(f"Mejor profundidad para Árbol de Decisión: {best_depth} con precisión promedio de {tree_grid.best_score_:.4f}")

# -------------------------------------------------------------------
# Modelo Final con la mejor profundidad
# -------------------------------------------------------------------
tree_model = DecisionTreeClassifier(random_state=42, max_depth=best_depth)

# Precisión promedio en validación cruzada
tree_scores = cross_val_score(tree_model, X, y_encoded, cv=skf, scoring='accuracy')
print(f"Precisión promedio del Árbol de Decisión (validación cruzada): {tree_scores.mean():.4f} ± {tree_scores.std():.4f}")

# Predicciones en validación cruzada para matriz de confusión y reporte
y_pred_tree = cross_val_predict(tree_model, X, y_encoded, cv=skf)

# Nombres de las clases
target_names = data['revenue_category'].unique()

# Reporte de clasificación
print("Reporte de Clasificación para Árbol de Decisión:")
print(classification_report(y_encoded, y_pred_tree, target_names=target_names))

# Matriz de confusión
cm_tree = confusion_matrix(y_encoded, y_pred_tree)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=target_names)
disp_tree.plot(cmap='Blues')
plt.title('Matriz de Confusión para Árbol de Decisión')
plt.show()


# -------------------------------------------------------------------
# 3. Naive Bayes
# -------------------------------------------------------------------
naive_bayes_model = GaussianNB()
naive_bayes_scores = cross_val_score(naive_bayes_model, X, y_encoded, cv=skf, scoring='accuracy')

print(f"Precisión promedio con Naive Bayes (validación cruzada): {naive_bayes_scores.mean():.4f} ± {naive_bayes_scores.std():.4f}")

# Predicciones y reporte
y_pred_nb = cross_val_predict(naive_bayes_model, X, y_encoded, cv=skf)
print("Reporte de Clasificación para Naive Bayes:")
print(classification_report(y_encoded, y_pred_nb, target_names=data['revenue_category'].unique()))

cm_nb = confusion_matrix(y_encoded, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=data['revenue_category'].unique())
disp_nb.plot(cmap='Blues')
plt.title('Matriz de Confusión para Naive Bayes')
plt.show()



# -------------------------------------------------------------------
# 4. SVM
# -------------------------------------------------------------------
svm_model = SVC(kernel='linear', random_state=42)
svm_scores = cross_val_score(svm_model, X, y_encoded, cv=skf, scoring='accuracy')

print(f"Precisión promedio con SVM (validación cruzada): {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")

y_pred_svm = cross_val_predict(svm_model, X, y_encoded, cv=skf)
print("Reporte de Clasificación para SVM:")
print(classification_report(y_encoded, y_pred_svm, target_names=data['revenue_category'].unique()))

cm_svm = confusion_matrix(y_encoded, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=data['revenue_category'].unique())
disp_svm.plot(cmap='Blues')
plt.title('Matriz de Confusión para SVM')
plt.show()



# -------------------------------------------------------------------
# 5. Random Forest
# -------------------------------------------------------------------
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_scores = cross_val_score(random_forest_model, X, y_encoded, cv=skf, scoring='accuracy')

print(f"Precisión promedio con Random Forest (validación cruzada): {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

y_pred_rf = cross_val_predict(random_forest_model, X, y_encoded, cv=skf)
print("Reporte de Clasificación para Random Forest:")
print(classification_report(y_encoded, y_pred_rf, target_names=data['revenue_category'].unique()))

cm_rf = confusion_matrix(y_encoded, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=data['revenue_category'].unique())
disp_rf.plot(cmap='Blues')
plt.title('Matriz de Confusión para Random Forest')
plt.show()





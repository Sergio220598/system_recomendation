# 🎬 Sistema de Recomendación

Este repositorio contiene el flujo de trabajo completo para el desarrollo de un **Sistema de Recomendación basado en Contenido** utilizando **embeddings** y dataset de peliculas **The Movie Dataset**. Este dataset contiene informacion de las peliculas y los ratings por parte de los usuarios. El proyecto abarca la limpieza de datos, Analisis exploratorio de datos, generacion de embeddings e implementacion del sistema de recomendacion.

## 📋 Requisitos Previos
- **Python 3.8**
- **Pandas** para manipulación de datos  
- **NumPy** para operaciones vectoriales  
- **Scikit-learn** para cálculos de similitud (cosine similarity)  
- **Jupyter Notebook** para el desarrollo interactivo  
- **The Moivie Dataset** Dataset de peliculas

## 📂 Estructura del Proyecto


│── **1_Limpieza.ipynb**: Extracción, transformación y carga de datos crudos.

│── **2_EDA.ipynb**: Análisis Exploratorio de Datos para entender distribuciones y outliers.

│── **3_Embedding.ipynb**: Preparación de embeddings para el sistema.

│── **4_sistema.ipynb (Archivo Principal)**: Implementación del sistema de recomendación.

│── **The_Movie_Dataset**: Dataset de peliculas

    │──movies_metadata.csv: Informacion de peliculas

    │──rating_small.csv: Rating de usuarios a las peliculas

│── **archivos**: Outputs

    │──dataset_merged.csv: Merge (solo películas con ratings) para garantizar IDs válidos

    │──embedding_movies.csv: Embeddings de peliculas


## 🧠 Análisis del Notebook: `4_sistema.ipynb`

Este notebook es el corazón del proyecto y se encarga de la lógica algorítmica. La lógica contempla lo siguiente:

**1.** El sistema funciona bajo la premisa de similitud de contenido:

**2.** Se cargan los embeddings de todas las películas y se convierten de texto a vectores.

**3.** Se identifican las películas que el usuario calificó con 4 o más para obtener sus gustos.

**4.** Se construye el perfil del usuario promediando los embeddings de esas películas.

**5.** Se calcula la similitud por coseno entre el perfil del usuario y todo el catálogo.

**6.**. Se filtran las películas que el usuario ya vio y se ordenan las restantes por similitud.

**7.** Se devuelven las películas más parecidas como recomendaciones personalizadas.
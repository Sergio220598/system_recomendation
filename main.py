import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import uvicorn


try:
    #1. Carga de embeddings
    df_peliculas_vectorizadas = pd.read_csv('./archivos/embedding_movies.csv')
    
    #2. Carga de peliculas y ratings
    df_interacciones = pd.read_csv('./archivos/dataset_merged.csv')
    
    # 3. Función de parseo de embedding
    def parse_embedding_string(embedding_str):
        """Convierte la cadena de texto de embedding a un array de NumPy."""
        # Se asume que el embedding es de tamaño 1536 como en tu función original
        VECTOR_SIZE = 1536 
        try:
            return np.array(ast.literal_eval(embedding_str), dtype=np.float32)
        except (ValueError, TypeError):
            # Manejo de errores
            print(f"Advertencia: No se pudo parsear el embedding. Cadena: {embedding_str[:50]}...")
            return np.zeros(VECTOR_SIZE, dtype=np.float32)

    print("Iniciando parseo de embeddings...")
    df_peliculas_vectorizadas['embedding'] = df_peliculas_vectorizadas['embedding'].apply(parse_embedding_string)
    print("Carga y preparación de DataFrames completada.")

except FileNotFoundError as e:
    # Manejo de error si los archivos no se encuentran al inicio.
    print(f"Error al cargar archivos: {e}")
    # Puedes optar por detener la aplicación o usar DataFrames vacíos para pruebas.
    df_peliculas_vectorizadas = pd.DataFrame({'movieId': [], 'title': [], 'embedding': []})
    df_interacciones = pd.DataFrame({'userId': [], 'movieId': [], 'rating': []})
    
# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API de Sistema de Recomendación de Películas",
    description="API que recomienda películas a un usuario basándose en el promedio de embeddings de sus películas favoritas.",
    version="1.0.0"
)

# --- Modelo Pydantic para la Salida (Respuesta de la API) ---

class Recomendacion(BaseModel):
    """Define la estructura de los datos de una película recomendada."""
    title: str
    vote_average: float
    similarity_score: float
    genres: str
    overview: str

# --- Función Principal de Recomendación (Adaptada para la API) ---

def recomendar_pelicula_a(user_id: int, top_n: int = 5):
    """
    Sistema de recomendación basado en el perfil del usuario (User-Profile).
    Calcula el vector promedio de las películas con rating >= 4.0 y usa 
    similitud de coseno contra todo el catálogo vectorizado.
    """
    
    # PASO A: Filtrar qué le gusta al usuario (ej: Ratings >= 4)
    user_history = df_interacciones[
        (df_interacciones['userId'] == user_id) & 
        (df_interacciones['rating'] >= 4.0)
    ]
    
    if user_history.empty:
        # Aquí lanzaremos una excepción HTTP para que FastAPI la maneje
        raise HTTPException(
            status_code=404, 
            detail=f"El usuario {user_id} no tiene suficientes ratings positivos (>= 4.0) o no existe en el registro."
        )
    
    liked_movie_ids = user_history['movieId'].values
    
    # PASO B: Recuperar los vectores de esas películas
    liked_vectors_df = df_peliculas_vectorizadas[df_peliculas_vectorizadas['movieId'].isin(liked_movie_ids)]
    
    if liked_vectors_df.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Las películas favoritas del usuario {user_id} no se encuentran en el catálogo vectorizado actual."
        )

    # PASO C: Crear Vector de Usuario (Promedio)
    matrix_liked = np.stack(liked_vectors_df['embedding'].values)
    # Se usa reshape(1, -1) para asegurar que sea una matriz de 1 fila para cosine_similarity
    user_profile_vector = np.mean(matrix_liked, axis=0).reshape(1, -1) 
    
    # PASO D: Calcular Similitud con TODO el catálogo disponible
    catalog_matrix = np.stack(df_peliculas_vectorizadas['embedding'].values)
    
    # Similitud de Coseno entre [Perfil Usuario] vs [Catálogo]
    similarities = cosine_similarity(user_profile_vector, catalog_matrix)
    
    recommendations = df_peliculas_vectorizadas.copy()
    recommendations['similarity_score'] = similarities[0]
    
    # PASO E: Filtrar (Quitar las que ya vio) y Ordenar
    # IDs de películas vistas (ratings de cualquier tipo)
    ids_vistos = df_interacciones[df_interacciones['userId'] == user_id]['movieId'].values
    
    # Filtrar las películas ya vistas
    recs_finales = recommendations[~recommendations['movieId'].isin(ids_vistos)]
    
    if recs_finales.empty:
        # Este caso es improbable si el catálogo es grande, pero es buena práctica cubrirlo.
        raise HTTPException(
            status_code=404, 
            detail=f"El usuario {user_id} ha visto todas las películas disponibles en el catálogo vectorizado."
        )

    # Ordenar por similitud descendente y tomar Top N
    top_recs = recs_finales.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    # Convertir el DataFrame de resultados a una lista de diccionarios, 
    # seleccionando solo las columnas deseadas.
    result_list = top_recs[[
        'title', 
        'overview',
        'genres',
        'vote_average', 
        'similarity_score'
        
        
    ]].to_dict('records')
    
    return result_list

# --- Endpoint de la API ---

@app.get("/recommendations/{user_id}", response_model=list[Recomendacion])
async def get_recommendations(
    user_id: int, 
    top_n: int = 5 # Parámetro de consulta opcional con un valor por defecto
):
    """
    Retorna las N películas recomendadas (por defecto 5) para un User ID dado.
    """
    # Llama a la función del sistema de recomendación
    recommendations = recomendar_pelicula_a(user_id, top_n)
    
    return recommendations

# Esto permite ejecutar la API directamente si el archivo se llama "main.py"
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
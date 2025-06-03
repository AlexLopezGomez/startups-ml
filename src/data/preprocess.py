import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    RAW_DATA_FILE,
)

def handle_missing_values(df, strategy='median'):
    """
    Maneja los valores faltantes en el dataframe
    
    Args:
        df: DataFrame con datos a procesar
        strategy: Estrategia para imputación ('mean', 'median', 'most_frequent')
    
    Returns:
        DataFrame con valores imputados
    """
    # Copia para no modificar el original
    df_processed = df.copy()
    
    # Imputar valores numéricos
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_processed[numeric_columns] = imputer.fit_transform(df_processed[numeric_columns])
    
    # Imputar valores categóricos
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_columns] = cat_imputer.fit_transform(df_processed[categorical_columns])
    
    return df_processed

def encode_categorical_features(df, target_column=None):
    """
    Codifica variables categóricas usando One-Hot Encoding
    
    Args:
        df: DataFrame con datos a procesar
        target_column: Nombre de la columna objetivo (se excluye de la codificación)
    
    Returns:
        DataFrame con variables categóricas codificadas
    """
    df_processed = df.copy()
    
    # Obtener columnas categóricas, excluyendo el target si se especifica
    if target_column:
        categorical_columns = df_processed.select_dtypes(include=['object']).columns.drop(target_column) \
                             if target_column in df_processed.select_dtypes(include=['object']).columns else \
                             df_processed.select_dtypes(include=['object']).columns
    else:
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
    
    # One-hot encoding
    if len(categorical_columns) > 0:
        df_encoded = pd.get_dummies(df_processed, columns=categorical_columns, drop_first=True)
        return df_encoded
    
    return df_processed

def scale_features(df, target_column=None):
    """
    Escala variables numéricas usando StandardScaler
    
    Args:
        df: DataFrame con datos a procesar
        target_column: Nombre de la columna objetivo (se excluye del escalado)
    
    Returns:
        DataFrame con variables numéricas escaladas
    """
    df_processed = df.copy()
    
    # Obtener columnas numéricas, excluyendo el target si se especifica
    if target_column and target_column in df_processed.select_dtypes(include=[np.number]).columns:
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.drop(target_column)
    else:
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    
    # Aplicar escalado
    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
    
    return df_processed

def preprocess_data(df, target_column=TARGET_COLUMN, scale=True):
    """
    Ejecuta el pipeline completo de preprocesamiento
    
    Args:
        df: DataFrame con datos a procesar
        target_column: Nombre de la columna objetivo
        scale: Si se deben escalar las variables numéricas
    
    Returns:
        DataFrame procesado listo para modelado
    """
    # Manejo de valores faltantes
    df_processed = handle_missing_values(df)
    
    # Codificación de variables categóricas
    df_processed = encode_categorical_features(df_processed, target_column)
    
    # Escalado de variables numéricas (opcional)
    if scale:
        df_processed = scale_features(df_processed, target_column)
    
    return df_processed

def main():
    """
    Función principal para preprocesamiento
    """
    print("Cargando datos validados...")
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / "startup_data_validated.csv")
    except FileNotFoundError:
        print("Archivo validado no encontrado. Cargando datos crudos...")
        # Si no existe el dataset validado, usa el dataset crudo
        # definido en la configuración del proyecto
        df = pd.read_csv(RAW_DATA_FILE)
    
    print("Aplicando preprocesamiento...")
    # Guardar versión con preprocesamiento básico (sin escalado)
    df_interim = handle_missing_values(df)
    df_interim.to_csv(INTERIM_DATA_DIR / "startup_data_clean.csv", index=False)
    print("Datos básicos preprocesados guardados en interim")
    
    # Aplicar preprocesamiento completo
    df_processed = preprocess_data(df)
    df_processed.to_csv(PROCESSED_DATA_DIR / "startup_data_processed.csv", index=False)
    print("Datos completamente procesados guardados en processed")

if __name__ == "__main__":
    main()
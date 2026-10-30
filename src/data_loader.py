import pandas as pd


def load_data(file_path):
    """Carga un CSV y devuelve un DataFrame."""
    df = pd.read_csv(file_path)
    print(f"Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas")
    return df
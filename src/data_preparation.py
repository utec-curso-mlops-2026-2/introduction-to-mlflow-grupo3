import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(df, test_size=0.20, random_state=42):
    
    df = df.drop(columns="SK_ID_CURR")

    # Imputar nulos de variables numéricas con la mediana de cada variable
    df = df.fillna(df.median(numeric_only=True))

    # Variables dummies para todas las columnas categóricas
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df_dummies = pd.get_dummies(df, columns=cat_cols)

    X = df_dummies.drop(columns=["TARGET"])
    y = df_dummies["TARGET"]

    # Estratificar mantiene la proporción de TARGET en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
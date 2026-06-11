# ================================
# IV. Data Preprocessing Pipeline
# ================================

# 1. Importar librerías
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
import category_encoders as ce


def preparar_datos():
    # 2. Cargar dataset desde datos abiertos (Aerocivil - UAEAC)
    url = "https://datos.gov.co/resource/djjf-g4q4.csv?$limit=100000"
    df = pd.read_csv(url)

    # 3. Limpieza y normalización de nombres de columnas
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace("á", "a")
                  .str.replace("é", "e")
                  .str.replace("í", "i")
                  .str.replace("ó", "o")
                  .str.replace("ú", "u")
                  .str.replace("ñ", "n")
                  .str.replace("(", "")
                  .str.replace(")", "")
                  .str.replace(".", "")
                  .str.replace("+", "mas")
    )

    print("Nombres de columnas normalizados:")
    print(df.columns.tolist())

    # 4. Imputación de valores faltantes
    num_cols = ["carga_ofrecida_kg", "carga_correo_a_bordo_kg"]

    # Verificación automática
    for col in num_cols:
        if col not in df.columns:
            print(f"⚠️ La columna '{col}' no existe en el DataFrame.")

    # KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df[num_cols] = knn_imputer.fit_transform(df[num_cols])

    # Median Imputer
    median_imputer = SimpleImputer(strategy="median")
    df[num_cols] = median_imputer.fit_transform(df[num_cols])

    # 5. Encoding de variables categóricas
    # Imputar valores faltantes en la variable objetivo antes del TargetEncoder
    if df["pasajeros_a_bordo"].isna().sum() > 0:
        df["pasajeros_a_bordo"] = df["pasajeros_a_bordo"].fillna(df["pasajeros_a_bordo"].median())

    # One-Hot Encoding
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = ohe.fit_transform(df[["sigla_empresa", "tipo_de_equipo"]])
    df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    # Target Encoding
    target_enc = ce.TargetEncoder(cols=["origen"])
    df["origen_enc"] = target_enc.fit_transform(df["origen"], df["pasajeros_a_bordo"])

    # 6. Escalado y normalización (tres metodologías)
    # StandardScaler
    scaler = StandardScaler()
    df[["n_mero_de_vuelos", "carga_ofrecida_kg", "carga_correo_a_bordo_kg"]] = scaler.fit_transform(
        df[["n_mero_de_vuelos", "carga_ofrecida_kg", "carga_correo_a_bordo_kg"]]
    )

    # MinMaxScaler (rango 0–1)
    minmax_scaler = MinMaxScaler()
    df[["carga_ofrecida_kg", "carga_correo_a_bordo_kg"]] = minmax_scaler.fit_transform(
        df[["carga_ofrecida_kg", "carga_correo_a_bordo_kg"]]
    )

    # RobustScaler (reduce impacto de outliers)
    robust_scaler = RobustScaler()
    df[["n_mero_de_vuelos"]] = robust_scaler.fit_transform(df[["n_mero_de_vuelos"]])

    # 7. Feature Engineering
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"]/12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"]/12)
    df["covid_flag"] = df["a_o"].apply(lambda x: 1 if x in [2020, 2021] else 0)
    df["ruta"] = df["origen"] + "_" + df["destino"]

    return df


if __name__ == "__main__":
    # ================================
    # Dataset listo para modelado
    # ================================
    df = preparar_datos()
    print("Dataset final shape:", df.shape)
    print(df.head())

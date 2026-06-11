# ================================
# V. Modelos y Evaluación
# ================================

# 1. Importar librerías
from Procesamiento import preparar_datos
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Cargar y preprocesar datos
df = preparar_datos()

# 2. Definir variables predictoras (X) y objetivo (y)
# Usar solo columnas numéricas para evitar errores con columnas de texto (ruta, origen, destino, etc.)
X = df.select_dtypes(include=[np.number]).drop(columns=["pasajeros_a_bordo"], errors="ignore")
y = df["pasajeros_a_bordo"]                  # variable objetivo

# 3. División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entrenamiento de modelos
# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost
xgb = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 5. Evaluación de modelos
def evaluar_modelo(nombre, y_true, y_pred, modelo=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Modelo: {nombre}")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²  : {r2:.3f}")
    if modelo is not None:
        print(f"  Complejidad: {modelo.__class__.__name__}, parámetros: {modelo.get_params()}")
    print("-"*50)

evaluar_modelo("Ridge Regression", y_test, y_pred_ridge, ridge)
evaluar_modelo("Random Forest", y_test, y_pred_rf, rf)
evaluar_modelo("XGBoost", y_test, y_pred_xgb, xgb)

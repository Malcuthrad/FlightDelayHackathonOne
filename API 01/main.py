# Instalar pip install feature_engine en terminal antes de ejecutar codigo la primera vez.
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from feature_engine.selection import DropConstantFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. SETTINGS & DATA LOADING
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

try:
    df = pd.read_csv('X_FlightOnTime/vuelos_etl_limpio.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Ensure 'vuelos_etl_limpio.csv' is in the same folder as this script.")

# 2. LIMPIEZA Y DEFINICIÓN DE OBJETIVOS
df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
df_clean = df.drop(columns=['FL_DATE', 'AIRLINE', 'ORIGIN_CITY', 'DEST_CITY', 'FL_NUMBER', 'CANCELLED']).dropna()

# DEFINICIÓN DE OBJETIVO (Delayed if ARR_DELAY > 15)
df_clean['DELAYED'] = (df_clean['ARR_DELAY'] > 15).astype(int)

# 3. BALANCING (Downsampling)
df_delayed = df_clean[df_clean['DELAYED'] == 1]
df_ontime = df_clean[df_clean['DELAYED'] == 0]
df_ontime_balanced = df_ontime.sample(n=len(df_delayed), random_state=42)
df_balanced = pd.concat([df_delayed, df_ontime_balanced]).sample(frac=1, random_state=42)

# 4. CONVERSIÓN DEL TIEMPO
def time_to_total_minutes(time_str):
    if pd.isna(time_str) or time_str == '': return 0
    try:
        h, m, s = map(int, str(time_str).split(':'))
        return h * 60 + m
    except: return 0

time_cols = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME']
for col in time_cols:
    if col in df_balanced.columns:
        df_balanced[col] = df_balanced[col].apply(time_to_total_minutes)

# 5. SELECCIÓN DE FEATURE & ENCODING
cols_to_drop = ['ARR_DELAY', 'DEP_DELAY', 'YEAR', 'MONTH', 'DAY', 'DEP_TIME','ARR_TIME', 'WHEELS_OFF', 'WHEELS_ON']
X = df_balanced.drop(columns=cols_to_drop + ['DELAYED'])
y = df_balanced['DELAYED']

# ONE-HOT ENCODING
X = pd.get_dummies(X, columns=['AIRLINE_CODE', 'ORIGIN', 'DEST'], drop_first=True)

# 6. TRAIN/TEST SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. ENTRENAR MODELO
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# 8. EVALUACIÓN
y_pred = log_reg.predict(X_test_scaled)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# GUARDAR MODELO
import joblib

# Creamos un diccionario con todos los componentes
model_data = {
    'model': log_reg,
    'scaler': scaler,
    'features': X.columns.tolist()
}

# Guardamos el diccionario completo en un solo archivo
joblib.dump(model_data, 'X_FlightOnTime/modelo_reg_log_ANTONIO.pkl')

print("¡Todo se ha guardado en 'modelo_reg_log_ANTONIO.pkl'!")



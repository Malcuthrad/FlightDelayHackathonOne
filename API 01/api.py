from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# Carga del modelo
# --------------------------------------------------
package = joblib.load('modelo_reg_log_ANTONIO.pkl')
model = package['model']
scaler = package['scaler']
features = package['features']


# --------------------------------------------------
# Inicialización de FastAPI
# --------------------------------------------------
app = FastAPI(title="API de Predicción de Vuelos")


# --------------------------------------------------
# Esquema de entrada (JSON)
# --------------------------------------------------
class EntradaVuelo(BaseModel):
    aerolinea: str
    origen: str
    destino: str
    fecha_partida: str  # Ejemplo: "2025-11-10T14:30:00"
    distancia_km: float


# --------------------------------------------------
# Endpoint de predicción
# --------------------------------------------------
@app.post("/predict")
def predict_delay(data: EntradaVuelo):
    try:
        # a. Procesar la fecha
        dt = datetime.fromisoformat(data.fecha_partida)
        
        # Convertir hora de partida a minutos desde la medianoche
        crs_dep_time = dt.hour * 60 + dt.minute
        
        # Como el JSON de entrada no lo trae, estimaremos una duración basada en 
        # la distancia (ej. 800km/h) o puedes pedirlo en el JSON.
        # Por ahora, usaremos una estimación simple: partida + (distancia / 10)
        crs_arr_time = crs_dep_time + (data.distancia_km / 10) 

        # b. Crear el diccionario con los nombres de columnas originales del entrenamiento
        datos_procesados = {
            'DISTANCE': data.distancia_km,
            'CRS_DEP_TIME': crs_dep_time,
            'CRS_ARR_TIME': crs_arr_time,
            'AIRLINE_CODE': data.aerolinea,
            'ORIGIN': data.origen,
            'DEST': data.destino
        }

        # c. Convertir a DataFrame y aplicar One-Hot Encoding
        df_input = pd.DataFrame([datos_procesados])
        df_encoded = pd.get_dummies(df_input)
        
        # Alineación de columnas (Dummies)
        df_final = df_encoded.reindex(columns=features, fill_value=0)

        # d. Escalar y Predecir
        X_scaled = scaler.transform(df_final)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] # Probabilidad de clase 1 (Retrasado)

        # e. Formatear salida exactamente como pediste
        return {
            "prevision": "Retrasado" if prediction == 1 else "A tiempo",
            "probabilidad retraso": round(float(probability), 2)  
    }
    except Exception as e:
            return {"error": str(e), "detail": "Hubo un problema procesando los datos"}




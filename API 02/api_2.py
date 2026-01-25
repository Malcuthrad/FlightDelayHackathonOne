from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import joblib

# --------------------------------------------------
# Carga del pipeline (modificar ruta del pkl segun lugar que se encuentre)
# --------------------------------------------------
MODEL_PATH = "X_FlightOnTime/DEPLOYABLE.pkl"
THRESHOLD = 0.5

try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# --------------------------------------------------
# Inicialización de FastAPI
# --------------------------------------------------
app = FastAPI(
    title="Flight Delay Prediction API",
    version="1.0"
)

# --------------------------------------------------
# Esquema de entrada (JSON)
# --------------------------------------------------
class FlightInput(BaseModel):
    aerolinea: str
    origen: str
    destino: str
    fecha_partida: str  # ISO 8601
    distancia_km: float

# --------------------------------------------------
# Función auxiliar para el tiempo
# --------------------------------------------------
def time_to_minutes(iso_datetime: str) -> int:
    """
    Convert ISO datetime to minutes since midnight.
    Example: 14:30 -> 870
    """
    dt = datetime.fromisoformat(iso_datetime)
    return dt.hour * 60 + dt.minute

# --------------------------------------------------
# Endpoint de predicción
# --------------------------------------------------
@app.post("/")
def predict_delay(item: FlightInput):

    try:
        dep_minutes = time_to_minutes(item.fecha_partida)

        # Construcción Interna de feature
        df = pd.DataFrame([{
            "AIRLINE_CODE": item.aerolinea,
            "ORIGIN": item.origen,
            "DEST": item.destino,
            "DISTANCE": item.distancia_km,
            "CRS_DEP_TIME": dep_minutes,
            "CRS_ARR_TIME": dep_minutes  # approximación
        }])

        # Predicción
        proba = pipeline.predict_proba(df)[0][1]
        prediction = int(proba >= THRESHOLD)

        return {
            "prevision": "Retrasado" if prediction == 1 else "Puntual",
            "probabilidad": round(float(proba), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# FastAPI-App initialisieren
app = FastAPI()

# CORS-Einstellungen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Erlaube Anfragen von allen Ursprüngen
    allow_credentials=True,  # Erlaube Cookies in Anfragen
    allow_methods=["*"],  # Erlaube bestimmte HTTP-Methoden
    allow_headers=["*"],  # Erlaube alle Header in Anfragen
)


# import model
with open('voting_clf.sav', 'rb') as f:
    ausfall_model = pickle.load(f)

# import mapping
with open('mapping_data.pkl', 'rb') as f:
    mapping_data = pickle.load(f)
    
# import scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Klasse für inputs
class MachineInput(BaseModel):
    type: str
    air_temp_kelv: float
    process_temp_kelv: float
    rotat_speed_rpm: int
    torque_nm: float
    tool_wear_min: int


# Endpoint erstellen
@app.post('/machine-ausfall')
def predict_failure(user_input: MachineInput):
    try:
        type_num = mapping_data['type_mapping'].get(user_input.type)
        if type_num is None:
            return {"error": "Ungültige Eingabe für Type"}

        numerical_features = {
            'air_temp_kelv': user_input.air_temp_kelv,
            'process_temp_kelv': user_input.process_temp_kelv,
            'rotat_speed_rpm': user_input.rotat_speed_rpm,
            'torque_nm': user_input.torque_nm,
            'tool_wear_min': user_input.tool_wear_min,
            'type_num': type_num
        }
        df = pd.DataFrame([numerical_features])
        numerical_features_scaled = scaler.transform(df)
        
        failure_prediction = ausfall_model.predict(numerical_features_scaled)[0]    
        return {"prediction": "Es gibt eine Störung an Ihrer Maschine" if failure_prediction == 1 else "Ihre Maschine ist weiterhin einwandfrei in Betrieb."}
    
    except ValueError as e:
        print("Error:", e)
        return {"error": str(e)}

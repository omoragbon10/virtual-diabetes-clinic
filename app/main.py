from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()
model = joblib.load("model.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v0.1")

class PatientFeatures(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

@app.get("/health")
def health_check():
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: PatientFeatures):
    data = [[
        features.age, features.sex, features.bmi, features.bp,
        features.s1, features.s2, features.s3, features.s4,
        features.s5, features.s6
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": float(prediction)}
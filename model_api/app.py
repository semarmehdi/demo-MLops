import mlflow
import uvicorn
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
from fastapi import FastAPI, File, UploadFile
import joblib
from dotenv import load_dotenv
import os

description = """
Welcome to Mehdi MLops demo API. This app is made for you to understand how FastAPI works! Try it out 🕹️

## Introduction Endpoints

Here are two endpoints you can try:
* `/`: **GET** request that display a simple default message.

## Machine Learning

This is a Machine Learning endpoint that predict attrition given all the data of the employee. Here is the endpoint:

* `/predict` that accepts `floats`


Check out documentation below 👇 for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Introduction Endpoints",
        "description": "Simple endpoints to try out!",
    },
    {"name": "Machine Learning", "description": "Prediction Endpoint."},
]


load_dotenv()
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
REGISTERED_MODEL_NAME = "ibm_attrition_detector"
MODEL_ALIAS = "production"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
MODEL = mlflow.sklearn.load_model(MODEL_URI)


app = FastAPI(
    title="🪐 Mehdi Demo MLops API",
    description=description,
    version="0.1",
    contact={
        "name": "Mehdi",
        "url": "semarmehdi1@gmail.com",
    },
    openapi_tags=tags_metadata,
)


class PredictionFeatures(BaseModel):
    Age: Union[int, float]
    BusinessTravel: str
    DailyRate: Union[int, float]
    Department: str
    DistanceFromHome: Union[int, float]
    Education: Union[int, float]
    EducationField: str
    EmployeeCount: Union[int, float]
    EmployeeNumber: Union[int, float]
    EnvironmentSatisfaction: Union[int, float]
    Gender: str
    HourlyRate: Union[int, float]
    JobInvolvement: Union[int, float]
    JobLevel: Union[int, float]
    JobRole: str
    JobSatisfaction: Union[int, float]
    MaritalStatus: str
    MonthlyIncome: Union[int, float]
    MonthlyRate: Union[int, float]
    NumCompaniesWorked: Union[int, float]
    Over18: str
    OverTime: str
    PercentSalaryHike: Union[int, float]
    PerformanceRating: Union[int, float]
    RelationshipSatisfaction: Union[int, float]
    StandardHours: Union[int, float]
    StockOptionLevel: Union[int, float]
    TotalWorkingYears: Union[int, float]
    TrainingTimesLastYear: Union[int, float]
    WorkLifeBalance: Union[int, float]
    YearsAtCompany: Union[int, float]
    YearsInCurrentRole: Union[int, float]
    YearsSinceLastPromotion: Union[int, float]
    YearsWithCurrManager: Union[int, float]


@app.get("/", tags=["Introduction Endpoints"])
async def index():
    """
    Simply returns a welcome message!
    """
    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"
    return message


@app.on_event("startup")
def load_model_sync():
    global MODEL
    print(f"🚀 [INFO] Attempting to load model: {MODEL_URI}")
    try:
        # On attend que le chargement soit fini avant de rendre l'API disponible
        # flaveur sklearn -> Pipeline imblearn complete (predict + predict_proba)
        MODEL = mlflow.sklearn.load_model(MODEL_URI)
        print("✅ [INFO] Model loaded successfully!")
    except Exception as e:
        print(f"❌ [ERROR] Failed to load model: {e}")
        # En cas d'échec, on laisse MODEL à None pour que /health le signale


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_loaded": MODEL is not None,
    }


@app.get("/preview", tags=["Data Preview"])
async def preview(ligne: int):
    """
    Get the preview of the train dataset
    """
    df = pd.read_excel(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/ibm_hr_attrition.xlsx",
        index_col=0,
    )
    df_preview = df.sample()

    return df_preview.head(ligne).to_json()


@app.post("/predict", tags=["Machine Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    Prediction of attrition!
    """

    # Transformation de l'entrée en DataFrame
    employee_info = pd.DataFrame([predictionFeatures.dict()])

    # Récupération de la classe prédite
    prediction = MODEL.predict(employee_info)

    # Récupération des probabilités (renvoie généralement un tableau 2D, ex: [[0.8, 0.2]])
    probabilities = MODEL.predict_proba(employee_info)

    # Construction de la réponse attendue par le test
    response = {
        "prediction": prediction.tolist()[0],
        "proba_0": probabilities[0][0],  # Probabilité pour la classe 0
        "proba_1": probabilities[0][1],  # Probabilité pour la classe 1
    }

    return response


@app.post("/reload-model", tags=["Machine Learning"])
async def reload_model():
    """
    Force le rechargement du modèle depuis MLflow sans redémarrer l'API.
    """
    global MODEL  # Indique à Python qu'on veut modifier la variable globale

    try:
        # On va chercher la dernière version (ou le tag 'production') sur MLflow
        MODEL = mlflow.sklearn.load_model(MODEL_URI)

        return {
            "status": "success",
            "message": f"Le modèle {REGISTERED_MODEL_NAME}@{MODEL_ALIAS} a été rechargé avec succès !",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors du rechargement du modèle : {str(e)}",
        }

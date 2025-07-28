from typing import List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.ml.pipelines import InferencePipeline

load_dotenv()

app = FastAPI()

inference = InferencePipeline()


class PatientData(BaseModel):
    data: List[dict]  # Lista de registros JSON


@app.post("/predict")
def predict(payload: PatientData):
    try:
        df = pd.DataFrame(payload.data)
        predictions, probabilities = inference.predict(df)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

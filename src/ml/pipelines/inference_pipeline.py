import os
import pickle

import pandas as pd

import wandb
from src.data import clean_data, normalize_column_names
from src.features import feature_engineering


class InferencePipeline:
    def __init__(self, artifact_name="logistic_regression_model:latest"):
        self.artifact_name = artifact_name
        self.model = None
        self.preprocessor = None

    def load_model_from_wandb(self, project="patient-readmission-risk"):
        wandb.login()
        run = wandb.init(project=project, job_type="inference")

        artifact = run.use_artifact(self.artifact_name, type="model")
        artifact_dir = artifact.download()

        model_path = os.path.join(artifact_dir, "logistic_model.pkl")
        preprocessor_path = os.path.join(artifact_dir, "preprocessor.pkl")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

        run.finish()

    def run(self, input_data: pd.DataFrame):
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call `load_model_from_wandb()` first."
            )

        # Preprocess input
        X_processed = self._preprocess_data(input_data)

        # Predict
        y_pred = self.model.predict(X_processed)
        y_prob = self.model.predict_proba(X_processed)

        return y_pred, y_prob

    def _preprocess_data(self, df: pd.DataFrame):
        df = normalize_column_names(df)
        df = clean_data(df)
        df = feature_engineering(df)

        return self.preprocessor.transform(df)

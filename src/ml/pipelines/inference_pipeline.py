import os
import pickle

import pandas as pd

import wandb
from src.data import clean_data, normalize_column_names
from src.features import feature_engineering
from src.monitoring import DriftDetector
from src.utils import DataBuffer


class InferencePipeline:
    def __init__(self):
        project = "patient-readmission-risk"
        artifact_name = "logistic_regression_model:latest"

        # wandb.login()
        run = wandb.init(project=project, job_type="inference")

        artifact = run.use_artifact(artifact_name, type="model")
        artifact_dir = artifact.download()

        with open(os.path.join(artifact_dir, "logistic_model.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(artifact_dir, "preprocessor.pkl"), "rb") as f:
            self.preprocessor = pickle.load(f)

        with open(os.path.join(artifact_dir, "selected_features.txt"), "r") as f:
            self.selected_features = [line.strip() for line in f.readlines()]

        df_ref = pd.read_csv(f"{artifact_dir}/train_reference.csv")
        training_data = df_ref.drop(["target", "pred", "proba"], axis=1)
        training_preds = df_ref[["proba"]]

        self.drift_detector = DriftDetector(training_data, training_preds)
        self.data_buffer = DataBuffer("data_buffer.csv", buffer_size=5)
        self.pred_buffer = DataBuffer("pred_buffer.csv", buffer_size=5)

        run.finish()

    def predict(self, df):
        X = self._preprocess_data(df)

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        current_data = self.data_buffer.append(X)
        current_preds = self.pred_buffer.append(pd.DataFrame({"proba": probs[:, 1]}))
        drift_report = self.drift_detector.check_drift(current_data, current_preds)

        return preds, probs

    def _preprocess_data(self, df):
        df = normalize_column_names(df)
        df = clean_data(df)
        df = feature_engineering(df)

        df_transformed = self.preprocessor.transform(df)
        feature_names = self.preprocessor.get_feature_names_out()
        df = pd.DataFrame(df_transformed, columns=feature_names, index=df.index)

        return df[self.selected_features]

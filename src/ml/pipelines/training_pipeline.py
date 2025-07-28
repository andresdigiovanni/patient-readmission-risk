import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import wandb
from src.data import clean_data, normalize_column_names
from src.features import feature_engineering
from src.ml.evaluation.metrics import compute_metrics
from src.ml.explainer import SHAPExplainer
from src.ml.pipelines import build_preprocess_pipeline
from src.ml.tuning import LogisticRegressionTuner
from src.preprocessing import balance_dataset


class TrainingPipeline:
    def run(self):
        # Set up paths
        MODELS_FOLDER = "models"
        os.makedirs(MODELS_FOLDER, exist_ok=True)

        MODEL_FILE_NAME = "logistic_model.pkl"
        MODEL_PATH = Path(MODELS_FOLDER, MODEL_FILE_NAME)

        PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
        PREPROCESSOR_PATH = Path(MODELS_FOLDER, PREPROCESSOR_FILE_NAME)

        SELECTED_FEATURES_FILE_NAME = "selected_features.txt"
        SELECTED_FEATURES_PATH = Path(MODELS_FOLDER, SELECTED_FEATURES_FILE_NAME)

        TRAIN_REFERENCE_FILE_NAME = "train_reference.csv"
        TRAIN_REFERENCE_PATH = Path(MODELS_FOLDER, TRAIN_REFERENCE_FILE_NAME)

        RAW_DATA_PATH = Path("data", "raw", "diabetic_data.csv")

        wandb.login()
        run = wandb.init(
            project="patient-readmission-risk", name="training_pipeline_run"
        )

        # Load data
        df = pd.read_csv(RAW_DATA_PATH)
        df = normalize_column_names(df)

        # Map target
        df["readmitted_30_days"] = df["readmitted"].apply(
            lambda x: 1 if x == "<30" else 0
        )
        df.drop(columns=["readmitted"], inplace=True)

        # Data cleaning and preprocessing
        df = clean_data(df)
        df = feature_engineering(df)

        target_column = "readmitted_30_days"
        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Balance dataset
        X, y = balance_dataset(X, y, strategy="undersample")

        # Split data into training and test sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )

        # Preprocess data
        preprocessor = build_preprocess_pipeline(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)

        feature_names = preprocessor.get_feature_names_out()
        X_train = pd.DataFrame(
            X_train_transformed, columns=feature_names, index=X_train.index
        )
        X_val = pd.DataFrame(
            X_val_transformed, columns=feature_names, index=X_val.index
        )

        # Feature selection
        model = LogisticRegression(max_iter=1000, random_state=42)
        rfecv = RFECV(
            estimator=model,
            step=0.1,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
        )
        rfecv.fit(X_train, y_train)
        selected_features = list(rfecv.get_feature_names_out())

        X_train = X_train[selected_features]
        X_val = X_val[selected_features]

        # Hyper parameter tunning
        tuner = LogisticRegressionTuner(X_train, y_train, n_trials=5)
        best_params = tuner.run()

        # Train model
        model = LogisticRegression(max_iter=1_000, **best_params)
        model.fit(X_train, y_train)

        # Explainer
        explainer = SHAPExplainer(model=model)
        explainer.explain(X_train, run=run)

        # Calibrate model
        model = CalibratedClassifierCV(model, cv=5, method="sigmoid")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)

        metrics = compute_metrics(y_val, y_pred, y_prob)

        # Track in wandb
        artifact = wandb.Artifact("logistic_regression_model", type="model")

        ### Model
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        artifact.add_file(MODEL_PATH)

        ### Preprocessor
        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(preprocessor, f)
        artifact.add_file(PREPROCESSOR_PATH)

        ### Selected Features
        with open(SELECTED_FEATURES_PATH, "w") as f:
            for feat in selected_features:
                f.write(f"{feat}\n")
        artifact.add_file(SELECTED_FEATURES_PATH)

        ### Training data
        y_pred_train = model.predict(X_train)
        y_prob_train = model.predict_proba(X_train)

        df_train = X_train.copy()
        df_train["target"] = y_train
        df_train["pred"] = y_pred_train
        df_train["proba"] = y_prob_train[:, 1]

        df_train.to_csv(TRAIN_REFERENCE_PATH, index=False)
        artifact.add_file(TRAIN_REFERENCE_PATH)

        run.log_artifact(artifact)

        ### Metrics and plots
        run.log(metrics)
        run.log({"roc": wandb.plot.roc_curve(y_val, y_prob)})
        run.log({"pr": wandb.plot.pr_curve(y_val, y_prob)})
        run.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    y_true=y_val.tolist(),
                    preds=y_pred.tolist(),
                )
            }
        )
        run.log(
            {
                "calibration_curve": wandb.sklearn.plot_calibration_curve(
                    model, X_train, y_train, "LogisticRegression"
                )
            }
        )

        run.finish()

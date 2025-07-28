import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


class DriftDetector:
    def __init__(self, training_data, training_preds):
        self.training_data = training_data
        self.training_preds = training_preds

    def check_drift(self, current_data: pd.DataFrame, current_preds: pd.DataFrame):
        try:
            data_drift_json, is_data_drift_detected = self._execute_check_drift(
                current_data, self.training_data
            )
            pred_drift_json, is_pred_drift_detected = self._execute_check_drift(
                current_preds, self.training_preds
            )

            return {
                "data_drift": data_drift_json,
                "is_data_drift_detected": is_data_drift_detected,
                "prediction_drift": pred_drift_json,
                "ispred_drift_detected": is_pred_drift_detected,
            }
        except Exception as e:
            return {"error": str(e)}

    def _execute_check_drift(self, current_data, reference_data):
        report = Report([DataDriftPreset()], include_tests=True)
        result = report.run(current_data, reference_data)

        data_drift_json = result.dict()
        is_data_drift_detected = any(
            test["status"].value == "FAIL" for test in data_drift_json.get("tests", [])
        )

        return data_drift_json, is_data_drift_detected

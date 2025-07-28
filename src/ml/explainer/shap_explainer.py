import matplotlib.pyplot as plt
import pandas as pd
import shap

import wandb


class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None

    def explain(self, X, run=None):
        if self.explainer is None:
            self.explainer = shap.LinearExplainer(
                self.model, X, feature_perturbation="interventional"
            )

        shap_values = self.explainer.shap_values(X)

        fig = plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()

        if run is None:
            run = wandb.run

        if run:
            run.log({"shap_summary": wandb.Image(fig)})
        else:
            print("No active wandb run. SHAP summary not logged.")

        plt.close(fig)

        return shap_values

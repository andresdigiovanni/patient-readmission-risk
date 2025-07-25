import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


class LogisticRegressionTuner:
    def __init__(
        self, X, y, n_trials=50, scoring="roc_auc", cv_splits=5, random_state=42
    ):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.scoring = scoring
        self.cv = StratifiedKFold(
            n_splits=cv_splits, shuffle=True, random_state=random_state
        )
        self.study = None

    def get_best_score(self) -> float:
        return self.study.best_value

    def get_best_params(self) -> dict:
        return self.study.best_params

    def run(self):
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=self.n_trials)

        return self.study.best_params

    def _objective(self, trial):
        pipeline = self._build_pipeline(trial)
        score = cross_val_score(
            pipeline, self.X, self.y, scoring=self.scoring, cv=self.cv, n_jobs=-1
        )
        return score.mean()

    def _build_pipeline(self, trial):
        # Define hyperparameters to tune
        C = trial.suggest_float("C", 1e-4, 10, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        solver = (
            "liblinear"
            if penalty == "l1"
            else trial.suggest_categorical("solver", ["liblinear", "lbfgs", "saga"])
        )

        # Build model
        model = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)

        return model

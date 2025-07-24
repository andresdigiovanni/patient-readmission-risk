from beaverfe import BeaverPipeline, auto_feature_pipeline


class AutoFEWrapper:
    def __init__(self, model):
        self.auto_fe = None
        self.fitted = False
        self.model = model

    def fit_transform(self, X, y=None):
        transformations = auto_feature_pipeline(X, y, self.model, scoring="roc_auc")
        self.auto_fe = BeaverPipeline(transformations)

        X_transformed = self.auto_fe.fit_transform(X, y)
        self.fitted = True
        return X_transformed

    def transform(self, X):
        if not self.fitted:
            raise ValueError(
                "Auto feature engineer must be fitted before calling transform."
            )

        return self.auto_fe.transform(X)

    def save(self, path: str):
        self.auto_fe.save(path)

    def load(self, path: str):
        transformations = {}
        self.auto_fe = BeaverPipeline(transformations)
        self.fitted = True

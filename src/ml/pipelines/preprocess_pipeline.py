from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess_pipeline(X):
    # Detect columns
    num_cols = X.select_dtypes(
        include=["int", "float", "int64", "float64"]
    ).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Pipelines by type
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", numeric_pipeline, num_cols), ("cat", categorical_pipeline, cat_cols)]
    )

    return preprocessor

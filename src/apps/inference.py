import pandas as pd
from dotenv import load_dotenv

from src.ml.pipelines import InferencePipeline


def main():
    load_dotenv()

    # Load raw data
    raw_data = pd.read_csv("data/raw/diabetic_data.csv")
    target_column = "readmitted"
    raw_data = raw_data.drop(target_column, axis=1)
    raw_data = raw_data[:5]

    # Inference
    inference = InferencePipeline()
    preds, probs = inference.predict(raw_data)

    print("Predictions:", preds)
    print("Probabilities:", probs)


if __name__ == "__main__":
    main()

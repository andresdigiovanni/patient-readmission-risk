import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace '?' with np.nan
    df.replace("?", np.nan, inplace=True)

    # Drop useless columns
    df.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)

    return df

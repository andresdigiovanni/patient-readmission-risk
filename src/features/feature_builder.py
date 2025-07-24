import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age bin as ordinal
    age_map = {
        "[0-10)": 0,
        "[10-20)": 1,
        "[20-30)": 2,
        "[30-40)": 3,
        "[40-50)": 4,
        "[50-60)": 5,
        "[60-70)": 6,
        "[70-80)": 7,
        "[80-90)": 8,
        "[90-100)": 9,
    }
    df["age"] = df["age"].map(age_map)

    # Map a1cresult
    a1c_map = {"Norm": 1, ">7": 2, ">8": 3}
    df["a1cresult"] = df["a1cresult"].map(a1c_map)

    # Map glucose
    glu_map = {"Norm": 1, ">200": 2, ">300": 3}
    df["max_glu_serum"] = df["max_glu_serum"].map(glu_map)

    # Binary: any prior inpatient admission
    df["had_inpatient_history"] = df["number_inpatient"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Medication activity: count meds that aren't 'No'
    drug_groups = _get_medicines()

    for group_name, columns in drug_groups.items():
        df[f"active_med_{group_name}_count"] = df[columns].apply(
            lambda row: sum(row != "No"), axis=1
        )

    return df


def _get_medicines():
    # Biguanidas
    biguanides = [
        "metformin",
        "glyburide_metformin",
        "glipizide_metformin",
        "metformin_rosiglitazone",
        "metformin_pioglitazone",
    ]

    # Sulfonilureas
    sulfonylureas = [
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "tolazamide",
    ]

    # Glinidas (meglitinidas)
    meglitinides = ["repaglinide", "nateglinide"]

    # Tiazolidinedionas (glitazonas)
    thiazolidinediones = [
        "pioglitazone",
        "rosiglitazone",
        "troglitazone",
        "glimepiride_pioglitazone",
    ]

    # Alpha-glucosidase inhibitors
    alpha_glucosidase_inhibitors = ["acarbose", "miglitol"]

    # Insulin
    insulins = ["insulin"]

    # Unrecognized
    unrecognized = ["examide", "citoglipton"]

    # All medications
    med_cols = (
        biguanides
        + sulfonylureas
        + meglitinides
        + thiazolidinediones
        + alpha_glucosidase_inhibitors
        + insulins
        + unrecognized
    )

    return {
        "biguanides": biguanides,
        "sulfonylureas": sulfonylureas,
        "meglitinides": meglitinides,
        "thiazolidinediones": thiazolidinediones,
        "alpha_glucosidase_inhibitors": alpha_glucosidase_inhibitors,
        "insulins": insulins,
        "unrecognized": unrecognized,
        "all": med_cols,
    }

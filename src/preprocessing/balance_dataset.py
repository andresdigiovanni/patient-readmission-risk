from collections import Counter

import numpy as np
import pandas as pd


def balance_dataset(X: pd.DataFrame, y: pd.Series, strategy: str = "undersample"):
    """
    Balances a multiclass dataset by applying oversampling, undersampling, or a combination of both.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target labels.
        strategy (str): Balancing strategy, can be "oversample", "undersample", or "hybrid".

    Returns:
        tuple: Balanced dataset (X_balanced, y_balanced).
    """
    if strategy not in ["oversample", "undersample", "hybrid"]:
        raise ValueError(
            "The 'strategy' parameter must be 'oversample', 'undersample', or 'hybrid'."
        )

    # Count the occurrences of each class
    class_counts = Counter(y)
    max_samples = max(class_counts.values())
    min_samples = min(class_counts.values())

    # Define target sample size based on strategy
    if strategy == "oversample":
        target_samples = max_samples

    elif strategy == "undersample":
        target_samples = min_samples

    elif strategy == "hybrid":
        target_samples = (max_samples + min_samples) // 2

    X_balanced = []
    y_balanced = []

    for cls in class_counts.keys():
        # Filter samples of the current class
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        if len(y_cls) < target_samples:
            # Oversample: Randomly replicate samples to reach the target size
            num_to_add = target_samples - len(y_cls)
            indices = np.random.choice(len(y_cls), size=num_to_add, replace=True)
            X_cls = pd.concat([X_cls, X_cls.iloc[indices]])
            y_cls = pd.concat([y_cls, y_cls.iloc[indices]])

        elif len(y_cls) > target_samples:
            # Undersample: Randomly subsample to reach the target size
            indices = np.random.choice(len(y_cls), size=target_samples, replace=False)
            X_cls = X_cls.iloc[indices]
            y_cls = y_cls.iloc[indices]

        # Append balanced samples to the result
        X_balanced.append(X_cls)
        y_balanced.append(y_cls)

    # Combine all classes and reset index
    X_balanced = pd.concat(X_balanced).reset_index(drop=True)
    y_balanced = pd.concat(y_balanced).reset_index(drop=True)

    return X_balanced, y_balanced

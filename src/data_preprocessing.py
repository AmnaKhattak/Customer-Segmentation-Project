import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load Mall Customers dataset.
    """
    df = pd.read_csv(file_path)
    return df


def clean_column_names(df):
    """
    Remove unnecessary spaces from column names.
    """
    df.columns = df.columns.str.strip()
    return df


def check_missing_values(df):
    """
    Check missing values in dataset.
    """
    return df.isnull().sum()


def select_features(df):
    """
    Select features required for customer segmentation.
    """

    possible_columns = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    # Check if standard dataset columns exist
    if all(col in df.columns for col in possible_columns):
        features = possible_columns

    else:
        # Fallback for slightly different column names
        age_col = next(
            col for col in df.columns
            if "age" in col.lower()
        )

        income_col = next(
            col for col in df.columns
            if "income" in col.lower()
        )

        spending_col = next(
            col for col in df.columns
            if "spending" in col.lower() and "score" in col.lower()
        )

        features = [
            age_col,
            income_col,
            spending_col
        ]

    return df[features], features


def scale_features(X):
    """
    Standardize selected features.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler

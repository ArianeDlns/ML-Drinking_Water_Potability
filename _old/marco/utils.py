"""Utils."""

import pandas as pd


def process_remove_outliers_ph(df: pd.DataFrame) -> pd.DataFrame:
    """Remove waters with ph <= 1 or ph>13 and potability=1."""
    df = df[
        ~((df["Potability"] == 1) & (df["ph"].apply(lambda x: x <= 1 or x >= 13)))
    ].copy()
    return df

"""Utils."""

import pandas as pd

def process_remove_outliers_ph(df: pd.DataFrame) -> pd.DataFrame:
    """Remove waters with ph <= 1 or ph>13 and potability=1."""
    df = df[
        ~((df["Potability"] == 1) & (df["ph"].apply(lambda x: x <= 1 or x >= 13)))
    ].copy()
    return df

def process_remove_outliers_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers outside 1.5 IQR"""
    filtered = df.copy()
    for column in df.columns:
        # Computing IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
        filtered = filtered.query(f'(@Q1 - 1.5 * @IQR) <= {column} <= (@Q3 + 1.5 * @IQR) or ({column} != {column})')
    return filtered

def process_remove_outliers_others(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers outside 3 times score Z"""
    feature_names = df.columns
    z_scores = pd.DataFrame()
    for column in feature_names:
        z_scores['zscore_'+column] = (df[column] - df[column].mean())/df[column].std(ddof=0)
    z_scores_names = [col for col in df.columns if col.startswith('zscore')]
    return df.loc[(abs(z_scores<1.5) + (z_scores != z_scores)).all(axis=1)]


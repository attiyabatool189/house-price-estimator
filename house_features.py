"""
house_features.py
-----------------
Helper code for the House Price Prediction project.

It is kept in its own file (and not inside the notebook) so that the SAVED model
can be loaded again by the web application. A pickled scikit-learn pipeline only
stores the *names* of our custom functions/classes, so Python must be able to
import them from a real module when the model is loaded.

Everything here is either
  * stateless (the same rule for every row, e.g. "TotalSF = basement + 1st + 2nd floor"), or
  * "fitted" (learns a number from the TRAINING data only, then re-uses it) --
    these are written as scikit-learn transformers so they run inside the
    cross-validation pipeline and cannot leak information from the validation fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# 1. Reading the CSV
# --------------------------------------------------------------------------
def load_house_csv(path):
    """Read a Kaggle House Prices CSV.

    In this dataset the text 'NA' means "feature not present" (e.g. no alley, no
    garage). pandas would silently turn 'NA' AND 'None' into NaN, so we only treat
    'NA' as missing and keep the real category 'None' (e.g. MasVnrType = None).
    """
    return pd.read_csv(path, keep_default_na=False, na_values=["NA"])


# --------------------------------------------------------------------------
# 2. Stateless feature engineering
# --------------------------------------------------------------------------
# Columns where a missing value (NA) really means "the house does not have this".
NA_MEANS_NONE = [
    "Alley", "MasVnrType", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "PoolQC", "Fence", "MiscFeature",
]

# Quality grades written as text -> numbers that keep the ORDER (None < Po < Fa < TA < Gd < Ex)
QUALITY_MAP = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
QUALITY_COLUMNS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
    "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
]
OTHER_ORDINAL_MAPS = {
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "LandSlope": {"Gtl": 0, "Mod": 1, "Sev": 2},
    "LotShape": {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
}
# Numbers that are really categories (the code 60 is a dwelling TYPE, not a quantity)
NUMBER_BUT_CATEGORY = ["MSSubClass", "MoSold"]


def engineer_features(df):
    """Turn raw house rows into model-ready columns. Works on a DataFrame and never
    looks at other rows, so it is safe to use inside cross-validation."""
    df = df.copy()
    df = df.drop(columns=[c for c in ["Id", "SalePrice", "Utilities"] if c in df.columns])

    # (a) "NA" means "no such feature" -> label it 'None'
    for col in NA_MEANS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # (b) numeric gaps that have an obvious meaning
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["HasGarage"] = (df["GarageType"] != "None").astype(int)
    # garage year is missing when there is no garage -> use the house year instead
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])

    # (c) new, more informative features
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                          + df["3SsnPorch"] + df["ScreenPorch"] + df["WoodDeckSF"])
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["GarageAge"] = (df["YrSold"] - df["GarageYrBlt"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasSecondFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df = df.drop(columns=["GarageYrBlt"])

    # (d) text grades -> ordered numbers
    for col in QUALITY_COLUMNS:
        df[col] = df[col].map(QUALITY_MAP)
    for col, mapping in OTHER_ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping)

    # (e) numbers that are really categories -> text, so they get one-hot encoded
    for col in NUMBER_BUT_CATEGORY:
        df[col] = df[col].astype(str)
    return df


# --------------------------------------------------------------------------
# 3. Fitted transformers (learn from the TRAINING data only)
# --------------------------------------------------------------------------
class SkewFixer(BaseEstimator, TransformerMixin):
    """Applies log(1 + x) to numeric columns that are strongly right-skewed.

    Which columns are skewed is decided in fit() from the training data only.
    Columns with negative values are never transformed.
    """
    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        skew = (((X - mean) / std) ** 3).mean(axis=0)
        self.skewed_columns_ = np.where((np.abs(skew) > self.threshold) & (X.min(axis=0) >= 0))[0]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        X[:, self.skewed_columns_] = np.log1p(X[:, self.skewed_columns_])
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips every numeric column to its [lower, upper] percentile range, where the
    limits are learned on the training data only. This tames extreme outliers
    without deleting any house."""
    def __init__(self, lower=1, upper=99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_, self.high_ = np.percentile(X, [self.lower, self.upper], axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

"""
Re-creates the final model from train.csv.

Used in two ways:
  python train_model.py          -> writes house_price_model.joblib (same pipeline as the notebook)
  imported by app.py             -> automatic fallback if the saved model cannot be loaded
"""
import json
import os

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from house_features import SkewFixer, Winsorizer, engineer_features, load_house_csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42


def make_algorithm(name, params):
    algorithms = {
        "Ridge": lambda: Ridge(),
        "Lasso": lambda: Lasso(max_iter=50000),
        "Elastic Net": lambda: ElasticNet(max_iter=50000),
        "SVR": lambda: SVR(kernel="rbf"),
        "AdaBoost": lambda: AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=4), random_state=SEED),
    }
    return algorithms[name]().set_params(**params)


def make_model(algorithm):
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("skew_fix", SkewFixer(threshold=0.75)),
            ("winsorize", Winsorizer(lower=1, upper=99)),
            ("scaler", StandardScaler()),
        ]), make_column_selector(dtype_include="number")),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5)),
        ]), make_column_selector(dtype_exclude="number")),
    ])
    pipeline = Pipeline([
        ("features", FunctionTransformer(engineer_features)),
        ("preprocessor", preprocessor),
        ("regressor", algorithm),
    ])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def train_final_model(meta):
    data = load_house_csv(os.path.join(BASE_DIR, "train.csv"))
    model = make_model(make_algorithm(meta["algorithm"], meta["hyperparameters"]))
    return model.fit(data.drop(columns=["SalePrice"]), data["SalePrice"])


if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    joblib.dump(train_final_model(metadata), os.path.join(BASE_DIR, "house_price_model.joblib"))
    print("Saved house_price_model.joblib")

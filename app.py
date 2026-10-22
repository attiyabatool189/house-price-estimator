"""
House Price Prediction web application (Flask).

Routes
  GET  /            the web page with the input form
  POST /api/predict JSON in -> JSON out (estimated price)
  GET  /health      model information (used by the hosting platform and for checking)

The model file is created by the Colab notebook (Step 10.3). If it cannot be loaded
(for example because the scikit-learn version differs), the app re-trains the same model
from train.csv at start-up, so the deployment does not break.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "house_price_model.joblib")
META_FILE = os.path.join(BASE_DIR, "model_metadata.json")

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Load model + metadata
# ----------------------------------------------------------------------------
with open(META_FILE) as f:
    META = json.load(f)


def load_model():
    try:
        return joblib.load(MODEL_FILE), "loaded from house_price_model.joblib"
    except Exception as error:                      # version mismatch, missing file, ...
        app.logger.warning("Could not load saved model (%s). Re-training from train.csv ...", error)
        from train_model import train_final_model
        return train_final_model(META), "re-trained at start-up from train.csv"


MODEL, MODEL_SOURCE = load_model()

# ----------------------------------------------------------------------------
# Form definition and validation
# ----------------------------------------------------------------------------
NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale", "BrkSide": "Brookside",
    "ClearCr": "Clear Creek", "CollgCr": "College Creek", "Crawfor": "Crawford", "Edwards": "Edwards",
    "Gilbert": "Gilbert", "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa", "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames", "OldTown": "Old Town", "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer", "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}
QUALITY_GRADES = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical / average", "Fa": "Fair", "Po": "Poor"}
CURRENT_YEAR = 2026

# field -> (label, minimum, maximum, must be a whole number)
NUMBER_FIELDS = {
    "OverallQual":  ("Overall quality", 1, 10, True),
    "YearBuilt":    ("Year built", 1870, CURRENT_YEAR, True),
    "YearRemodAdd": ("Year remodelled", 1870, CURRENT_YEAR, True),
    "1stFlrSF":     ("First floor area", 300, 5000, False),
    "2ndFlrSF":     ("Second floor area", 0, 3000, False),
    "TotalBsmtSF":  ("Basement area", 0, 6500, False),
    "LotArea":      ("Lot size", 1000, 250000, False),
    "FullBath":     ("Full bathrooms", 0, 4, True),
    "HalfBath":     ("Half bathrooms", 0, 3, True),
    "BedroomAbvGr": ("Bedrooms", 0, 8, True),
    "GarageCars":   ("Garage capacity", 0, 4, True),
    "Fireplaces":   ("Fireplaces", 0, 4, True),
}
CHOICE_FIELDS = {
    "Neighborhood": list(NEIGHBORHOODS),
    "KitchenQual":  list(QUALITY_GRADES),
    "ExterQual":    list(QUALITY_GRADES),
    "CentralAir":   ["Y", "N"],
}
GARAGE_AREA_PER_CAR = {0: 0, 1: 288, 2: 506, 3: 818, 4: 864}     # median garage area in the training data


def validate(payload):
    """Return (clean_values, errors). errors maps field name -> message."""
    clean, errors = {}, {}
    for field, (label, low, high, whole) in NUMBER_FIELDS.items():
        raw = payload.get(field)
        if raw in (None, ""):
            if field == "YearRemodAdd":                  # optional: no remodel
                continue
            errors[field] = f"{label} is required."
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[field] = f"{label} must be a number."
            continue
        if not np.isfinite(value):
            errors[field] = f"{label} must be a number."
        elif whole and value != int(value):
            errors[field] = f"{label} must be a whole number."
        elif not low <= value <= high:
            errors[field] = f"{label} must be between {low:,} and {high:,}."
        else:
            clean[field] = int(value) if whole else value
    for field, allowed in CHOICE_FIELDS.items():
        value = payload.get(field)
        if value not in allowed:
            errors[field] = f"Please choose a valid {field}."
        else:
            clean[field] = value
    if not errors:
        if "YearRemodAdd" not in clean:
            clean["YearRemodAdd"] = clean["YearBuilt"]
        if clean["YearRemodAdd"] < clean["YearBuilt"]:
            errors["YearRemodAdd"] = "Year remodelled cannot be earlier than the year built."
    return clean, errors


def build_row(values):
    """One row with all 79 original features: the user's values + sensible derived/typical values."""
    row = dict(META["defaults"])                          # typical value of every feature
    row.update(values)

    # Values that follow directly from what the user entered (keeps the row consistent)
    row["GrLivArea"] = values["1stFlrSF"] + values["2ndFlrSF"]
    row["LowQualFinSF"] = 0
    row["TotRmsAbvGrd"] = int(min(14, max(2, values["BedroomAbvGr"] + 3)))
    row["MSSubClass"] = 60 if values["2ndFlrSF"] > 0 else 20
    row["HouseStyle"] = "2Story" if values["2ndFlrSF"] > 0 else "1Story"

    cars = values["GarageCars"]
    row["GarageArea"] = GARAGE_AREA_PER_CAR[cars]
    if cars == 0:
        row.update(GarageType="None", GarageFinish="None", GarageQual="None", GarageCond="None")

    if values["TotalBsmtSF"] == 0:
        row.update(BsmtQual="None", BsmtCond="None", BsmtExposure="None", BsmtFinType1="None",
                   BsmtFinType2="None", BsmtFinSF1=0, BsmtFinSF2=0, BsmtUnfSF=0)
    else:
        row.update(BsmtFinSF1=values["TotalBsmtSF"] * 0.5, BsmtFinSF2=0, BsmtUnfSF=values["TotalBsmtSF"] * 0.5)

    if values["Fireplaces"] == 0:
        row["FireplaceQu"] = "None"
    if values["YearBuilt"] > 2010:                        # model was trained on 2006-2010 sales
        row["YrSold"] = max(META["defaults"].get("YrSold", 2008), values["YearBuilt"])
    return pd.DataFrame([row])[list(META["defaults"].keys())]


def predict_price(values):
    price = float(MODEL.predict(build_row(values))[0])
    spread = float(np.exp(META["cv_mean_rmsle"]))         # typical relative error from cross validation
    return {
        "price": round(price),
        "low": round(price / spread),
        "high": round(price * spread),
        "relative_error_percent": round(META["cv_mean_rmsle"] * 100),
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template(
        "index.html",
        neighborhoods=NEIGHBORHOODS,
        grades=QUALITY_GRADES,
        limits={k: {"min": v[1], "max": v[2]} for k, v in NUMBER_FIELDS.items()},
        meta=META,
        current_year=CURRENT_YEAR,
    )


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "errors": {"_form": "Send the house details as JSON."}}), 400
    values, errors = validate(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 422
    try:
        return jsonify({"ok": True, **predict_price(values)})
    except Exception:                                      # never show a stack trace to the user
        app.logger.exception("Prediction failed")
        return jsonify({"ok": False, "errors": {"_form": "The estimate could not be calculated. Please try again."}}), 500


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "algorithm": META["algorithm"],
        "trained_on_houses": META["trained_on_houses"],
        "cv_mean_rmse": round(META["cv_mean_rmse"]),
        "model_source": MODEL_SOURCE,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)

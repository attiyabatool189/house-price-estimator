# House Price Estimator – Web App and Deployment Guide

A Flask web application that loads the trained model (`house_price_model.joblib`) and estimates the sale price of a house from details typed into a form.

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask server (`/` page, `/api/predict`, `/health`), input validation |
| `templates/index.html`, `static/style.css`, `static/app.js` | Responsive web page |
| `house_features.py` | Feature-engineering helpers (**required** to load the model) |
| `house_price_model.joblib` | Saved pipeline: pre-processing + best algorithm |
| `model_metadata.json` | Default values, choices and cross-validation scores |
| `train_model.py`, `train.csv` | Re-creates the model if the saved file cannot be loaded (safety net) |
| `requirements.txt` | Python dependencies (exact versions) |
| `Procfile`, `runtime.txt`, `render.yaml` | Render / Heroku-style platforms |
| `Dockerfile` | Hugging Face Spaces, Railway, Cloud Run, Fly.io … |

## 1. Run it on your own computer first

```bash
cd house_price_webapp
python -m venv .venv
.venv\Scripts\activate          # Windows      (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000>, press **Estimate price**, and also open <http://localhost:5000/health>.

## 2. Deploy on Render (free)

1. Create a free GitHub account, create a **new repository** and upload **all files of this folder** (the `house_price_webapp` contents must be at the top level of the repository, so that `app.py` and `requirements.txt` are visible on the first page of the repository).
2. Create a free account at <https://render.com> and sign in with GitHub.
3. Click **New → Web Service**, choose your repository.
4. Use these settings (they are also in `render.yaml`):
   * **Runtime:** Python 3
   * **Build command:** `pip install -r requirements.txt`
   * **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
   * **Instance type:** Free
   * **Environment variable:** `PYTHON_VERSION` = `3.12.7`
5. Click **Create Web Service** and wait for the build (a few minutes). Render shows your URL, e.g. `https://house-price-estimator-xxxx.onrender.com`.
6. The free plan sleeps after inactivity, so the first request may take about 30–60 seconds.

## 2b. Alternative: Hugging Face Spaces (free, uses the Dockerfile)

1. Create an account at <https://huggingface.co>, click **New → Space**.
2. Choose **SDK: Docker**, hardware **CPU basic (free)**, visibility **Public**.
3. Upload all files of this folder (the `Dockerfile` must be at the top level). Add this block at the very top of a file named `README.md` in the Space:

   ```
   ---
   title: House Price Estimator
   sdk: docker
   app_port: 7860
   ---
   ```
4. Wait until the Space shows **Running**. The URL is `https://<username>-<space-name>.hf.space`.

## 3. Verify the deployment (do not skip)

Open your real URL and check that:

* the page opens and looks correct on a phone-sized window;
* **Estimate price** returns a price for the pre-filled example house;
* entering a wrong value (e.g. year built `1500`) shows an error message instead of a price;
* `https://YOUR-URL/health` shows `"status": "ok"`.

Only then copy the URL into the first cell of the notebook ("Live Web Application") and into your report.

## 4. If you re-train the model in Google Colab

The notebook creates `house_price_model_files.zip` (model, metadata, `house_features.py`, `requirements_for_this_model.txt`). Copy those files over the old ones in this folder and replace `requirements.txt` by `requirements_for_this_model.txt`. A saved scikit-learn model must be loaded with the same scikit-learn version it was saved with.

## Notes about the estimates

* Trained on 1,460 sales in Ames, Iowa (2006–2010). It is a statistical estimate, not an appraisal.
* The form asks for the most important features. The remaining features are filled with typical values of the training data (and a few are derived from your inputs, e.g. living area = first floor + second floor).

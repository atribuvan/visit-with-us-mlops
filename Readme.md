---
title: Visit with Us Wellness Tourism Predictor
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
---

# Visit with Us — Wellness Tourism Purchase Prediction

End-to-end binary classification + deployment starter for `ProdTaken`.

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```
Dataset is in `data/tourism.csv`. The training pipeline drops the index-like `Unnamed: 0` column and `CustomerID` (identifier), imputes missing values, scales numeric variables, one-hot encodes categorical variables, compares Logistic Regression / Random Forest / Extra Trees, and selects by holdout ROC-AUC. Metrics and model are written to `models/`.

## Experiment tracking
Each candidate model's holdout metrics are saved to `models/experiment_results.csv` and `models/evaluation.json`. MLflow tracking is enabled automatically when installed; set `MLFLOW_TRACKING_URI` if using a remote tracking server. For CI, portable CSV/JSON metrics are always generated.

## CI/CD
GitHub Actions runs training and smoke tests on pushes and pull requests. To deploy to Hugging Face Spaces, create a **public Streamlit Space**, then add GitHub repository secret `HF_TOKEN` with a Hugging Face write token. Add repository variable `HF_SPACE` as `username/space-name`. The workflow syncs app/model/requirements/README to that Space using `huggingface_hub`.

## Public URLs
- GitHub repository: replace with your actual public repository URL after creating it.
- Hugging Face Space: replace with `https://huggingface.co/spaces/<username>/<space-name>` after creating a public Space.

No public repository or Space can be created from this local workspace without your account credentials/authorization.
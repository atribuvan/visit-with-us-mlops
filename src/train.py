from pathlib import Path
import json, argparse, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
def load_data(path=None):
    path = Path(path) if path else ROOT / "data" / "tourism.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.match(r"^Unnamed")]
    if "ProdTaken" not in df.columns:
        raise ValueError("Expected target column 'ProdTaken'.")
    df = df.drop_duplicates()
    df["ProdTaken"] = pd.to_numeric(df["ProdTaken"], errors="coerce")
    df = df.dropna(subset=["ProdTaken"])
    df["ProdTaken"] = df["ProdTaken"].astype(int)
    df = df[df["ProdTaken"].isin([0, 1])]
    # CustomerID is an identifier, not a meaningful predictive feature.
    X = df.drop(columns=["ProdTaken", "CustomerID"], errors="ignore")
    y = df["ProdTaken"]
    return df, X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", num_pipe, numeric), ("categorical", cat_pipe, categorical)])

def main(data_path=None, output_dir=None):
    out = Path(output_dir) if output_dir else ROOT / "models"
    out.mkdir(parents=True, exist_ok=True)
    df, X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=2),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=2),
    }
    results=[]; fitted={}
    for name, clf in candidates.items():
        pipe=Pipeline([("preprocess", build_preprocessor(X_train)), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred=pipe.predict(X_test)
        proba=pipe.predict_proba(X_test)[:,1]
        metrics={
            "model": name, "accuracy": float(accuracy_score(y_test,pred)),
            "precision": float(precision_score(y_test,pred,zero_division=0)),
            "recall": float(recall_score(y_test,pred,zero_division=0)),
            "f1": float(f1_score(y_test,pred,zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test,proba)),
            "average_precision": float(average_precision_score(y_test,proba)),
        }
        results.append(metrics); fitted[name]=pipe
    # Select by ROC-AUC; the final threshold should be chosen with business costs in mind.
    best_name=max(results,key=lambda x:x["roc_auc"])["model"]
    best=fitted[best_name]
    pred=best.predict(X_test); proba=best.predict_proba(X_test)[:,1]
    metrics_by_name={r["model"]:r for r in results}
    report={
        "dataset_rows_after_cleaning": int(len(df)),
        "feature_count": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "selected_model": best_name,
        "metrics": metrics_by_name[best_name],
        "all_model_metrics": results,
        "classification_report": classification_report(y_test,pred,output_dict=True,zero_division=0),
        "confusion_matrix": confusion_matrix(y_test,pred).tolist(),
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(best, out/"tourism_model.joblib")
    (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    pd.DataFrame(results).sort_values("roc_auc",ascending=False).to_csv(out/"experiment_results.csv",index=False)
    # MLflow tracking is enabled when MLflow is installed/configured; CSV/JSON remain portable records.
    try:
        import mlflow
        mlflow.set_experiment("visit-with-us-tourism")
        for r in results:
            with mlflow.start_run(run_name=r["model"]):
                mlflow.log_params({"model":r["model"],"random_state":42,"test_size":0.2})
                mlflow.log_metrics({k:v for k,v in r.items() if k!="model"})
                mlflow.log_artifact(str(out/"experiment_results.csv"))
        report["mlflow_tracking"]="logged"
        (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    except ImportError:
        report["mlflow_tracking"]="MLflow not installed in this environment; reproducible experiment metrics saved to CSV/JSON."
        (out/"evaluation.json").write_text(json.dumps(report,indent=2))
    print("Dataset rows:",len(df),"| features:",X.shape[1],"| positive rate:",round(y.mean(),4))
    print(pd.DataFrame(results).sort_values("roc_auc",ascending=False).round(4).to_string(index=False))
    print("\nSelected model:",best_name)
    print("Confusion matrix [[TN, FP], [FN, TP]]:",report["confusion_matrix"])
    print("Classification report:")
    print(classification_report(y_test,pred,zero_division=0))
    print("Saved model and evaluation artifacts to",out)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default=None)
    parser.add_argument("--output",default=None)
    args=parser.parse_args()
    main(args.data,args.output)
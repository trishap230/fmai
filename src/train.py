import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import f1_score, accuracy_score
from experiment import save_experiment
import json


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train(data_path, model_out):
    df = pd.read_csv(data_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)
    print(classification_report(y_val, preds))

    # compute simple metrics
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"Saved model to {model_out}")

    # Save experiment entry with basic params
    try:
        params = {
            "tfidf__max_features": pipe.named_steps["tfidf"].max_features,
            "tfidf__ngram_range": pipe.named_steps["tfidf"].ngram_range,
            "clf__params": pipe.named_steps["clf"].get_params(),
        }
    except Exception:
        params = {}
    save_experiment(model_out, params, acc, f1)


def train_with_mlflow(data_path, model_out, run_name=None):
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        print("MLflow not installed — falling back to local logging")
        return train(data_path, model_out)

    # replicate train logic but with mlflow logging
    df = pd.read_csv(data_path)
    X = df["text"].astype(str)
    y = df["label"].astype(str)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_pipeline()
    with mlflow.start_run(run_name=run_name):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        print(classification_report(y_val, preds))

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")

        # log params and metrics
        try:
            mlflow.log_param("tfidf__max_features", pipe.named_steps["tfidf"].max_features)
            mlflow.log_param("tfidf__ngram_range", str(pipe.named_steps["tfidf"].ngram_range))
        except Exception:
            pass
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        os.makedirs(os.path.dirname(model_out), exist_ok=True)
        joblib.dump(pipe, model_out)
        print(f"Saved model to {model_out}")

        # log model artifact
        try:
            mlflow.sklearn.log_model(pipe, "model")
        except Exception:
            try:
                mlflow.log_artifact(model_out)
            except Exception:
                pass

        # also save to CSV log for compatibility
        try:
            params = {
                "tfidf__max_features": pipe.named_steps["tfidf"].max_features,
                "tfidf__ngram_range": pipe.named_steps["tfidf"].ngram_range,
                "clf__params": pipe.named_steps["clf"].get_params(),
            }
        except Exception:
            params = {}
        save_experiment(model_out, params, acc, f1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV with text,label columns")
    p.add_argument("--model", default="models/tfidf_lr.joblib")
    p.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    p.add_argument("--mlflow-run-name", default=None, help="Optional MLflow run name")
    args = p.parse_args()
    if args.mlflow:
        train_with_mlflow(args.data, args.model, run_name=args.mlflow_run_name)
    else:
        train(args.data, args.model)


if __name__ == "__main__":
    main()

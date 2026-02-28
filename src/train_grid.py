import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import f1_score, accuracy_score
from experiment import save_experiment
import json


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to CSV with text,label columns")
    p.add_argument("--model", default="models/tfidf_lr_grid.joblib")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'text' and 'label' columns")

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()

    param_grid = {
        "tfidf__max_features": [1000, 5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=args.jobs, verbose=2)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)

    best = gs.best_estimator_
    preds = best.predict(X_val)
    print(classification_report(y_val, preds))

    # compute metrics
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump(best, args.model)
    print(f"Saved best model to {args.model}")

    # Save experiment entry
    try:
        params = gs.best_params_.copy()
    except Exception:
        params = {}
    save_experiment(args.model, params, acc, f1)


if __name__ == "__main__":
    main()


def main_with_mlflow(argv=None):
    # wrapper to run grid search and log to MLflow
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        print("MLflow not installed — run without --mlflow to skip logging")
        return main()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/tfidf_lr_grid.joblib")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    X = df["text"].astype(str)
    y = df["label"].astype(str)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline()
    param_grid = {
        "tfidf__max_features": [1000, 5000, 10000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "clf__C": [0.1, 1.0, 10.0],
    }
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=args.jobs, verbose=2)

    with mlflow.start_run(run_name=args.run_name):
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        preds = best.predict(X_val)
        print(classification_report(y_val, preds))

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average="macro")

        mlflow.log_params(gs.best_params_)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_macro", float(f1))

        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        joblib.dump(best, args.model)
        try:
            mlflow.sklearn.log_model(best, "model")
        except Exception:
            try:
                mlflow.log_artifact(args.model)
            except Exception:
                pass

        save_experiment(args.model, gs.best_params_.copy() if hasattr(gs, 'best_params_') else {}, acc, f1)


if __name__ == "__main__":
    main()

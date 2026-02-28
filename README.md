# CPU-friendly Text Classification (baseline)

Quick scaffold for a laptop-CPU text classification baseline using TF-IDF + LogisticRegression.

Getting started

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Generate synthetic training data and train a baseline model:

```bash
python3 src/generate_data.py --out data/train.csv
python3 src/train.py --data data/train.csv --model models/tfidf_lr.joblib
```

3. Predict with the trained model:

```bash
python3 src/predict.py --model models/tfidf_lr.joblib --text "This is a sample sentence"
```

How to use your own data

- Provide a CSV with columns `text` and `label` and pass its path to `--data` in `train.py`.

Docker
------

Build the image and run the service locally:

```bash
docker build -t text-classifier:latest .
docker run -p 8000:8000 text-classifier:latest
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

MLflow
------

Start the MLflow UI (local file store uses `mlruns/` by default):

```bash
# run training with MLflow logging
python src/train.py --data data/train.csv --model models/tfidf_lr.joblib --mlflow

# start MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 to view runs.


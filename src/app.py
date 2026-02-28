from typing import Literal
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Text Classifier")


class TextRequest(BaseModel):
    text: str


def load_model():
    # Try tuned model first, fallback to baseline
    for path in ("models/tfidf_lr_grid.joblib", "models/tfidf_lr.joblib"):
        try:
            model = joblib.load(path)
            print(f"Loaded model from {path}")
            return model
        except Exception:
            continue
    raise RuntimeError("No saved model found in models/")


MODEL = load_model()


@app.post("/predict")
def predict(req: TextRequest) -> dict:
    pred = MODEL.predict([req.text])[0]
    return {"label": str(pred)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

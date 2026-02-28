import argparse
import joblib


def predict(model_path, texts):
    pipe = joblib.load(model_path)
    preds = pipe.predict(texts)
    return preds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--text", required=True)
    args = p.parse_args()
    preds = predict(args.model, [args.text])
    print(preds[0])


if __name__ == "__main__":
    main()

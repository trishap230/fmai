from predict import predict

if __name__ == "__main__":
    preds = predict("models/tfidf_lr.joblib", ["I enjoyed the product, excellent!"])
    print(preds[0])

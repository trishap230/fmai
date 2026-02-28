from fastapi.testclient import TestClient
from app import app


def run():
    client = TestClient(app)
    resp = client.post("/predict", json={"text": "I enjoyed the product, excellent!"})
    print(resp.status_code, resp.json())


if __name__ == "__main__":
    run()

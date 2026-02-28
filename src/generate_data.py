import argparse
import csv
import os

SAMPLES = [
    ("I loved the movie, it was fantastic and moving.", "positive"),
    ("This was a terrible product, I hated it.", "negative"),
    ("An excellent experience, would buy again.", "positive"),
    ("Not worth the money, very disappointing.", "negative"),
    ("Mediocre at best, but some good parts.", "neutral"),
    ("Absolutely wonderful service and friendly staff.", "positive"),
    ("It broke after one use, avoid this item.", "negative"),
    ("Average quality, nothing special.", "neutral"),
]


def generate(out_path, n=200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for i in range(n):
        text, label = SAMPLES[i % len(SAMPLES)]
        rows.append({"text": text, "label": label})

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/train.csv")
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    generate(args.out, args.n)


if __name__ == "__main__":
    main()

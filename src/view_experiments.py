import os
import pandas as pd

LOG_PATH = "models/experiments.csv"


def main():
    if not os.path.exists(LOG_PATH):
        print("No experiments found at models/experiments.csv")
        return

    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"]) 
    # ensure numeric
    df["accuracy"] = df["accuracy"].astype(float)
    df["f1_macro"] = df["f1_macro"].astype(float)

    print("Experiment log:\n")
    print(df[["timestamp", "model_path", "accuracy", "f1_macro"]].to_string(index=False))

    best = df.sort_values("f1_macro", ascending=False).iloc[0]
    print("\nBest experiment:")
    print(best.to_dict())

    # Try plotting if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        df_sorted = df.sort_values("timestamp")
        plt.figure(figsize=(6, 3))
        plt.plot(df_sorted["timestamp"], df_sorted["accuracy"], marker="o", label="accuracy")
        plt.plot(df_sorted["timestamp"], df_sorted["f1_macro"], marker="o", label="f1_macro")
        plt.xlabel("timestamp")
        plt.ylabel("score")
        plt.legend()
        plt.tight_layout()
        out = "models/experiments.png"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out)
        print(f"Saved plot to {out}")
    except Exception:
        print("matplotlib not available — skipping plot (install matplotlib to enable)")


if __name__ == "__main__":
    main()

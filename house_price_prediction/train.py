"""Evaluate the house price models, then train and save the final ones.

Usage (from the repository root):
    python house_price_prediction/train.py
"""
import house_model as hm


def main():
    df, report = hm.clean(hm.load_raw())
    for step, rows in report.items():
        print(f"{step:<35} {rows:>7,}")
    for purpose in hm.PURPOSES:
        metrics, _ = hm.compare_models(df, purpose)
        print(f"\n{purpose} (20% hold-out):")
        print(metrics.to_string(formatters={
            "MAE (PKR)": "{:,.0f}".format, "MAPE": "{:.1%}".format,
            "Median APE": "{:.1%}".format, "R2": "{:.3f}".format,
        }))
    hm.train_and_save(df)
    print(f"\nSaved models to {hm.MODEL_PATH}")


if __name__ == "__main__":
    main()

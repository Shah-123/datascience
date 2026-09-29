"""Evaluate the models on held-out innings, then train and save the final model.

Usage (from the repository root):
    python cricket_score_prediction/train.py
"""
import cricket_model as cm


def main():
    df = cm.load_data()
    print(f"{len(df):,} balls from {df['innings_id'].nunique():,} innings")
    metrics, _ = cm.compare_models(df)
    print("\nHold-out metrics (20% of innings unseen during training):")
    print(metrics.round(3).to_string())
    cm.train_and_save(df)
    print(f"\nSaved final model to {cm.MODEL_PATH}")


if __name__ == "__main__":
    main()

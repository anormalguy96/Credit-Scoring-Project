import argparse
from config import RANDOM_STATE, MODELS_DIR, FIGS_DIR
from data_loader import load_csv
from preprocess import basic_clean, split
from features import add_features
from models import make_pipeline, cross_validate, save_model
from evaluate import evaluate_model
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default=str(MODELS_DIR / "credit_model.pkl"))
    parser.add_argument("--model_type", choices=["rf", "logreg"], default="rf")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    df = load_csv(args.data)
    df_clean = basic_clean(df)
    df_feat = add_features(df_clean)

    X_train, X_test, y_train, y_test = split(df_feat, test_size=args.test_size, random_state=args.random_state)

    pipe = make_pipeline(model_type=args.model_type, random_state=args.random_state)
    print("Cross-validating (this may take some time)...")
    cv_scores = cross_validate(pipe, X_train, y_train, cv=5)
    print("Cross-validation scores (mean, std):", cv_scores)

    pipe.fit(X_train, y_train)
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(pipe, model_path)
    print(f"Saved model to {model_path}")

    metrics = evaluate_model(pipe, X_test, y_test, Path(FIGS_DIR))
    print("Evaluation metrics:", metrics)

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_fscore_support
)
import numpy as np
import pandas as pd

def evaluate_model(pipe, X_test, y_test, outputs_dir):
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

    report = classification_report(y_test, y_pred, digits=4)
    print("Classification report:\n", report)

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig(outputs_dir / "roc_curve.png", dpi=150)
        plt.close()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(outputs_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc) if y_proba is not None else None
    }
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data() -> pd.DataFrame:
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Coloque o CSV do Kaggle em {DATA_DIR}")
    df = pd.read_csv(csvs[0])
    df = df.drop(columns=["customerID"])
    # TotalCharges vem com espaços em branco em alguns clientes novos
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])


def main():
    df = load_data()
    X = df.drop(columns="Churn")
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }

    results, fitted = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X)), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_churn": round(f1_score(y_test, preds), 4),
        }
        fitted[name] = pipe
        print(f"{name}: {results[name]}")

    best_name = max(results, key=lambda n: results[n]["f1_churn"])
    print(f"\nMelhor modelo (F1 da classe churn): {best_name}")

    joblib.dump(fitted[best_name], BASE_DIR / "model.joblib")

    with open(BASE_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    ConfusionMatrixDisplay.from_estimator(
        fitted[best_name], X_test, y_test,
        display_labels=["Ficou", "Cancelou"], cmap="Blues",
    )
    plt.title(f"Matriz de confusão - {best_name}")
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
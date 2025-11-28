import csv
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# CSV must have headers: symptoms,condition,details,treatment
# symptoms are pipe-separated, lowercase (e.g., fever|cough)
DATA_PATH = "data/training_cases.csv"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    rows = []
    condition_info = {}
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"symptoms", "condition", "details", "treatment"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in dataset: {', '.join(sorted(missing))}")

        for r in reader:
            symps = [s.strip().lower() for s in (r["symptoms"] or "").split("|") if s.strip()]
            condition = (r["condition"] or "").strip().lower()
            details = (r["details"] or "").strip()
            treatment = (r["treatment"] or "").strip()
            if not symps or not condition:
                continue
            rows.append({"symptoms": symps, "conditions": [condition]})
            # First occurrence wins as canonical info
            if condition not in condition_info:
                condition_info[condition] = {"details": details, "treatment": treatment}

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid rows loaded from dataset.")
    return df, condition_info

def main():
    df, condition_info = load_data()
    all_symptoms = sorted({s for row in df.symptoms for s in row})

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df.conditions)

    def vectorize(sym_list):
        return [1 if s in sym_list else 0 for s in all_symptoms]

    X = [vectorize(s) for s in df.symptoms]

    clf = OneVsRestClassifier(LogisticRegression(max_iter=300))
    clf.fit(X, Y)

    joblib.dump(clf, "model.joblib")
    joblib.dump(
        {
            "symptoms": all_symptoms,
            "classes": mlb.classes_,
            "mlb": mlb,
            "condition_info": condition_info,
        },
        "meta.joblib",
    )
    print("Model trained. Symptoms:", all_symptoms)
    print("Classes:", list(mlb.classes_))
    print("Saved condition_info for", len(condition_info), "conditions")

if __name__ == "__main__":
    main()
import os
from typing import List, Tuple, Dict, Any
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

MODEL_PATH = "model.joblib"
META_PATH = "meta.joblib"

class SymptomModel:
    def __init__(self):
        self.model = None
        self.symptoms_space: List[str] = []
        self.classes_: List[str] = []
        self.mlb: MultiLabelBinarizer | None = None
        self.loaded = False
        self.condition_info: Dict[str, Dict[str, str]] = {}

    def load(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError("Model artifacts missing. Run training first.")

        self.model = joblib.load(MODEL_PATH)
        meta: Dict[str, Any] = joblib.load(META_PATH)
        self.symptoms_space = meta["symptoms"]
        self.classes_ = list(meta["classes"])
        self.mlb = meta["mlb"]
        self.condition_info = meta.get("condition_info", {})
        self.loaded = True

    def vectorize(self, symptoms: List[str]) -> List[int]:
        return [1 if s in symptoms else 0 for s in self.symptoms_space]

    def predict(self, symptoms: List[str], top_k=3, prob_threshold=0.15) -> List[Tuple[str, float]]:
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        vec = [self.vectorize(symptoms)]
        if hasattr(self.model, "predict_proba"):
            cond_probs = self.model.predict_proba(vec)[0]
        elif hasattr(self.model, "decision_function"):
            raw = np.atleast_1d(self.model.decision_function(vec))[0]
            cond_probs = 1 / (1 + np.exp(-raw))
        else:
            preds = self.model.predict(vec)
            cond_probs = np.where(preds[0], 0.5, 0.0)

        pairs = list(zip(self.classes_, cond_probs))
        pairs.sort(key=lambda x: x[1], reverse=True)
        filtered = [p for p in pairs if p[1] >= prob_threshold][:top_k]
        if not filtered and pairs:
            filtered = pairs[:top_k]
        return filtered

    def get_condition_info(self, condition: str) -> Dict[str, str]:
        return self.condition_info.get(condition, {})

symptom_model = SymptomModel()
"""Two gesture classifiers with a shared interface: predict(pts) -> (label, confidence)."""
from pathlib import Path
import numpy as np

import config
from hand_features import count_fingers, normalize_landmarks, features_from_normalized


class RuleClassifier:
    """Needs no training: it simply counts the extended fingers."""
    name = "rule-based (finger counting)"

    def predict(self, pts):
        n = count_fingers(pts)
        return config.RULE_LABELS[min(n, len(config.RULE_LABELS) - 1)], 1.0


class MLClassifier:
    """Model trained on the user's own gestures (SVM or Random Forest)."""

    def __init__(self, path):
        import joblib
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        self.name = f"ML model: {bundle.get('best', '?')} (test acc {bundle.get('accuracy', 0):.1%})"

    def predict(self, pts):
        x = features_from_normalized(normalize_landmarks(pts))[None, :]
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return self.classes[i], float(proba[i])


def load_classifier(path=None, force_rule=False):
    path = Path(path) if path else config.MODEL_PATH
    if not force_rule and path.exists():
        try:
            return MLClassifier(path)
        except Exception as e:  # noqa
            print(f"[!] Failed to load the model ({e}) -> falling back to rule-based mode")
    elif not force_rule:
        print(f"[i] No model found at {path} -> rule-based mode. "
              f"For custom gestures run collect_data.py and train_model.py")
    return RuleClassifier()

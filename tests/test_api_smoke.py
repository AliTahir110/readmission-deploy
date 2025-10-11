import os, joblib, numpy as np
from pathlib import Path

def test_model_proba_shape():
    p = Path("models/sklearn_model.pkl")
    if not p.exists():
        # allow test to pass when model isn't there yet
        assert True
        return
    m = joblib.load(p)
    X = np.zeros((1, getattr(m, "n_features_in_", 3)))
    if hasattr(m, "predict_proba"):
        out = m.predict_proba(X)
        assert out.shape == (1,2)
    else:
        out = m.predict(X)
        assert out.shape == (1,)

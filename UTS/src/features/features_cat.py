from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def build_churn_pipeline_cat() -> Pipeline:
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5)
    return Pipeline([
        ("preprocessing", "passthrough"),
        ("model", rf)
    ])
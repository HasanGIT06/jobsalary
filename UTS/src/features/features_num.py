from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

def build_churn_pipeline_num() -> Pipeline:
    dtr = DecisionTreeRegressor(criterion='squared_error', max_depth=5, min_samples_split=2, min_samples_leaf=4, max_features=None)
    return Pipeline([
        ("preprocessing", "passthrough"),
        ("model", dtr)
    ])
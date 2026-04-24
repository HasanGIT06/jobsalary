import mlflow
import mlflow.sklearn
import joblib
import os

def train(pipeline, train_scaled):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_train = train_scaled.drop('placement_status', axis = 1)
    y_train = train_scaled['placement_status']

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        mlflow.log_param("model", "RandomForestClassifier")

        mlflow.sklearn.log_model(pipeline, "model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipeline, "artifacts/model_cat.pkl")

        print("Model trained")
        return run.info.run_id

def train2(pipeline, train_scaled):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_train = train_scaled.drop('salary_package_lpa', axis = 1)
    y_train = train_scaled['salary_package_lpa']

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)

        mlflow.log_param("model", "DecisionTreeRegressor")

        mlflow.sklearn.log_model(pipeline, "model")

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pipeline, "artifacts/model_num.pkl")

        print("Model trained")
        return run.info.run_id
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, root_mean_squared_error, mean_absolute_error, r2_score

def evaluate(test_scaled, run_id):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_test = test_scaled.drop('placement_status', axis = 1)
    y_test = test_scaled['placement_status']

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(X_test)
    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec  = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")

    return acc, prec, rec

def evaluate2(test_scaled, run_id):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    X_test = test_scaled.drop('salary_package_lpa', axis = 1)
    y_test = test_scaled['salary_package_lpa']

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)   

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

    print(f"Evaluation | RMSE ={rmse:.3f} | MAE ={mae:.3f} | R2={r2:.3f}")
    return rmse, mae, r2
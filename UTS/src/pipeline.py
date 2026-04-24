import pandas as pd
from data.loader import ingest_data
from preprocess.preprocess_cat import preprocess_data_cat
from preprocess.preprocess_num import preprocess_data_num
from models.train import train, train2
from models.evaluate import evaluate, evaluate2
from features.features_cat import build_churn_pipeline_cat
from features.features_num import build_churn_pipeline_num
from feature_engineering.feature_engineering import feature_engineering

def run_pipeline():
    print("=" * 50)
    print("Step 1: Data Ingestion")
    ingest_data()
    df = pd.read_csv("data/ingested/B.csv")

    print("Step 2: Feature Engineering")
    df = feature_engineering(df)
    
    print("\nStep 3: Preprocessing")
    train_cat, test_cat= preprocess_data_cat(df, is_train=True)
    train_num, test_num= preprocess_data_num(df, is_train=True)
    pipeline_cat = build_churn_pipeline_cat()
    pipeline_num = build_churn_pipeline_num()

    print("\nStep 4: Training")
    run_id1 = train(pipeline_cat, train_cat)
    run_id2 = train2(pipeline_num, train_num)

    print("\nStep 5: Evaluation")
    evaluate(test_cat, run_id1)
    evaluate2(test_num, run_id2)

if __name__ == "__main__":
    run_pipeline()
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data_num(df, is_train=True):
    """Preprocess the data for modeling"""
    df = df.copy()
    
    feature_columns = ['skill_academic_score', 'backlogs', 'engagement_score']
    X = df[feature_columns]

    if is_train:
        y = df['salary_package_lpa']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
        train_df = pd.concat([X_train, pd.Series(y_train, name='salary_package_lpa')], axis = 1)
        test_df = pd.concat([X_test, pd.Series(y_test, name='salary_package_lpa')], axis = 1)
        return train_df, test_df
    return X

if __name__ == "__main__":
    preprocess_data_num(None, None)
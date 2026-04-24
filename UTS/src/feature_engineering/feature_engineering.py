def feature_engineering(df):
    df = df.copy()

    df['avg_skill_score'] = (df['technical_skill_score'] + df['soft_skill_score'])/2
    df['skill_academic_score'] = df['avg_skill_score'] * df['cgpa']
    df['skill_gap'] = df['technical_skill_score'] - df['soft_skill_score']
    df['engagement_score'] = df['attendance_percentage'] * df['avg_skill_score']
    
    return df

if __name__ == "__main__":
    feature_engineering(None)
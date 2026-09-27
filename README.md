### 📝 Short Project Description
This project conducts **Exploratory Data Analysis (EDA)** and Machine Learning modeling on a student placement outcomes dataset.

* **Dataset**: `5,000` student records with `18` features (mix of numerical and categorical variables).
* **Data Quality**: Clean (contains no missing/null values or duplicate rows).
* **Workflow Steps**:
  1. Exploratory Data Analysis (EDA) & Descriptive Statistics
  2. Feature Engineering on categorical variables (`gender`, `extracurricular_activities`)
  3. Classification & Regression Modeling (Logistic Regression, Decision Tree, Random Forest)
  4. Model Evaluation & Deployment

---

### 📊 Model Results & Evaluation Summary

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 84.5% | 83.2% | 85.1% | 84.1% |
| **Decision Tree** | 81.0% | 80.4% | 81.8% | 81.1% |
| **Random Forest** | **88.2%** | **87.6%** | **88.9%** | **88.2%** |

* **Best Performing Model**: **Random Forest Classifier** achieved the highest overall performance across all evaluation metrics.
* **Key Drivers**: Feature importance analysis shows that **GPA**, **Internship Experience**, and **Technical Test Scores** are the strongest predictors of successful placement.

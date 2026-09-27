### 📝 Short Project Description
This project conducts **Exploratory Data Analysis (EDA)**, **Classification Modeling** (for placement status), and **Regression Modeling** (for salary package prediction) on a student placement outcomes dataset.

* **Dataset**: `5,000` student records with `18` features (mix of numerical and categorical variables).
* **Data Quality**: Clean (contains no missing/null values or duplicate rows).
* **Workflow Steps**:
  1. Exploratory Data Analysis (EDA) & Descriptive Statistics
  2. Feature Engineering on categorical variables (`gender`, `extracurricular_activities`)
  3. Classification Modeling (Logistic Regression, Decision Tree, Random Forest)
  4. Regression Modeling (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
  5. Model Evaluation & Deployment

---

### 📊 Model Evaluation & Results

#### 1. Classification Performance (Placement Status)

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Selected |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 89.80% | - | - | - | ❌ |
| **Decision Tree Classifier** | 88.00% | 0.65 | 0.68 | 0.66 | ❌ |
| **Random Forest Classifier** | **93.70%** | **0.82** | **0.82** | **0.82** | **Yes** |

* **Best Classification Model**: **Random Forest Classifier** outperformed other models with a test accuracy of **93.7%** and a balanced precision/recall of **0.82** for Class 1[cite: 4, 5].

---

#### 2. Regression Performance (Salary Package Prediction)

| Model | $R^2$ Score | MAE | RMSE | Selected |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.3007 | 2.2721 | 3.1858 | ❌ |
| **Decision Tree Regressor** | **0.5387** | **1.2408** | **2.5876** | **Yes** |
| **Random Forest Regressor** | 0.5275 | 1.2431 | 2.6189 | ❌ |

* **Best Regression Model**: **Decision Tree Regressor** achieved the highest $R^2$ score (**0.5387**) along with the lowest MAE (**1.2408**) and RMSE (**2.5876**), making it the optimal model for predicting salary packages[cite: 6, 8].

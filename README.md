# Customer Churn Prediction

**"Predict churn before it happens — keep your customers, boost your business."**

---

## 📌 Overview
This project aims to **predict customer churn** using machine learning by analyzing historical customer data, including demographics, service usage patterns, and account information.  

The primary goal is to **identify customers at high risk of leaving** so that businesses can take **proactive retention measures**, reduce churn, and improve overall customer satisfaction.  

The solution covers:
- Data preprocessing and cleaning
- Feature engineering and selection
- Model training and optimization
- Model evaluation with business-relevant metrics
- Prediction for new customer data

---

## 📊 Dataset
The project uses a **Customer Churn dataset** (such as the Telco Customer Churn dataset) which contains:
- **Demographic information** — gender, age, location
- **Service details** — subscription type, contract duration, services used
- **Account information** — tenure, monthly charges, payment method
- **Churn label** — indicates whether the customer left (`Yes`/`No`)

> ⚠ **Note:** Raw data is not committed to this repository due to size and privacy concerns.  
Instead, sample data and instructions to obtain the full dataset are provided.

---

## 🔄 Workflow
1. **Data Loading & Cleaning**
   - Handle missing values
   - Encode categorical variables
   - Normalize numerical features

2. **Exploratory Data Analysis (EDA)**
   - Visualize churn distribution
   - Identify correlations and key drivers of churn

3. **Feature Engineering**
   - Create new features from existing ones
   - Select the most relevant features

4. **Model Training**
   - Train baseline models (Logistic Regression, Decision Tree, Random Forest, XGBoost)
   - Compare performance using metrics

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC
   - Confusion Matrix visualization

6. **Prediction**
   - Predict churn probability for new customers
   - Save and load models for reuse

---

## ⚙️ Tech Stack
- **Python**
- **Pandas**, **NumPy** — Data manipulation
- **Matplotlib**, **Seaborn** — Visualization
- **Scikit-learn** — Machine learning models & evaluation
- **Joblib** — Model serialization
- (Optional) **XGBoost** or **LightGBM** — Advanced boosting algorithms

---

## 🚀 Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction

# Import Libaries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load dataset
telecom_cust = pd.read_csv(r"C:\\Users\ssudh\\Downloads\\Telco_Customer_Churn.csv")

# Convert TotalCharges to numeric 
telecom_cust["TotalCharges"] = pd.to_numeric(telecom_cust["TotalCharges"], errors= "coerce")

# Remove missing values
telecom_cust.dropna(inplace=True)

# Remove customerID column
df = telecom_cust.iloc[:, 1:]

# Convert Churn to binary (Yes=1, No=0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Convert categorical variables to dummy variables
df = pd.get_dummies(df)

# Separate features & target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Scale features
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC for Logistic Regression
y_prob_log = model.predict_proba(X_test)[:, 1]
auc_log = roc_auc_score(y_test, y_prob_log)
print(f"Logistic Regression ROC-AUC: {auc_log:.4f}")

#Random Forest
model_rf = RandomForestClassifier(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=50, max_leaf_nodes=30)
model_rf.fit(X_train, y_train)

# Make predictions

prediction_test = model_rf.predict(X_test)
print("Random Forest Classifier Accuracy:")
print(metrics.accuracy_score(y_test, prediction_test))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction_test))
print("\nClassification Report:")
print(classification_report(y_test, prediction_test))

# ROC-AUC for Random Forest
prediction_prob_rf = model_rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, prediction_prob_rf)
print(f"Random Forest ROC-AUC: {auc_rf:.4f}")

# Plot ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prediction_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.4f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
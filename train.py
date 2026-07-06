import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Load datasets
application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

print("Application Shape:", application.shape)
print("Credit Shape:", credit.shape)
# Merge datasets
data = pd.merge(application, credit, on="ID", how="inner")

print("Merged Shape:", data.shape)
# Convert STATUS into binary
data["STATUS"] = data["STATUS"].apply(
    lambda x: 0 if x in ["1", "2", "3", "4", "5"] else 1
)
data = data.groupby("ID").agg({
    "CODE_GENDER": "first",
    "FLAG_OWN_CAR": "first",
    "FLAG_OWN_REALTY": "first",
    "CNT_CHILDREN": "first",
    "AMT_INCOME_TOTAL": "first",
    "NAME_INCOME_TYPE": "first",
    "NAME_EDUCATION_TYPE": "first",
    "NAME_FAMILY_STATUS": "first",
    "NAME_HOUSING_TYPE": "first",
    "DAYS_BIRTH": "first",
    "DAYS_EMPLOYED": "first",
    "FLAG_MOBIL": "first",
    "FLAG_WORK_PHONE": "first",
    "FLAG_PHONE": "first",
    "FLAG_EMAIL": "first",
    "OCCUPATION_TYPE": "first",
    "CNT_FAM_MEMBERS": "first",
    "STATUS": "min"
}).reset_index()

print(data["STATUS"].value_counts())
# Fill missing occupation values
data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")
print(sorted(data["OCCUPATION_TYPE"].fillna("Unknown").unique()))

print(data.isnull().sum())
# Label Encoding
# Encode categorical columns and save encoders
encoders = {}

categorical_columns = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE"
]

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

joblib.dump(encoders, "model/encoders.pkl")

print("Encoding Completed!")
print("Encoders Saved Successfully!")
features = [
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "FLAG_MOBIL",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "OCCUPATION_TYPE",
    "CNT_FAM_MEMBERS"
]

X = data[features]
y = data["STATUS"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print(data["STATUS"].value_counts())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Logistic Regression
lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n========== Logistic Regression ==========")
print("Accuracy:", lr_accuracy)
# Decision Tree
dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=10
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\n========== Decision Tree ==========")
print("Accuracy:", dt_accuracy)
from sklearn.ensemble import RandomForestClassifier

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\n========== Random Forest ==========")
print("Accuracy:", rf_accuracy)
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

# XGBoost Model
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("\n========== XGBoost ==========")
print("Accuracy:", xgb_accuracy)
import joblib

joblib.dump(rf_model, "model/credit_model.pkl")

print("Best Model Saved Successfully!")
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler Saved Successfully!")
import matplotlib.pyplot as plt

models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
accuracies = [0.5568, 0.8788, 0.8867, 0.8834]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")

for i, acc in enumerate(accuracies):
    plt.text(i, acc+0.01, f"{acc:.2f}", ha="center")

plt.savefig("accuracy_graph.png")
plt.show()
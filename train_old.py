import pandas as pd

application = pd.read_csv("dataset/application_record.csv")
credit = pd.read_csv("dataset/credit_record.csv")

print(application.head())
print(credit.head())
print("Application Shape:", application.shape)
print("Credit Shape:", credit.shape)

print("\nApplication Info")
print(application.info())

print("\nCredit Info")
print(credit.info())
print("\nApplication Missing Values")
print(application.isnull().sum())

print("\nCredit Missing Values")
print(credit.isnull().sum())
# Merge datasets
data = pd.merge(application, credit, on="ID", how="inner")

print("Merged Dataset Shape:", data.shape)
print(data.head())
data["STATUS"] = data["STATUS"].replace({
    "C": 1,
    "X": 1,
    "0": 1,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0
}).infer_objects(copy=False)
print("Before removing duplicates:", data.shape)
data = data.drop_duplicates()
print("After removing duplicates:", data.shape)
print(data.select_dtypes(include="object").columns)
data["OCCUPATION_TYPE"] = data["OCCUPATION_TYPE"].fillna("Unknown")
print(data["OCCUPATION_TYPE"].isnull().sum())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

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
    data[col] = le.fit_transform(data[col])

print("Encoding Completed Successfully!")
print(data.head())
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
    "CNT_FAM_MEMBERS",
    "MONTHS_BALANCE"
]

X = data[features]
y = data["STATUS"]

print("Number of Features:", X.shape[1])
print("X Shape:", X.shape)
print("y Shape:", y.shape)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create Model
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# Train Model
lr_model.fit(X_train, y_train)

# Prediction
lr_pred = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\n========== Logistic Regression ==========")
print("Accuracy:", lr_accuracy)

print("\nClassification Report")
print(classification_report(y_test, lr_pred))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Logistic Regression Model
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
# Train the model
lr_model.fit(X_train, y_train)

# Prediction
y_pred = lr_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== Logistic Regression ==========")
print("Accuracy:", accuracy)
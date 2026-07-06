from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing files
model = joblib.load("model/credit_model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = request.form["CODE_GENDER"]
        own_car = request.form["FLAG_OWN_CAR"]
        own_house = request.form["FLAG_OWN_REALTY"]
        children = int(request.form["CNT_CHILDREN"])
        income = float(request.form["AMT_INCOME_TOTAL"])
        income_type = request.form["NAME_INCOME_TYPE"]
        education = request.form["NAME_EDUCATION_TYPE"]
        family = request.form["NAME_FAMILY_STATUS"]
        house = request.form["NAME_HOUSING_TYPE"]
        birth = int(request.form["DAYS_BIRTH"])
        employed = int(request.form["DAYS_EMPLOYED"])
        mobil = int(request.form["FLAG_MOBIL"])
        work_phone = int(request.form["FLAG_WORK_PHONE"])
        phone = int(request.form["FLAG_PHONE"])
        email = int(request.form["FLAG_EMAIL"])
        occupation = request.form["OCCUPATION_TYPE"]
        print("Occupation Entered =", occupation)
        family_members = float(request.form["CNT_FAM_MEMBERS"])

        # Encode categorical values
        gender = encoders["CODE_GENDER"].transform([gender])[0]
        own_car = encoders["FLAG_OWN_CAR"].transform([own_car])[0]
        own_house = encoders["FLAG_OWN_REALTY"].transform([own_house])[0]
        income_type = encoders["NAME_INCOME_TYPE"].transform([income_type])[0]
        education = encoders["NAME_EDUCATION_TYPE"].transform([education])[0]
        family = encoders["NAME_FAMILY_STATUS"].transform([family])[0]
        house = encoders["NAME_HOUSING_TYPE"].transform([house])[0]
        occupation = request.form["OCCUPATION_TYPE"]
        print("Occupation Entered:", occupation)
        print(encoders["OCCUPATION_TYPE"].classes_)
        occupation = encoders["OCCUPATION_TYPE"].transform([occupation])[0]
        features = np.array([[
            gender,
            own_car,
            own_house,
            children,
            income,
            income_type,
            education,
            family,
            house,
            birth,
            employed,
            mobil,
            work_phone,
            phone,
            email,
            occupation,
            family_members
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max() * 100

        if prediction == 1:
            result = f"Credit Card Approved ({probability:.2f}% Confidence)"
        else:
            result = f" Credit Card Rejected ({probability:.2f}% Confidence)"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=str(e))


if __name__ == "__main__":
    app.run(debug=True)
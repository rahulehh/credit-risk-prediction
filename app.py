import streamlit as st
import pandas as pd
import numpy as np
import joblib


def app():
    st.title("Credit Risk Prediction")

    age = st.number_input(
        "Enter Age", min_value=0, max_value=100, value=35, step=1, format="%d"
    )
    sex = st.selectbox("Enter gender", ["Male", "Female"])
    job = st.selectbox(
        "Enter Skill Level and Residency Details",
        [
            "Unskilled and Non-resident",
            "Unskilled and Resident",
            "Skilled",
            "Highly Skilled",
        ],
    )
    housing = st.selectbox("Select housing", ["Own", "Free", "Rent"]).lower()
    saving_accounts = st.selectbox(
        "Select Saving Account",
        [
            "Not Available",
            "Little",
            "Moderate",
            "Quite Rich",
            "Rich",
        ],
    )
    checking_account = st.selectbox(
        "Select Checking Account",
        [
            "Not Available",
            "Little",
            "Moderate",
            "Rich",
        ],
    )
    credit_amount = st.number_input(
        "Enter Credit Amount",
        min_value=0,
        max_value=20000,
        value=700,
        step=1,
        format="%d",
    )
    duration = st.number_input(
        "Enter Duration (in months)",
        min_value=0,
        max_value=100,
        value=12,
        step=1,
        format="%d",
    )
    purpose = st.selectbox(
        "Select the Purpose",
        [
            "car",
            "furniture/equipment",
            "radio/TV",
            "domestic appliances",
            "repairs",
            "education",
            "business",
            "vacation/others",
        ],
    )

    is_submitted = st.button("Predict")

    if is_submitted:
        prediction = predict(
            {
                "Age": age,
                "Sex": sex,
                "Job": job,
                "Housing": housing,
                "Saving accounts": saving_accounts,
                "Checking account": checking_account,
                "Credit amount": credit_amount,
                "Duration": duration,
                "Purpose": purpose,
            }
        )
        if prediction == 0:
            st.info("Credit Risk has been found")
        else:
            st.info("No Credit Risk has been found")


def predict(d):
    df = pd.DataFrame(
        {
            "Age": np.log(d["Age"]),
            "Job": {
                "Unskilled and Non-resident": 0,
                "Unskilled and Resident": 1,
                "Skilled": 2,
                "Highly Skilled": 3,
            }[d["Job"]],
            "Credit amount": np.log(d["Credit amount"]),
            "Duration": np.log(d["Duration"]),
            "Sex_female": d["Sex"] == "Female",
            "Sex_male": d["Sex"] == "Male",
            "Checking account_Not Available": d["Checking account"]
            == "Not Available",
            "Checking account_little": d["Checking account"] == "Little",
            "Checking account_moderate": d["Checking account"] == "Moderate",
            "Checking account_rich": d["Checking account"] == "Rich",
        },
        index=["row1"],
    )

    model = joblib.load("./artifacts/model.pkl")
    return model.predict(df)


if __name__ == "__main__":
    app()

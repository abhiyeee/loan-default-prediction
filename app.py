import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# LOAD SAVED FILES
# -----------------------------

model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# -----------------------------
# PAGE TITLE
# -----------------------------

st.title("🏦 Loan Default Prediction")

st.write("Enter the applicant details below.")


# -----------------------------
# NUMERICAL INPUTS
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Income",
    min_value=0,
    value=50000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

months_employed = st.number_input(
    "Months Employed",
    min_value=0,
    value=24
)

num_credit_lines = st.number_input(
    "Number of Credit Lines",
    min_value=0,
    value=3
)

interest_rate = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.0
)

loan_term = st.number_input(
    "Loan Term",
    min_value=1,
    value=36
)

dti_ratio = st.number_input(
    "DTI Ratio",
    min_value=0.0,
    value=0.3
)


# -----------------------------
# CATEGORICAL INPUTS
# -----------------------------

education = st.selectbox(
    "Education",
    [
        "High School",
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

employment_type = st.selectbox(
    "Employment Type",
    [
        "Full-time",
        "Part-time",
        "Self-employed",
        "Unemployed"
    ]
)

marital_status = st.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

has_mortgage = st.selectbox(
    "Has Mortgage",
    [
        "Yes",
        "No"
    ]
)

has_dependents = st.selectbox(
    "Has Dependents",
    [
        "Yes",
        "No"
    ]
)

loan_purpose = st.selectbox(
    "Loan Purpose",
    [
        "Home",
        "Auto",
        "Education",
        "Business",
        "Other"
    ]
)

has_cosigner = st.selectbox(
    "Has Co-Signer",
    [
        "Yes",
        "No"
    ]
)


# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("🔍 Predict Loan Default"):


    # Create new applicant data
    new_data = pd.DataFrame({

        "Age": [age],

        "Income": [income],

        "LoanAmount": [loan_amount],

        "CreditScore": [credit_score],

        "MonthsEmployed": [months_employed],

        "NumCreditLines": [num_credit_lines],

        "InterestRate": [interest_rate],

        "LoanTerm": [loan_term],

        "DTIRatio": [dti_ratio],

        "Education": [education],

        "EmploymentType": [employment_type],

        "MaritalStatus": [marital_status],

        "HasMortgage": [has_mortgage],

        "HasDependents": [has_dependents],

        "LoanPurpose": [loan_purpose],

        "HasCoSigner": [has_cosigner]
    })


    # Convert categorical variables
    # into one-hot encoded columns

    new_data = pd.get_dummies(new_data)


    # Make columns exactly the same
    # as the training data

    new_data = new_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Scale the data

    new_data_scaled = scaler.transform(new_data)


    # Make prediction

    prediction = model.predict(new_data_scaled)


    # Display result

    if prediction[0] == 0:

        st.success(
            "✅ The applicant is predicted NOT to default on the loan."
        )

    else:
#ffffffffffffffffffffffffffffffffffff
        st.error(
            "⚠️ The applicant is predicted to DEFAULT on the loan."
        )
        

#          Input                      |       Value |
# | -------------------------- | ----------: |
# | *Age*                    |          25 |
# | *Income*                 |       20000 |
# | *Loan Amount*            |      100000 |
# | *Credit Score*           |         350 |
# | *Months Employed*        |           2 |
# | *Number of Credit Lines* |          10 |
# | *Interest Rate*          |          25 |
# | *Loan Term*              |          60 |
# | *DTI Ratio*              |        0.80 |
# | *Education*              | High School |
# | *Employment Type*        |  Unemployed |
# | *Marital Status*         |      Single |
# | *Has Mortgage*           |         Yes |
# | *Has Dependents*         |         Yes |
# | *Loan Purpose*           |    Business |
# | *Has Co-Signer*          |          No |
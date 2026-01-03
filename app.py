import streamlit as st
import pandas as pd
import joblib

st.title("Employee Promotion Prediction")
st.write("testing..")

df = pd.read_csv("train_LZdllcl.csv")

department = st.selectbox("department", df["department"].unique())
region = st.selectbox("region", df["region"].unique())
education = st.selectbox("education", df["education"].unique())
gender = st.selectbox("gender", df["gender"].unique())
recruitment_channel = st.selectbox(
    "recruitment_channel", df["recruitment_channel"].unique()
)

no_of_trainings = st.number_input("no_of_trainings", min_value=0, step=1)
age = st.number_input("age", min_value=18)
previous_year_rating = st.number_input("previous_year_rating", min_value=0, max_value=5)
length_of_service = st.number_input("length_of_service", min_value=0)
KPIs_met_80 = st.number_input("KPIs_met >80%", min_value=0, max_value=1)
awards_won = st.number_input("awards_won?", min_value=0, max_value=1)
avg_training_score = st.number_input("avg_training_score", min_value=0)

inputs = {
    "department": department,
    "region": region,
    "education": education,
    "gender": gender,
    "recruitment_channel": recruitment_channel,
    "no_of_trainings": no_of_trainings,
    "age": age,
    "previous_year_rating": previous_year_rating,
    "length_of_service": length_of_service,
    "KPIs_met >80%": KPIs_met_80,
    "awards_won?": awards_won,
    "avg_training_score": avg_training_score
}

if st.button("Predict"):
    model = joblib.load("rfc_pipeline_model.pkl")
    X_input = pd.DataFrame([inputs])
    prediction = model.predict(X_input)

    if prediction[0] == 1:
        st.success("🎉 Employee WILL be promoted")
    else:
        st.error("❌ Employee will NOT be promoted")

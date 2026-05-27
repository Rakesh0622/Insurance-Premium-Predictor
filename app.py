import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# load model
model = pickle.load(open('model_gb.pkl', 'rb'))

# scaler
scaler = MinMaxScaler()

# title
st.title("Insurance Premium Predictor")

# inputs
age = st.number_input(
    'Age',
    min_value=1,
    max_value=100,
    value=25
)

gender = st.selectbox(
    'Gender',
    ('male', 'female')
)

salary = st.number_input(
    'Annual Salary (₹)',
    min_value=10000,
    max_value=10000000,
    value=500000
)

bmi = st.number_input(
    'BMI',
    min_value=10.0,
    max_value=100.0,
    value=30.0
)

smoker = st.selectbox(
    'Smoker',
    ('yes', 'no')
)

children = st.number_input(
    'Number of Children',
    min_value=0,
    max_value=10,
    value=2
)

region = st.selectbox(
    'Region',
    ('southwest', 'southeast', 'northwest', 'northeast')
)

interest_rate = st.number_input(
    'Interest Rate (%)',
    min_value=1.0,
    max_value=20.0,
    value=8.0
)

years = st.number_input(
    'Policy Duration (Years)',
    min_value=1,
    max_value=40,
    value=10
)

# encoding
Smoker = 1 if smoker == 'yes' else 0

sex_female = 1 if gender == 'female' else 0
sex_male = 1 if gender == 'male' else 0

region_dict = {
    'southwest': 0,
    'northwest': 1,
    'northeast': 2,
    'southeast': 3
}

Region = region_dict[region]

# dataframe
input_features = pd.DataFrame({
    'age': [age],
    'salary': [salary],
    'bmi': [bmi],
    'children': [children],
    'Smoker': [Smoker],
    'sex_female': [sex_female],
    'sex_male': [sex_male],
    'Region': [Region]
})

# scaling
input_features[['age', 'salary', 'bmi']] = scaler.fit_transform(
    input_features[['age', 'salary', 'bmi']]
)

# prediction
# prediction
if st.button('Predict'):

    predictions = model.predict(input_features)

    # yearly premium
    premium = round(np.exp(predictions[0]), 2)

    # total paid by user
    total_paid = premium * years

    # final return amount
    final_return = total_paid * (
        (1 + (interest_rate / 100)) ** years
    )

    final_return = round(final_return, 2)

    # outputs
    st.success(f"Predicted Insurance Premium: ₹{premium}")

    st.success(f"Final Return Amount: ₹{final_return}")
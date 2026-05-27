import pandas as pd
import numpy as np
import pickle
import random

from faker import Faker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

# ----------------------------
# CREATE DATASET
# ----------------------------

fake = Faker()

rows = []

regions = ['southwest', 'southeast', 'northwest', 'northeast']

for i in range(500):

    age = random.randint(18, 65)

    sex = random.choice(['male', 'female'])

    bmi = round(random.uniform(18, 40), 1)

    children = random.randint(0, 5)

    smoker = random.choice(['yes', 'no'])

    region = random.choice(regions)

    salary = random.randint(200000, 1500000)

    charges = (
        age * 200 +
        bmi * 350 +
        children * 500 +
        (15000 if smoker == 'yes' else 0) +
        (salary * 0.02) +
        random.randint(-3000, 3000)
    )

    rows.append([
        age,
        sex,
        bmi,
        children,
        smoker,
        region,
        salary,
        round(charges, 2)
    ])

# dataframe
df = pd.DataFrame(rows, columns=[
    'age',
    'sex',
    'bmi',
    'children',
    'smoker',
    'region',
    'salary',
    'charges'
])

# save csv
df.to_csv('insurance.csv', index=False)

print("CSV File Created Successfully")

# ----------------------------
# ENCODING
# ----------------------------

df['Smoker'] = df['smoker'].map({
    'yes': 1,
    'no': 0
})

df['sex_female'] = df['sex'].map({
    'female': 1,
    'male': 0
})

df['sex_male'] = df['sex'].map({
    'male': 1,
    'female': 0
})

region_dict = {
    'southwest': 0,
    'northwest': 1,
    'northeast': 2,
    'southeast': 3
}

df['Region'] = df['region'].map(region_dict)

# ----------------------------
# FEATURES
# ----------------------------

X = df[[
    'age',
    'salary',
    'bmi',
    'children',
    'Smoker',
    'sex_female',
    'sex_male',
    'Region'
]]

y = np.log(df['charges'])

# ----------------------------
# SCALING
# ----------------------------

scaler = MinMaxScaler()

X[['age', 'salary', 'bmi']] = scaler.fit_transform(
    X[['age', 'salary', 'bmi']]
)

# ----------------------------
# SPLIT
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# MODEL
# ----------------------------

model = GradientBoostingRegressor()

model.fit(X_train, y_train)

# ----------------------------
# PREDICTION
# ----------------------------

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("Model Accuracy:", score)

# ----------------------------
# SAVE MODEL
# ----------------------------

pickle.dump(model, open('model_gb.pkl', 'wb'))

print("Model Saved Successfully")
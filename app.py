import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load your data
# Replace this with your actual dataset loading step
df_pub = pd.read_csv("df_pub.csv")  # Replace with the actual file name

# Preprocess the data
income_midpoints = [15000, 39000, 61500, 92500, 150000, 24000, 52500]
df_pub['income_midpoints'] = [income_midpoints] * len(df_pub)
df_melted = df_pub.melt(id_vars=['INSTNM'], 
                        value_vars=['NPT41_PUB', 'NPT42_PUB', 'NPT43_PUB', 'NPT44_PUB', 'NPT45_PUB','NPT4_048_PUB','NPT4_3075_PUB'], 
                        var_name='IncomeRange', 
                        value_name='NetPrice')

df_melted['income_midpoints'] = df_melted['IncomeRange'].map({
    'NPT41_PUB': 15000, 
    'NPT42_PUB': 39000, 
    'NPT43_PUB': 61500, 
    'NPT44_PUB': 92500, 
    'NPT45_PUB': 150000,
    'NPT4_048_PUB': 24000,
    'NPT4_3075_PUB': 52500
})

# Train models for each institution
institution_models = {}

for institution in df_melted['INSTNM'].unique():
    inst_data = df_melted[df_melted['INSTNM'] == institution]
    X = inst_data[['income_midpoints']]
    y = inst_data['NetPrice']
    
    model = LinearRegression()
    model.fit(X, y)
    institution_models[institution] = model

# Prediction function
def predict_net_price(inst_name, income):
    if inst_name not in institution_models:
        return f"Institution {inst_name} not found"
    
    if income > 200000:
        cost = df_pub.loc[df_pub['INSTNM'] == inst_name, 'COSTT4_A']
        if cost.empty:
            return f"Cost information not available for {inst_name}"
        return cost.values[0]
    
    model = institution_models[inst_name]
    return model.predict([[income]])[0]

# Streamlit app
st.title("Net Price Predictor")

# Institution selection
institution = st.selectbox("Select an Institution", df_pub['INSTNM'].unique())

# Income input
income = st.number_input("Enter your Income", min_value=0, step=1000)

# Predict button
if st.button("Predict Net Price"):
    if institution and income:
        result = predict_net_price(institution, income)
        st.write(f"Predicted Net Price for {institution} is :  $ {result:,.2f}" if isinstance(result, (int, float)) else result)
    else:
        st.write("Please select an institution and enter an income.")

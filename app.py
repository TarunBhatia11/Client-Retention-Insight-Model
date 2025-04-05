import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the encoders
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Page layout
st.set_page_config(page_title='Client Retention Insights', layout='wide')
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Client Retention Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Create two columns: input on left, output on right
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("### 📥 Enter Client Details")

    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.slider('🎂 Age', 18, 95)
    credit_score = st.number_input('💳 Credit Score', 0, 1000)
    balance = st.number_input('💰 Balance', 0.0, 100000.0)
    estimated_salary = st.number_input('🧾 Estimated Salary')
    tenure = st.slider('📆 Tenure (years)', 0, 10)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4)
    has_credit_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('📶 Is Active Member?', [0, 1])

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Display result in the right column
with right_col:
    st.markdown("### 🔍 Prediction Result")
    if prediction_prob > 0.5:
        st.success(f"⚠️ The customer is **likely to churn**.\n\n🧮 Probability: **{prediction_prob:.2%}**")
    else:
        st.info(f"✅ The customer is **likely to stay**.\n\n🧮 Probability: **{(1 - prediction_prob):.2%}**")

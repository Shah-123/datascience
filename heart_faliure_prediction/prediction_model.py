import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('data/heart_failure_clinical_records.csv')

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

st.title('Heart Failure Prediction App')

st.sidebar.header('Patient Data')
def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 60)
    anaemia = st.sidebar.selectbox('Anaemia', (0, 1))
    creatinine_phosphokinase = st.sidebar.slider('Creatinine Phosphokinase', 0, 1000, 200)
    diabetes = st.sidebar.selectbox('Diabetes', (0, 1))
    ejection_fraction = st.sidebar.slider('Ejection Fraction', 10, 80, 30)
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', (0, 1))
    platelets = st.sidebar.slider('Platelets', 100000, 500000, 250000)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.0, 10.0, 1.5)
    serum_sodium = st.sidebar.slider('Serum Sodium', 110, 150, 135)
    sex = st.sidebar.selectbox('Sex', (0, 1))
    smoking = st.sidebar.selectbox('Smoking', (0, 1))
    time = st.sidebar.slider('Time', 0, 300, 100)
    
    data = {
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    features = pd.DataFrame(data, index=[0])
    return features



input_df = user_input_features()


data_combined = pd.concat([input_df, X], axis=0)


data_combined_scaled = scaler.transform(data_combined)


input_data_scaled = data_combined_scaled[:1]

prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)

st.subheader('User Input Features')
st.write(input_df)

st.subheader('Prediction')
st.write('Death Event Prediction: ', 'Yes' if prediction[0] == 1 else 'No')
st.write('Prediction Probability: {:.2f}'.format(prediction_prob[0][1]))

st.subheader('Dataset')
st.write(data)

st.subheader('Data Visualization')

if st.checkbox('Show correlation heatmap'):
    st.write('Correlation Heatmap')
    corr_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    st.pyplot(plt)

# Age distribution
if st.checkbox('Show age distribution'):
    st.write('Age Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], kde=True)
    plt.title('Distribution of Age')
    st.pyplot(plt)

# Pairplot
if st.checkbox('Show pairplot'):
    st.write('Pairplot of Data')
    sns.pairplot(data, hue='DEATH_EVENT')
    st.pyplot(plt)

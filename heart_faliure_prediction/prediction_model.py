import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

data = load_data('data/heart_failure_clinical_records.csv')

X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)



st.title('Heart Failure Prediction App')
st.sidebar.header('Patient Data')


def user_input_features():
    st.sidebar.markdown("### Age of the patient (years)")
    age = st.sidebar.slider('Age', 18, 100, 60)
    
    st.sidebar.markdown("### Decrease of red blood cells or hemoglobin")
    anaemia = st.sidebar.selectbox('Anaemia', ('No', 'Yes'))
    
    st.sidebar.markdown("### Level of the CPK enzyme in the blood (mcg/L)")
    creatinine_phosphokinase = st.sidebar.slider('Creatinine Phosphokinase', 0, 7861, 200)
    
    st.sidebar.markdown("### If the patient has diabetes")
    diabetes = st.sidebar.selectbox('Diabetes', ('No', 'Yes'))
    
    st.sidebar.markdown("### Percentage of blood leaving the heart at each contraction (percentage)")
    ejection_fraction = st.sidebar.slider('Ejection Fraction', 14, 80, 30)
    
    st.sidebar.markdown("### If the patient has hypertension")
    high_blood_pressure = st.sidebar.selectbox('High Blood Pressure', ('No', 'Yes'))
    
    st.sidebar.markdown("### Platelets in the blood (k/µL)")
    platelets = st.sidebar.slider('Platelets', 25, 850, 250)
    
    st.sidebar.markdown("### Level of serum creatinine in the blood (mg/dL)")
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.0, 10.0, 1.5)
    
    st.sidebar.markdown("### Level of serum sodium in the blood (mEq/L)")
    serum_sodium = st.sidebar.slider('Serum Sodium', 110, 150, 135)
    
    st.sidebar.markdown("### Sex of the patient (woman: 0, man: 1)")
    sex = st.sidebar.selectbox('Sex', ('Female', 'Male'))
    
    st.sidebar.markdown("### If the patient smokes or not")
    smoking = st.sidebar.selectbox('Smoking', ('No', 'Yes'))
    
    st.sidebar.markdown("### Follow-up period (days)")
    time = st.sidebar.slider('Time', 0, 300, 100)
    
    data = {
        'age': age,
        'anaemia': 1 if anaemia == 'Yes' else 0,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': 1 if diabetes == 'Yes' else 0,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': 1 if high_blood_pressure == 'Yes' else 0,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': 1 if sex == 'Male' else 0,
        'smoking': 1 if smoking == 'Yes' else 0,
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
st.write('Death Event Prediction:', 'Yes' if prediction[0] == 1 else 'No')
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
    plt.clf()  

if st.checkbox('Show age distribution'):
    st.write('Age Distribution')
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age'], kde=True)
    plt.title('Distribution of Age')
    st.pyplot(plt)
    plt.clf()

if st.checkbox('Show count plots for categorical features'):
    categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    
    for feature in categorical_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=data, hue='DEATH_EVENT', palette="pastel")
        st.header(f' Death by {feature.capitalize()} ')
        plt.title(f'Count of {feature.capitalize()} by Death Event')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Count')
        plt.legend(title='Death Event', loc='upper right')
        st.pyplot(plt)
        plt.clf()

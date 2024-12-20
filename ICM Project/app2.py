import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

@st.cache_resource
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Function to drop unnecessary columns
def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')

# Create additional features based on existing columns
def create_features(df):
    df['NON_VAC_011'] = df['RECALL_011_CHK'] - df['RECALL_011_VAC']
    df['NON_VAC_1259'] = df['RECALL_1259_CHK'] - df['RECALL_1259_VAC']
    df['NON_VEC_GUEST'] = df['GUEST_CHK'] - df['GUEST_VAC']
    df['FM_NON_VEC_059'] = df['FM_059_CHK'] - df['FM_059_VAC']
    return df

# Function to identify and retain highly correlated columns
def Correlated_Columms(df):
    target_variable = 'NA'
    correlation_matrix = df.corr()
    target_correlations = correlation_matrix[target_variable].abs().sort_values(ascending=False)


    top_15_features = target_correlations.drop(target_variable).head(14)
    top_features_df = df[['MONITORID', 'NA'] + top_15_features.index.tolist()]

    return top_features_df
def group_data(df):
    Grouped_Monitor_DATA = df.groupby(['MONITORID', 'CAMP_ID']).sum().reset_index()
    return Grouped_Monitor_DATA

def preprocess_data(df):
    columns_to_drop = ['CAMP_DAY', 'DAY_WORK', 'CLUSTER_NUMBER', 'TOTAL_HH', 'HRMP',
    'TEAM_NUMBER', 'CORRECT_DOOR_MARK', 'CLUSTER_DATE', 'ICM_ID',
    'UCID', 'NA_VAC']
    df = drop_columns(df, columns_to_drop)
    df = create_features(df)
    df = group_data(df)
    df = Correlated_Columms(df)
    return df

@st.cache_resource
def train_model(df):
    df = preprocess_data(df)
    X = df.drop(['MONITORID', 'CAMP_ID', 'NA'], axis=1)
    y = df['NA']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler




# Streamlit app
st.title("Child NA Prediction App")
st.write("Upload your data, preprocess it, and make predictions for MONITORID and CAMP_ID.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())
    try:
        preprocessed_df = preprocess_data(df)
        st.write("Preprocessed Data:")
        st.dataframe(preprocessed_df.head())


        model, scaler = train_model(df)
    
    
        # MONITOR_ID = st.sidebar.number_input("Enter Monitor ID:", value=0, step=1)
        # CAMP_ID = st.number_input("Enter Camp ID:", value=0, step=1)

        # if st.button("Predict"):
        #     prediction_features = preprocessed_df[(preprocessed_df['MONITORID'] == MONITOR_ID) & (preprocessed_df['CAMP_ID'] == CAMP_ID)]

        # if not prediction_features.empty:
        # # Drop unnecessary columns for prediction
        #     prediction_features = prediction_features.drop(columns=['MONITORID', 'CAMP_ID', 'NA'], errors='ignore')
        #     prediction_features_scaled = scaler.transform(prediction_features)

        # # Make predictions
        #     predictions = model.predict(prediction_features_scaled)
        #     predicted_na = int(predictions[0])  # Extract single prediction
        #     st.write(f"Predicted NA for MONITORID {MONITOR_ID} and CAMP_ID {CAMP_ID} is: {predicted_na}")
        # else:
        #     st.error("No matching data found for the entered MONITORID and CAMP_ID. Please ensure the values are correct.")

            
        if 'MONITORID' in preprocessed_df.columns and 'CAMP_ID' in preprocessed_df.columns:
            
            prediction_features = preprocessed_df.drop(columns=['MONITORID', 'CAMP_ID', 'NA'], errors='ignore')
            prediction_features_scaled = scaler.transform(prediction_features)

            predictions = model.predict(prediction_features_scaled)

            preprocessed_df['Predicted_NA'] = predictions
            preprocessed_df['Predicted_NA'] =preprocessed_df['Predicted_NA'].astype(int)
            
            
            preprocessed_df['Miss_match_predcition'] = preprocessed_df['NA'] - preprocessed_df['Predicted_NA']

            output_df = preprocessed_df[['MONITORID', 'CAMP_ID', 'Predicted_NA', 'NA', 'Miss_match_predcition']]

            positive_mismatches = output_df[output_df['Miss_match_predcition'] > 0]
            
            
            if not positive_mismatches.empty:
                
                max_positive_diff = positive_mismatches['Miss_match_predcition'].max()
                
                max_positive_row = positive_mismatches[positive_mismatches['Miss_match_predcition'] == max_positive_diff]
                st.write("Row with Highest Positive Difference:")
                
                
                st.dataframe(max_positive_row)
            else:
                st.write("No positive mismatches found.")

            # Filter negative mismatches
            negative_mismatches = output_df[output_df['Miss_match_predcition'] < 0]
            if not negative_mismatches.empty:
                min_negative_diff = negative_mismatches['Miss_match_predcition'].min()
                min_negative_row = negative_mismatches[negative_mismatches['Miss_match_predcition'] == min_negative_diff]
                st.write("Row with Highest Negative Difference:")
                st.dataframe(min_negative_row)
            else:
                st.write("No negative mismatches found.")

            correct_predictions = output_df[output_df['Miss_match_predcition'] == 0]
            st.write(f"{len(correct_predictions)} predictions are exact matches:")
            st.dataframe(correct_predictions)

            
            
            NEGATIVE_PREDCITION = output_df[output_df['Predicted_NA']<0]
            
            st.write('Model predctions That are Negative ',NEGATIVE_PREDCITION)
            overall_prediction_sum =preprocessed_df['Predicted_NA'].sum()
            overall_actual_sum = preprocessed_df['NA'].sum()

            st.write("Predicted_NA:")
            st.dataframe(output_df)
            st.write ("Total Actual NA reported ",overall_actual_sum)
            st.write('Total Predicted NA',overall_prediction_sum)
            
            percentage =( overall_prediction_sum/overall_actual_sum)*100
            percentage =100 - percentage
            st.write ("percentange Loss",percentage.round(2))
            
            
            csv = output_df.to_csv(index=False)
            st.download_button(label="Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("Preprocessed data does not contain 'MONITORID' or 'CAMP_ID'. Check the uploaded file.")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
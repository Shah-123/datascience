import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# Load the sample data to fit the model initially
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df 

def train_model(df):
    # Separate features and target variable
    X = df.drop(['runs_off_bat_x'], axis=True)
    y = df['runs_off_bat_x']

    # Define categorical and numeric columns
    categorical_columns = ['venue', 'batting_team', 'bowling_team']
    numeric_columns = ['balls_left', 'wicket_left', 'Current_Score', 'Crr', 'last_five']

    # Preprocessing for numerical data (scaling) and categorical data (one-hot encoding)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(sparse_output=False), categorical_columns)
        ],
        remainder='drop'
    )

    # Create a pipeline that combines preprocessing and model training
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())
    ])

    # Fit the model
    model.fit(X, y)
    return model

def main():
    # Set the page configuration
    st.set_page_config(page_title="Cricket Runs Prediction", page_icon="🏏", layout="centered")
    
    # App title and description
    st.title("🏏 Cricket Runs Prediction App")
    st.markdown("""
    Welcome to the Cricket Runs Prediction App! This tool uses machine learning to predict the number of runs 
    based on various match conditions. Provide the match details below to get started.
    """)
    
    # Load and train the model
    data = load_data('Cricket_APP/final_data.csv')
    model = train_model(data)

    st.write("### Input Features for Prediction")

    # Organize input features into columns for better layout
    col1, col2, col3 = st.columns(3)
    with col1:
        venue = st.selectbox("Venue", data['venue'].unique())
    with col2:
        batting_team = st.selectbox("Batting Team", data['batting_team'].unique())
    with col3:
        bowling_team = st.selectbox("Bowling Team", data['bowling_team'].unique())

    col4, col5, col6 = st.columns(3)
    with col4:
        balls_left = st.number_input("Balls Left", min_value=0, max_value=300, step=1)
    with col5:
        wicket_left = st.number_input("Wickets Left", min_value=0, max_value=10, step=1)
    with col6:
        Current_Score = st.number_input("Current Score", min_value=0, max_value=500, step=1)

    col7, col8 = st.columns(2)
    with col7:
        Crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, step=0.1)
    with col8:
        last_five = st.number_input("Runs in Last Five Overs", min_value=0, max_value=100, step=1)

    # Prepare the user input for prediction
    user_input = pd.DataFrame({
        'venue': [venue],
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'balls_left': [balls_left],
        'wicket_left': [wicket_left],
        'Current_Score': [Current_Score],
        'Crr': [Crr],
        'last_five': [last_five]
    })

    # Predict runs based on user input
    if st.button("Predict Runs"):
        prediction = model.predict(user_input)
        st.success(f"Predicted Runs: **{int(prediction[0])}**")

    # Footer
    st.markdown("""
    ---
    **Note:** This prediction is based on historical data and machine learning algorithms. Use it as a guide and enjoy the game!
    """)

if __name__ == "__main__":
    main()

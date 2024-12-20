import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Child NA Prediction Dashboard",
    page_icon="👶",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data(filepath):
    return pd.read_csv(filepath)



def drop_columns(df, columns_to_drop):
    
    return df.drop(columns=columns_to_drop, errors='ignore')


def group_data(df):
    return df.groupby(['MONITORID', 'CAMP_ID']).sum().reset_index()




#New features will be created based on existing columns
def create_features(df):
    df['NON_VAC_011'] = df['RECALL_011_CHK'] - df['RECALL_011_VAC']
    df['NON_VAC_1259'] = df['RECALL_1259_CHK'] - df['RECALL_1259_VAC']
    df['NON_VEC_GUEST'] = df['GUEST_CHK'] - df['GUEST_VAC']
    df['FM_NON_VEC_059'] = df['FM_059_CHK'] - df['FM_059_VAC']
    return df
#top 15 features will be selected based on correlation with the target variable
def Correlated_Columms(df):
    target_variable = 'NA'
    correlation_matrix = df.corr()
    target_correlations = correlation_matrix[target_variable].abs().sort_values(ascending=False)
    top_features = target_correlations.drop(target_variable).head(14)
    return df[['MONITORID', 'NA'] + top_features.index.tolist()]


def preprocess_data(df):
    columns_to_drop = [
        'CAMP_DAY', 'DAY_WORK', 'CLUSTER_NUMBER', 'TOTAL_HH', 'HRMP',
        'TEAM_NUMBER', 'CORRECT_DOOR_MARK', 'CLUSTER_DATE', 'ICM_ID',
        'UCID', 'NA_VAC'
    ]
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
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    #calculate the evalutaion matric
    y_pred = model.predict(scaler.transform(X_test))
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, scaler, mae, mse, r2

def main():
    st.title("🏥 Child NA Prediction Dashboard")
    st.markdown("---")

    with st.sidebar:
        st.header("📊 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            st.markdown("---")
            st.header("🎯 Make Predictions")
            MONITOR_ID = st.number_input("Monitor ID", value=0, step=1)
            CAMP_ID = st.number_input("Camp ID", value=0, step=1)
            predict_button = st.button("🔍 Predict", use_container_width=True)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            preprocessed_df = preprocess_data(df)
            model, scaler ,mae,mse,r2= train_model(df)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Raw Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.subheader(" Preprocessed Data Preview")
                st.dataframe(preprocessed_df.head(), use_container_width=True)

            if predict_button:
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                prediction_features = preprocessed_df[
                    (preprocessed_df['MONITORID'] == MONITOR_ID) & 
                    (preprocessed_df['CAMP_ID'] == CAMP_ID)
                ]

                if not prediction_features.empty:
                    prediction_features = prediction_features.drop(
                        columns=['MONITORID', 'CAMP_ID', 'NA'], 
                        errors='ignore'
                    )
                    prediction_features_scaled = scaler.transform(prediction_features)
                    predicted_na = int(model.predict(prediction_features_scaled)[0])
                    
                    col1, col2, col3,col4 = st.columns(4)
                    with col1:
                        st.metric("Monitor ID", MONITOR_ID)
                    with col2:
                        st.metric("Camp ID", CAMP_ID)
                    with col3:
                        st.metric("Predicted NA", predicted_na)
                    with col4:
                        st.metric(
                            "Actual NA", 
                            preprocessed_df[(preprocessed_df['CAMP_ID'] == CAMP_ID) & (preprocessed_df['MONITORID'] == MONITOR_ID)]["NA"].values[0] 
)
                else:
                    st.error("No matching data found for the given Monitor ID and Camp ID")

            st.markdown("---")
            st.subheader("📊 Analysis Dashboard")
            
            prediction_features = preprocessed_df.drop(
                columns=['MONITORID', 'CAMP_ID', 'NA'], 
                errors='ignore'
            )
            prediction_features_scaled = scaler.transform(prediction_features)
            predictions = model.predict(prediction_features_scaled)
            
            preprocessed_df['Predicted_NA'] = predictions.astype(int)
            preprocessed_df['Miss_match_prediction'] = preprocessed_df['NA'] - preprocessed_df['Predicted_NA']
            
            col1, col2,  = st.columns(2)
            with col1:
                st.metric(
                    "Total Actual NA (Not Available Children)", 
                    f"{preprocessed_df['NA'].sum():,}"
                )
            with col2:
                st.metric(
                    "Total Predicted NA(Not Available Children)", 
                    f"{preprocessed_df['Predicted_NA'].sum():,}"
                )

            
            # Create metrics section
            st.subheader("📊 Model Performance Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Mean Absolute Error (MAE)",
                    value=f"{mae:.2f}",
                    help="Average absolute difference between predicted and actual values"
                )

            with col2:
                st.metric(
                    label=" Mean Square Error (MSE)",
                    value=f"{mse:.2f}",
                    help="mean square error between predictions and actual values"
                )

            with col3:
                st.metric(
                    label="R² Score",
                    value=f"{r2:.2f}",
                    help="Proportion of variance in the dependent variable predictable from independent variable(s)"
                )


            
            # Download section
            st.markdown("---")
            output_df = preprocessed_df[['MONITORID', 'CAMP_ID', 'Predicted_NA', 'NA']]
            st.subheader("Predicted NA VS Actual NA ")
            st.write(output_df)
            
            csv = output_df.to_csv(index=False)
            # st.subheader("Predicted NA VS Actual NA ")
            # st.write(output_df)
            st.markdown("---")
            st.subheader("📥 Download Results")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        # Welcome message when no file is uploaded
        st.info("👋 Welcome! Please upload a CSV file to begin the analysis.")
        st.markdown("""
        ### How to use this dashboard:
        1. Upload your CSV file using the sidebar
        2. View the raw and preprocessed data
        3. Enter Monitor ID and Camp ID for specific predictions
        4. Analyze results through interactive visualizations
        5. Download the prediction results
        """)

if __name__ == "__main__":
    main()
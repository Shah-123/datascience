import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data():
    df = pd.read_csv('../data/clean House price.csv')
    df['property_type'] = df['property_type'].str.replace('Room', 'House')
    return df

def visualize(df):
    st.header("Data Visualizations")
    
    sns.set(style="whitegrid")

    # Distribution of Prices
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df['price_in_lac'], color='blue', fill=True, ax=ax)
    ax.set_title('Distribution of Prices', fontsize=15, fontweight='bold')
    ax.set_xlabel('Price in Lac', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    sns.despine()
    st.pyplot(fig)

    # Breakdown of property types by city
    fig1 = px.histogram(df, x='property_type', color='property_type', title='Distribution of Property Types')
    st.plotly_chart(fig1)

    # Number of Bedrooms vs. Price
    fig2 = px.histogram(df, x='price_in_lac', nbins=50, title='Distribution of Prices')
    st.plotly_chart(fig2)

    # Pie Chart for Price Distribution by City
    st.subheader('Price Distribution by City')
    city_price_df = df.groupby('city')['price_in_lac'].value_counts().reset_index()
    fig3 = px.pie(city_price_df, names='city', values='price_in_lac', title='Price Distribution by City')
    st.plotly_chart(fig3)

def train_and_evaluate(df):
    categorical_features = ['property_type', 'purpose', 'location', 'city']
    numerical_features = ['bedrooms', 'Area_in_Marla', 'baths']
    df['price_in_lac'] = np.log1p(df['price_in_lac'])

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Preparing the data
    X = df.drop(['price_in_lac'], axis=1)
    y = df['price_in_lac']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'DecisionTree Regression': DecisionTreeRegressor(),
        'CatBoost Regressor': CatBoostRegressor(),
        'XGBoost': XGBRegressor()
    }

    metrics = []

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Calculating metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics.append({
            'Model': name,
            'Mean Absolute Error': mae,
            'Mean Squared Error': mse,
            'R2 Score': r2
        })

        st.subheader(f'Actual vs Predicted Prices ({name})')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.scatter(y_test, y_pred, alpha=0.3, color='blue')
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        ax.set_xlabel('Actual Prices', fontsize=12)
        ax.set_ylabel('Predicted Prices', fontsize=12)
        ax.set_title(f'Actual vs Predicted Prices ({name})', fontsize=15, fontweight='bold')
        st.pyplot(fig)
    st.subheader('XGBoost Regressor Actual price  vs predicted Price ')
    df1 = pd.DataFrame({'Actual Price in lacs': np.expm1(y_test), 'Predicted Price in lacs': np.expm1(y_pred)})
    st.write(df1.sample(10))

    metrics_df = pd.DataFrame(metrics)
    st.header('Perfromance of different Algorithms')
    st.write(metrics_df)

    return models['XGBoost'], preprocessor  

def predict_price(model, preprocessor, df):
    st.sidebar.header('Predict Your House Price')
    property_type = st.sidebar.selectbox('Property Type', df['property_type'].unique())
    purpose = st.sidebar.selectbox('Purpose', df['purpose'].unique())
    location = st.sidebar.selectbox('Location', df['location'].unique())
    city = st.sidebar.selectbox('City', df['city'].unique())
    bedrooms = st.sidebar.slider('Bedrooms', 0, 10, 3)
    Area_in_Marla = st.sidebar.slider('Area in Marla', 0, 100, 50)
    baths = st.sidebar.slider('Bathrooms', 0, 10, 3)

    input_data = pd.DataFrame({
        'property_type': [property_type],
        'purpose': [purpose],
        'location': [location],
        'city': [city],
        'bedrooms': [bedrooms],
        'Area_in_Marla': [Area_in_Marla],
        'baths': [baths]
    })

    if st.sidebar.button('Predict Price'):
        input_data_transformed = preprocessor.transform(input_data)
        prediction = model.predict(input_data_transformed)
        prediction2 = np.expm1(prediction)
        st.sidebar.write(f'**Predicted House Price in Lac:** {prediction2[0]:.2f}')

def main():
    st.title('House Price Visualization and Prediction')
    df = load_data()

    tab1, tab2 = st.tabs(["Visualizations", "Model Performance"])
    
    with tab1:
        visualize(df)
    
    with tab2:
        model, preprocessor = train_and_evaluate(df)

    predict_price(model, preprocessor, df)

if __name__ == "__main__":
    main()
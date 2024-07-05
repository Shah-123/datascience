import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_csv('../data/exams.csv')
    df['ethnicity'] = df['race/ethnicity']
    df = df.drop('race/ethnicity', axis=1)
    df['lunch'] = df['lunch'].str.replace('^free/', '', regex=True)
    return df

# Function to plot histograms
def plot_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.histplot(data=df, x='math score', kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Math Scores')
    axes[0].set_xlabel('Math Score')
    sns.histplot(data=df, x='reading score', kde=True, ax=axes[1])
    axes[1].set_title('Distribution of Reading Scores')
    axes[1].set_xlabel('Reading Score')
    sns.histplot(data=df, x='writing score', kde=True, ax=axes[2])
    axes[2].set_title('Distribution of Writing Scores')
    axes[2].set_xlabel('Writing Score')
    st.pyplot(fig)

# Function to plot correlation matrix
def plot_correlation_matrix(df):
    correlation_matrix = df[['math score', 'reading score', 'writing score']].corr()
    st.write(correlation_matrix)

# Function to plot boxplot
def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='gender', y='math score', hue='test preparation course', data=df, ax=ax)
    ax.set_title('Math Scores by Gender and Test Preparation')
    st.pyplot(fig)

# Function to plot bar plot
def plot_barplot(df):
    avg_scores_by_education = df.groupby('parental level of education')[['math score', 'reading score', 'writing score']].mean().round(2)
    avg_scores_by_education = avg_scores_by_education.sort_values('math score', ascending=False)
    st.write(avg_scores_by_education)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=avg_scores_by_education.reset_index(), x='parental level of education', y='math score', ax=ax)
    ax.set_title('Average Test Scores by Parental Education Level')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# Function to train model and evaluate
def train_and_evaluate(df):
    categorical_features = ['gender', 'parental level of education', 'lunch', 'test preparation course', 'ethnicity']
    numerical_features = ['reading score', 'writing score']
    X = df.drop('math score', axis=1)
    y = df['math score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write(f'Mean Absolute Error: {mae:.2f}')
    st.write(f'R² Score: {r2:.2f}')
    st.write(f'Mean Absolute Percentage Error (MAPE): {mape:.2%}')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_percentage_error')
    cv_mape = -np.mean(cv_scores)
    st.write(f'Cross-validated MAPE: {cv_mape:.2%}')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
    ax.set_xlabel('Actual Scores')
    ax.set_ylabel('Predicted Scores')
    ax.set_title('Actual vs Predicted Math Scores')
    st.pyplot(fig)
    return model

# Function to predict math score
def predict_score(model, df):
    st.sidebar.header('Predict Your Math Score')
    gender = st.sidebar.selectbox('Gender', df['gender'].unique())
    parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', df['parental level of education'].unique())
    lunch = st.sidebar.selectbox('Lunch', df['lunch'].unique())
    test_preparation_course = st.sidebar.selectbox('Test Preparation Course', df['test preparation course'].unique())
    ethnicity = st.sidebar.selectbox('Ethnicity', df['ethnicity'].unique())
    reading_score = st.sidebar.slider('Reading Score', 0, 100, 50)
    writing_score = st.sidebar.slider('Writing Score', 0, 100, 50)
    input_data = pd.DataFrame({
        'gender': [gender],
        'parental level of education': [parental_level_of_education],
        'lunch': [lunch],
        'test preparation course': [test_preparation_course],
        'ethnicity': [ethnicity],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })
    if st.sidebar.button('Predict Math Score'):
        prediction = model.predict(input_data)
        st.sidebar.write(f'Predicted Math Score: {prediction[0]:.2f}')

# Main function
def main():
    st.title('Exam Scores Analysis')

    # Dataset description
    st.header('About Dataset')
    st.markdown('''
    **Description:** This dataset contains information on the performance of high school students in mathematics, including their grades and demographic information. The data was collected from three high schools in the United States.

    **Columns:**
    - **Gender:** The gender of the student (male/female)
    - **ethnicity:** The student's racial or ethnic background (Asian, African-American, Hispanic, etc.)
    - **Parental level of education:** The highest level of education attained by the student's parent(s) or guardian(s)
    - **Lunch:** Whether the student receives free or reduced-price lunch (yes/no)
    - **Test preparation course:** Whether the student completed a test preparation course (yes/no)
    - **Math score:** The student's score on a standardized mathematics test
    - **Reading score:** The student's score on a standardized reading test
    - **Writing score:** The student's score on a standardized writing test
    ''')

    df = load_data()
    model = train_and_evaluate(df)
    predict_score(model, df)
    
    tab1, tab2 = st.tabs(["Visualization", "Model Performance"])
    
    with tab1:
        st.header('Dataset')
        st.write(df.head())
        st.header('Distribution of Scores')
        plot_histograms(df)
        st.header('Correlation Matrix')
        plot_correlation_matrix(df)
        st.header('Math Scores by Gender and Test Preparation')
        plot_boxplot(df)
        st.header('Average Scores by Parental Education Level')
        plot_barplot(df)
    
    with tab2:
        st.header('Model Training and Evaluation')
        train_and_evaluate(df)

if __name__ == "__main__":
    main()

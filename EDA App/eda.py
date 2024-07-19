import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
import io
import seaborn as sns 
from sklearn.preprocessing import StandardScaler

def display_data(data):
    st.subheader("Dataset Overview")
    with st.expander("View Dataset"):
        st.dataframe(data.head(100))
    
    st.subheader("Data Summary")
    with st.expander("View Data Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Basic Information:")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
        with col2:
            st.write("Descriptive Statistics:")
            st.write(data.describe())
    
    st.subheader("Column Information")
    with st.expander("View Column Details"):
        st.write(data.dtypes)
        st.write("Missing Values:")
        st.write(data.isnull().sum())

def preprocess_data(data):
    st.sidebar.header("Data Preprocessing")
    
    if st.sidebar.checkbox("Handle Missing Values"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns
        
        for col in numeric_columns:
            fill_method = st.sidebar.selectbox(f"Fill method for {col}", ["Mean", "Median", "Mode", "Interpolation", "None"])
            if fill_method == "Mean":
                data[col].fillna(data[col].mean(), inplace=True)
            elif fill_method == "Median":
                data[col].fillna(data[col].median(), inplace=True)
            elif fill_method == "Mode":
                data[col].fillna(data[col].mode()[0], inplace=True)
            elif fill_method == "Interpolation":
                data[col].interpolate(method='linear', inplace=True)
        
        for col in categorical_columns:
            fill_method = st.sidebar.selectbox(f"Fill method for {col}", ["Mode", "None"])
            if fill_method == "Mode":
                data[col].fillna(data[col].mode()[0], inplace=True)
        
        st.sidebar.success("Missing values handled.")
    
    if st.sidebar.checkbox("Apply Feature Scaling"):
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        st.sidebar.success("Feature scaling applied.")
    
    if st.sidebar.checkbox("Remove Outliers"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        z_scores = np.abs(stats.zscore(data[numeric_columns]))
        data = data[(z_scores < 3).all(axis=1)]
        st.sidebar.success("Outliers removed.")
    
    return data

def explore_data(data):
    st.header("🏰 Welcome to the Kingdom of EDA")
    
    display_data(data)
    
    st.subheader("Advanced Data Visualization")
    
    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns

    plot_type = st.selectbox("Select plot type", [
        "Scatter Plot", "Line Plot", "Histogram", "Box Plot", "Violin Plot",
        "Pair Plot", "Correlation Heatmap", "Bar Plot", "Pie Chart", 
        "3D Scatter Plot", "Categorical Bar Plot", "Categorical Box Plot"
    ])
    
    if plot_type in ["Scatter Plot", "Line Plot", "Bar Plot"]:
        x_col = st.selectbox("Select X-axis column", data.columns)
        y_col = st.selectbox("Select Y-axis column", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        
        if plot_type == "Scatter Plot":
            fig = px.scatter(data, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
        elif plot_type == "Line Plot":
            fig = px.line(data, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
        elif plot_type == "Bar Plot":
            if data[x_col].nunique() > 10:
                top_10 = data[x_col].value_counts().nlargest(10).index
                data_subset = data[data[x_col].isin(top_10)]
                fig = px.bar(data_subset, x=x_col, y=y_col, color=hue_col, title=f"Top 10 categories: {x_col} vs {y_col}")
            else:
                fig = px.bar(data, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
        
        st.plotly_chart(fig)
    
    elif plot_type in ["Histogram", "Box Plot", "Violin Plot"]:
        col = st.selectbox("Select column", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        
        if plot_type == "Histogram":
            fig = px.histogram(data, x=col, color=hue_col, title=f"Histogram of {col}", )
        elif plot_type == "Box Plot":
            fig = px.box(data, y=col, color=hue_col, title=f"Box Plot of {col}")
        elif plot_type == "Violin Plot":
            fig = px.violin(data, y=col, color=hue_col, title=f"Violin Plot of {col}")
        # elif plot_type == 'Kde(kernal density plot)':
        #     fig = sns.kdeplot(data[col], shade=True, color=hue_col)
        #     plt.title(f"Kde Plot of  {col}")             
        st.plotly_chart(fig)
    
    elif plot_type == "Pair Plot":
        cols = st.multiselect("Select columns for pair plot", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        if cols:
            fig = px.scatter_matrix(data[cols], color=hue_col)
            st.plotly_chart(fig)
    
    elif plot_type == "Correlation Heatmap":
        corr = data[numeric_columns].corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect="auto")
        st.plotly_chart(fig)
    
    elif plot_type == "Pie Chart":
        col = st.selectbox("Select column", categorical_columns)
        pie_data = data[col].value_counts()
        fig = px.pie(values=pie_data.values, names=pie_data.index, title=f"Pie Chart of {col}")
        st.plotly_chart(fig)
    
    elif plot_type == "3D Scatter Plot":
        x_col = st.selectbox("Select X-axis column", numeric_columns)
        y_col = st.selectbox("Select Y-axis column", numeric_columns)
        z_col = st.selectbox("Select Z-axis column", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=hue_col)
        st.plotly_chart(fig)
    
    elif plot_type == "Categorical Bar Plot":
        x_col = st.selectbox("Select X-axis column (categorical)", categorical_columns)
        y_col = st.selectbox("Select Y-axis column (numeric)", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        
        if data[x_col].nunique() > 10:
            top_10 = data[x_col].value_counts().nlargest(10).index
            data_subset = data[data[x_col].isin(top_10)]
            fig = px.bar(data_subset, x=x_col, y=y_col, color=hue_col, title=f"Top 10 categories: {x_col} vs {y_col}")
        else:
            fig = px.bar(data, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
        
        st.plotly_chart(fig)
    
    elif plot_type == "Categorical Box Plot":
        x_col = st.selectbox("Select X-axis column (categorical)", categorical_columns)
        y_col = st.selectbox("Select Y-axis column (numeric)", numeric_columns)
        hue_col = st.selectbox("Select Hue column (optional)", [None] + list(categorical_columns))
        
        if data[x_col].nunique() > 10:
            top_10 = data[x_col].value_counts().nlargest(10).index
            data_subset = data[data[x_col].isin(top_10)]
            fig = px.box(data_subset, x=x_col, y=y_col, color=hue_col, title=f"Top 10 categories: {x_col} vs {y_col}")
        else:
            fig = px.box(data, x=x_col, y=y_col, color=hue_col, title=f"{x_col} vs {y_col}")
        
        st.plotly_chart(fig)


def main():
    st.set_page_config(page_title="EDA Web App", page_icon="📊", layout="wide")
    st.title("Exploratory Data Analysis (EDA) Web Application")

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = preprocess_data(data)
        explore_data(data)
    else:
        st.info("Please upload a CSV file to begin the analysis.")

if __name__ == "__main__":
    main()

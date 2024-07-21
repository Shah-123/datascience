import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import io
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import boxcox
import statsmodels.api as sm
# import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report

# Set page config
st.set_page_config(page_title="EDA Master", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve UI
st.markdown("""
    <style>
    .reportview-container {
        background: linear-gradient(to right, #f0f2f6, #e0e2e6);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #f0f2f6, #d0d2d6);
    }
    .Widget>label {
        color: #31333F;
        font-weight: bold;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #0068c9;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        color: #31333F;
    }
    .stSelectbox>div>div>select {
        color: #31333F;
    }
    h1 {
        color: #0068c9;
    }
    h2 {
        color: #31333F;
    }
    .stAlert > div {
        color: #31333F;
        background-color: #f0f2f6;
        border: 2px solid #0068c9;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a title and subtitle
st.title("📊 EDA Master")
st.markdown("### Unleash the power of data exploration with our advanced EDA tool")

# Function to load and display data
def load_data(data):
    st.subheader("🔎 Data Preview")
    with st.expander("View Data Sample"):
        st.dataframe(data.head(100))
    
    st.subheader("📊 Data Summary")
    with st.expander("View Summary Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(data.describe())
        with col2:
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
    
    st.subheader("ℹ️ Data Information")
    with st.expander("View Data Types and Non-Null Counts"):
        st.write(data.dtypes)
        st.write(data.isnull().sum())

# Function for data preprocessing
def Data_preprocessing(data):
    st.subheader("🛠️ Data Preprocessing")
    
    with st.expander("Handle Missing Values"):
        if data.isnull().sum().sum() == 0:
            st.success("No missing values found in the dataset!")
        else:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            categorical_columns = data.select_dtypes(exclude=[np.number]).columns
            
            for col in numeric_columns:
                method = st.selectbox(f"Choose method for {col}", ["None", "Mean", "Median", "Mode", "Interpolation"])
                if method != "None":
                    if method == "Mean":
                        data[col].fillna(data[col].mean(), inplace=True)
                    elif method == "Median":
                        data[col].fillna(data[col].median(), inplace=True)
                    elif method == "Mode":
                        data[col].fillna(data[col].mode()[0], inplace=True)
                    elif method == "Interpolation":
                        data[col].interpolate(method='linear', inplace=True)
            
            for col in categorical_columns:
                method = st.selectbox(f"Choose method for {col}", ["None", "Mode", "New Category"])
                if method != "None":
                    if method == "Mode":
                        data[col].fillna(data[col].mode()[0], inplace=True)
                    elif method == "New Category":
                        data[col].fillna("Unknown", inplace=True)

    with st.expander("Feature Scaling"):
        if st.checkbox("Apply Standard Scaling"):
            scaler = StandardScaler()
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    with st.expander("Outlier Handling"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if st.checkbox(f"Remove outliers in {col}"):
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    with st.expander("Data Transformation"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            transformation = st.selectbox(f"Transform {col}", ["None", "Log", "Square Root"])
            if transformation != "None":
                if transformation == "Log":
                    data[col] = np.log1p(data[col])
                elif transformation == "Square Root":
                    data[col] = np.sqrt(data[col])


    with st.expander("Encoding Categorical Variables"):
        method = st.selectbox("Choose encoding method", ["None", "One-Hot Encoding", "Label Encoding"])
        if method != "None":
            if method == "One-Hot Encoding":
                data = pd.get_dummies(data)
            elif method == "Label Encoding":
                le = LabelEncoder()
                for col in data.select_dtypes(include=[object]).columns:
                    data[col] = le.fit_transform(data[col])
    
    return data

def Drop_columns(data):
    st.subheader("🗑️ Drop Columns")

    with st.expander("Drop Columns"):
        col_to_drop = st.multiselect("Select columns to drop", data.columns)
        if col_to_drop:
            data = data.drop(col_to_drop, axis=1)
    
    return data

# Function for feature engineering
def feature_engineering(data):
    st.subheader("🔧 Feature Engineering")
    
    with st.expander("Create Binned Features"):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if st.checkbox(f"Bin {col}"):
                n_bins = st.slider(f"Number of bins for {col}", 2, 10, 5)
                data[f"{col}_binned"] = pd.cut(data[col], bins=n_bins)
    
    with st.expander("Extract Date Features"):
        date_columns = data.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if st.checkbox(f"Extract features from {col}"):
                data[f"{col}_year"] = data[col].dt.year
                data[f"{col}_month"] = data[col].dt.month
                data[f"{col}_day"] = data[col].dt.day
                data[f"{col}_dayofweek"] = data[col].dt.dayofweek
    
    return data

# Function for statistical analysis
def statistical_analysis(data):
    st.subheader("📈 Statistical Analysis")
    
    with st.expander("Descriptive Statistics"):
        st.dataframe(data.describe())
    
    with st.expander("Correlation Analysis"):
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    
    with st.expander("Hypothesis Testing"):
        test_type = st.selectbox("Choose test type", ["T-Test", "ANOVA", "Chi-Square", "Mann-Whitney U"])
        col1 = st.selectbox("Select first column", data.columns)
        col2 = st.selectbox("Select second column", data.columns)
        if test_type == "T-Test":
            t_stat, p_value = stats.ttest_ind(data[col1], data[col2])
            st.write(f"T-Statistic: {t_stat}, P-Value: {p_value}")
        elif test_type == "ANOVA":
            st.write("ANOVA test not implemented in this example")
        elif test_type == "Chi-Square":
            st.write("Chi-Square test not implemented in this example")
        elif test_type == "Mann-Whitney U":
            u_stat, p_value = stats.mannwhitneyu(data[col1], data[col2])
            st.write(f"U-Statistic: {u_stat}, P-Value: {p_value}")

# Function for visualizations
def visualization(data):
    st.subheader("🎨 Data Visualization")
    
    plot_type = st.selectbox("Choose Plot Type", [
        "Scatter Plot", "Line Plot", "Histogram", "Box Plot", "Violin Plot",
        "Bar Plot", "Pie Chart", "Pair Plot", "Heatmap"
    ])
    
    if plot_type == "Scatter Plot":
        x_col = st.selectbox("X-Axis", data.columns)
        y_col = st.selectbox("Y-Axis", data.columns)
        fig = px.scatter(data, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif plot_type == "Line Plot":
        x_col = st.selectbox("X-Axis", data.columns)
        y_col = st.selectbox("Y-Axis", data.columns)
        fig = px.line(data, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif plot_type == "Histogram":
        col = st.selectbox("Select Column", data.columns)
        fig = px.histogram(data, x=col)
        st.plotly_chart(fig)
    
    elif plot_type == "Box Plot":
        col = st.selectbox("Select Column", data.columns)
        fig = px.box(data, y=col)
        st.plotly_chart(fig)
    
    elif plot_type == "Violin Plot":
        col = st.selectbox("Select Column", data.columns)
        fig = px.violin(data, y=col)
        st.plotly_chart(fig)
    
    elif plot_type == "Bar Plot":
        col = st.selectbox("Select Column", data.columns)
        fig = px.bar(data, x=col, y=data.index)
        st.plotly_chart(fig)
    
    elif plot_type == "Pie Chart":
        col = st.selectbox("Select Column", data.columns)
        fig = px.pie(data, names=col)
        st.plotly_chart(fig)
    
    elif plot_type == "Pair Plot":
        sns.pairplot(data)
        st.pyplot()
    
    elif plot_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# Function for natural language processing
def nlp_analysis(data):
    st.subheader("🗣️ Natural Language Processing")
    
    with st.expander("Word Cloud"):
        col = st.selectbox("Select Text Column for Word Cloud", data.columns)
        text = ' '.join(data[col].astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
    
    with st.expander("Sentiment Analysis"):
        col = st.selectbox("Select Text Column for Sentiment Analysis", data.columns)
        data['Sentiment'] = data[col].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        fig = px.histogram(data, x='Sentiment')
        st.plotly_chart(fig)

# Function for clustering
def clustering(data):
    st.subheader("🔍 Clustering")
    
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
    method = st.selectbox("Choose Clustering Method", ["KMeans", "Agglomerative Clustering"])
    
    if method == "KMeans":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clusters)
    elif method == "Agglomerative Clustering":
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < 2:
        st.warning("Clustering requires at least two numeric columns.")
    else:
        model.fit(data[numeric_columns])
        data['Cluster'] = model.labels_
        fig = px.scatter_matrix(data, dimensions=numeric_columns, color='Cluster')
        st.plotly_chart(fig)

# Function for dimensionality reduction
def dimensionality_reduction(data):
    st.subheader("🔻 Dimensionality Reduction")
    
    method = st.selectbox("Choose Method", ["PCA", "t-SNE"])
    n_components = st.slider("Select Number of Components", 2, 10, 2)
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) < n_components:
        st.warning(f"Selected number of components ({n_components}) is greater than the number of numeric columns ({len(numeric_columns)}).")
    else:
        if method == "PCA":
            model = PCA(n_components=n_components)
        elif method == "t-SNE":
            model = TSNE(n_components=n_components)
        
        reduced_data = model.fit_transform(data[numeric_columns])
        reduced_df = pd.DataFrame(reduced_data, columns=[f"Component_{i+1}" for i in range(n_components)])
        reduced_df['Cluster'] = data['Cluster'] if 'Cluster' in data.columns else 0
        fig = px.scatter_matrix(reduced_df, dimensions=reduced_df.columns[:-1], color='Cluster')
        st.plotly_chart(fig)

# # Function for generating pandas profiling report
# def generate_eda_report(data):
#     st.subheader("📋 Automated EDA Report")
#     report = data.profile_report(title="Pandas Profiling Report")
#     st_profile_report(report)

# Function for time series analysis
def time_series_analysis(data):
    st.subheader("⏰ Time Series Analysis")
    
    date_col = st.selectbox("Select Date Column", data.select_dtypes(include=['datetime64']).columns)
    value_col = st.selectbox("Select Value Column", data.select_dtypes(include=[np.number]).columns)
    
    data.set_index(date_col, inplace=True)
    decomposition = sm.tsa.seasonal_decompose(data[value_col], model='additive')
    fig = decomposition.plot()
    st.pyplot(fig)

# Main app logic
def main():
    st.sidebar.title("Upload and Settings")
    
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            data = pd.read_excel(uploaded_file)
        
        load_data(data)
        
        data = Data_preprocessing(data)
        data = Drop_columns(data)
        data = feature_engineering(data)
        
        analysis_type = st.sidebar.selectbox("Choose Analysis Type", [
            "Statistical Analysis", "Data Visualization", "Natural Language Processing",
            "Clustering", "Dimensionality Reduction", "Automated EDA Report", "Time Series Analysis"
        ])
        
        if analysis_type == "Statistical Analysis":
            statistical_analysis(data)
        elif analysis_type == "Data Visualization":
            visualization(data)
        elif analysis_type == "Natural Language Processing":
            nlp_analysis(data)
        elif analysis_type == "Clustering":
            clustering(data)
        elif analysis_type == "Dimensionality Reduction":
            dimensionality_reduction(data)
        # elif analysis_type == "Automated EDA Report":
        #     generate_eda_report(data)
        elif analysis_type == "Time Series Analysis":
            time_series_analysis(data)

if __name__ == "__main__":
    main()

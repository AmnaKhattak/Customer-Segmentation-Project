import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.title("🛍️ Customer Segmentation Web App")
st.write("Yeh app K-Means Clustering aur PCA ka use karke mall customers ko segments me divide karti hai.")

# 1. Load Data
@st.cache_data
def load_data():
    return pd.read_csv('data/Mall_Customers.csv')

try:
    df = load_data()
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

    # Features selection
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])

    df['Cluster'] = pipeline.fit_predict(X)

    st.subheader("Clustering Results Summary")
    st.write(df['Cluster'].value_counts())

    # Visualization for Streamlit
    st.subheader("PCA Cluster Visualization")
    
    scaled_data = pipeline.named_steps['scaler'].transform(X)
    pca_data = pipeline.named_steps['pca'].transform(scaled_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=pca_data[:, 0], y=pca_data[:, 1], 
        hue=df['Cluster'], palette='viridis', s=100, alpha=0.8, edgecolor='k', ax=ax
    )
    ax.set_title('Customer Segments Visualized via PCA')
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Kripya ensure karein ke 'data/Mall_Customers.csv' file GitHub par sahi path par maujood hai.")

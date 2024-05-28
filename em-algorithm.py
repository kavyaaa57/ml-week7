import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris

# Load the dataset
dataset = load_iris()
X = pd.DataFrame(dataset.data)
X.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']
y = pd.DataFrame(dataset.target)
y.columns = ['Targets']

# Define colormap
colormap = np.array(['red', 'lime', 'black'])

# Create a Streamlit app
st.title('Iris Dataset Clustering')
st.write('This app performs clustering on the Iris dataset using KMeans and Gaussian Mixture Models (GMM).')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    n_clusters = st.sidebar.slider('Number of clusters for KMeans and GMM', 2, 10, 3)
    return n_clusters

n_clusters = user_input_features()

# Display the real classification plot
st.subheader('Real Classification')
fig, ax = plt.subplots()
ax.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets], s=40)
ax.set_title('Real')
st.pyplot(fig)

# KMeans Clustering
st.subheader('KMeans Clustering')
model = KMeans(n_clusters=n_clusters)
model.fit(X)
predY = np.choose(model.labels_, list(range(n_clusters))).astype(np.int64)
fig, ax = plt.subplots()
ax.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY % 3], s=40)  # Use modulo to avoid index errors
ax.set_title('KMeans')
st.pyplot(fig)

# GMM Clustering
st.subheader('GMM Clustering')
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(xs)
y_cluster_gmm = gmm.predict(xs)
fig, ax = plt.subplots()
ax.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm % 3], s=40)  # Use modulo to avoid index errors
ax.set_title('GMM Classification')
st.pyplot(fig)

st.write('Note: The colors represent different clusters or classes.')

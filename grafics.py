# Imports
from src.carga_dados import carregar_dados

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import plotly.express as px
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

# Data Loader
dataset = 'dataset/post_pandemic_remote_work_health_impact_2025.csv'
df = carregar_dados(dataset)

# Data processing for clustering
map_burnout = {'Low': 1, 'Medium': 2, 'High': 3}
df['Burnout_Level_Num'] = df['Burnout_Level'].map(map_burnout)

cols_cluster = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 
                'Social_Isolation_Score', 'Burnout_Level_Num']
df_cluster = df[cols_cluster].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

k = 3 
kmeans = KMeans(n_clusters=k, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

st.title("Média por cluster")
cluster = df_cluster.groupby('cluster').mean()
cluster

X = df[['Age', 'Hours_Per_Week', 'Burnout_Level_Num','Work_Life_Balance_Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

df_cluster.groupby('cluster').mean()

fig = px.scatter(
    df_cluster, x='Age', y='Burnout_Level_Num', color='cluster',
    hover_data=['Hours_Per_Week', 'Work_Life_Balance_Score']
)
fig.show()

# Interface
st.title("Visualização de Clusters")
cluster_id = st.selectbox("Selecione o cluster:", sorted(df_cluster['cluster'].unique()))

# Filtrar os registros do cluster escolhido
cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
st.write(f"Total de registros no cluster {cluster_id}: {len(cluster_data)}")
st.dataframe(cluster_data)

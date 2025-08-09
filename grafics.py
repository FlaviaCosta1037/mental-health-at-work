# Imports
from src.carga_dados import carregar_dados
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# =========================
# App Header
# =========================
st.set_page_config(page_title="Workplace Mental Health Analysis", page_icon="🧠", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🧠 Workplace Mental Health Analysis
    </h1>
    <p style='text-align: center; color: gray;'>
        Clustering project to understand mental health and well-being patterns in the post-pandemic workplace
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# =========================
# Data Loading
# =========================
dataset = 'dataset/post_pandemic_remote_work_health_impact_2025.csv'
df = carregar_dados(dataset)

# Mapping values
map_burnout = {'Low': 1, 'Medium': 2, 'High': 3}
df['Burnout_Level_Num'] = df['Burnout_Level'].map(map_burnout)
df['Mental_Health_Status_Num'] = df['Mental_Health_Status'].notna().astype(int)

cols_cluster = [
    'Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 
    'Social_Isolation_Score', 'Burnout_Level_Num', 'Mental_Health_Status_Num'
]

df_cluster = df[cols_cluster].dropna()

# =========================
# Clustering
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

k = 3 
kmeans = KMeans(n_clusters=k, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

# Show cluster averages
st.subheader("📊 Cluster Averages")
st.dataframe(df_cluster.groupby('cluster').mean())

# =========================
# PCA and Visualization
# =========================
X = df[['Age', 'Hours_Per_Week', 'Burnout_Level_Num','Work_Life_Balance_Score','Mental_Health_Status_Num']].dropna()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

kmeans_pca = KMeans(n_clusters=3, random_state=42)
clusters = kmeans_pca.fit_predict(X_pca)

fig = px.scatter(
    df_cluster, x='Age', y='Burnout_Level_Num', color='cluster',
    hover_data=['Hours_Per_Week', 'Work_Life_Balance_Score','Mental_Health_Status_Num'],
    title="Age vs Burnout Level by Cluster"
)
st.plotly_chart(fig, use_container_width=True)

# =========================
# Cluster Filter
# =========================
st.subheader("🔍 Explore Records by Cluster")
cluster_id = st.selectbox("Select cluster:", sorted(df_cluster['cluster'].unique()))
cluster_data = df_cluster[df_cluster['cluster'] == cluster_id]
st.write(f"Total records in cluster {cluster_id}: **{len(cluster_data)}**")
st.dataframe(cluster_data)

# =========================
# Footer
# =========================
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 14px;'>
        Developed by <b>Flávia Costa</b> | Data Science Project - 2025<br>
        Source code available on <a href='https://github.com/FlaviaCosta1037/mental-health-at-work' target='_blank'>GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)

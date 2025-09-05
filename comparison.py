from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import joblib
import pandas as pd

def predict_channel_popularity(channel_data):
    # Load trained model
    kmeans = joblib.load("kmeans_model.pkl")

    # Select same features used during training
    features = ["log_views_per_video", "log_subs_per_video", "log_engagement"]
    X_new = channel_data[features]

    # Predict clusters for all rows
    clusters = kmeans.predict(X_new)

    # Load cluster order to label mapping
    cluster_order = (
        pd.read_csv("clustered_data.csv")
        .groupby("cluster")["advanced_engagement"]
        .mean()
        .sort_values()
        .index
    )

    popularity_labels = {
        cluster_order[0]: "Low Popularity",
        cluster_order[1]: "Medium Popularity",
        cluster_order[2]: "High Popularity"
    }

    # Map each cluster to its label
    cluster_labels = [popularity_labels[c] for c in clusters]

    return clusters, cluster_labels

def clustering(channels_data):
    clusters, popularity_labels = predict_channel_popularity(channels_data)
    channels_data['cluster'] = clusters
    channels_data['popularity_level'] = popularity_labels
    return channels_data

# Visualize with PCA for clarity
def visualize(channels_data):
    if len(channels_data) > 1:
        st.subheader("Engagement Comparison")
        st.bar_chart(
            channels_data.set_index('title')['engagement']
        )
 
        X = channels_data[['views_per_video_scaled', 'subscribers_per_video_scaled', 'engagement_scaled']]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        channels_data['pca1'] = X_pca[:, 0]
        channels_data['pca2'] = X_pca[:, 1]
        # Shift advanced_engagement to be all positive for marker size
        channels_data['adv_engagement_plot'] = channels_data['advanced_engagement'] - channels_data['advanced_engagement'].min() + 1

        # Interactive scatter plot with Plotly

        fig = px.scatter(
            channels_data,
            x='pca1',
            y='pca2',
            color='popularity_level',
            size='adv_engagement_plot',   # bigger circles = more advanced engagement
            hover_data={
                'title': True,
                'viewCount': True,
                'subscriberCount': True,
                'videoCount': True,
                'advanced_engagement': True,
                'pca1': False,
                'pca2': False
            },
            color_discrete_map={
                'Low Popularity': 'red',
                'Medium Popularity': 'orange',
                'High Popularity': 'green'
            },
            title="YouTube Channel Popularity Comparison"
        )

        fig.update_layout(
            legend_title="Popularity Level",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)






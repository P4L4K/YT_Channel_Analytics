from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import joblib
import pandas as pd

def predict_channel_popularity(channel_data):
    # Load trained model
    kmeans = joblib.load("model_data//kmeans_model.pkl")

    # Select same features used during training
    features = ["log_views_per_video", "log_subs_per_video", "log_engagement"]
    X_new = channel_data[features]

    # Predict clusters for all rows
    clusters = kmeans.predict(X_new)

    # Load cluster order to label mapping
    cluster_order = (
        pd.read_csv("csv_files//clustered_data.csv")
        .groupby("cluster")["advanced_popularity"]
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

def assign_rank_with_cluster_boost(channels_data):
    # Map cluster popularity label to numerical boost
    cluster_boost_map = {
        "High Popularity": 3,
        "Medium Popularity": 2,
        "Low Popularity": 1
    }
    # Add boost to advanced_popularity score
    channels_data['adjusted_popularity'] = (
        channels_data['advanced_popularity'] + channels_data['popularity_level'].map(cluster_boost_map)
    )

    # Rank all channels descending by adjusted popularity
    channels_data['rank'] = channels_data['adjusted_popularity'].rank(method='dense', ascending=False).astype(int)

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
        # Shift advanced_popularity to be all positive for marker size
        channels_data['adv_popularity_plot'] = channels_data['advanced_popularity'] - channels_data['advanced_popularity'].min() + 1

        # Interactive scatter plot with Plotly
        fig = px.scatter(
            channels_data,
            x='pca1',
            y='pca2',
            color='popularity_level',
            size='adv_popularity_plot',
            hover_data={
                'title': True,
                'viewCount': True,
                'subscriberCount': True,
                'videoCount': True,
                'advanced_popularity': True,
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
            xaxis_title="Summary Popularity Metric 1",
            yaxis_title="Summary Popularity Metric 2",
            template="plotly_white",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Radar chart for top 3 channels by adjusted popularity
        import plotly.graph_objects as go

        # Use the SAME metrics that drive the advanced_popularity calculation
        display_categories = [
            'View Count',
            'Subscriber Count', 
            'Engagement Ratio',
            'Video Count'
        ]

        metric_map = {
            'View Count': 'viewCount_scaled',
            'Subscriber Count': 'subscriberCount_scaled',
            'Engagement Ratio': 'engagement_scaled',
            'Video Count': 'videoCount_scaled'
        }

        top_channels = channels_data.sort_values(by='adjusted_popularity', ascending=False).head(3)
        channel_titles = top_channels['title'].tolist()
        df = channels_data

        colors = ['blue', 'orange', 'green']
        fig_radar = go.Figure()
        for i, title in enumerate(channel_titles):
            row = df[df['title'] == title].iloc[0]
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[metric_map[cat]] for cat in display_categories],
                theta=display_categories,
                fill='toself',
                name=title,
                line_color=colors[i]
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Top 3 Channels Metrics Radar Chart (Ranking Components)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

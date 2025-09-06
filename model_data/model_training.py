from package.data_preprocessing import data_prep
import pandas as pd
import joblib
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go



def clustering(channels_data):
    # Define feature sets
    feature_sets = {
        "scaled": ["views_per_video_scaled", "subscribers_per_video_scaled", "engagement_scaled"],
        "log": ["log_views_per_video", "log_subs_per_video", "log_engagement"]
    }

    best_overall_score = -1
    best_labels = None
    best_kmeans = None
    best_feature_set = None

    # Evaluate each feature set with fixed 3 clusters
    for feature_type, features in feature_sets.items():
        X = channels_data[features]

        kmeans = KMeans(n_clusters=3, n_init=10, random_state=80)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)

        print(f"🔎 Feature set: {feature_type}, Silhouette Score: {score:.3f}")

        if score > best_overall_score:
            best_overall_score = score
            best_labels = labels
            best_kmeans = kmeans
            best_feature_set = feature_type

    # Assign best clustering labels
    channels_data["cluster"] = best_labels

    # Rank clusters by average engagement
    cluster_order = (
        channels_data.groupby("cluster")["advanced_popularity"]
        .mean()
        .sort_values()
        .index
    )

    popularity_labels = {
        cluster_order[0]: "Low Popularity",
        cluster_order[1]: "Medium Popularity",
        cluster_order[2]: "High Popularity",
    }
    channels_data["popularity_level"] = channels_data["cluster"].map(popularity_labels)

    print(
        f"✅ Best feature set: {best_feature_set}, "
        f"Clusters: 3, Best Silhouette Score: {best_overall_score:.3f}"
    )

    # Save model
    joblib.dump(best_kmeans, "kmeans_model.pkl")

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


def visualize(channels_data):
    # Engagement bar chart
    fig_bar = px.bar(
        channels_data,
        x="title",
        y="engagement",
        title="Engagement Comparison",
        labels={"engagement": "Engagement Score", "title": "Channel"},
    )
    fig_bar.show()

    # PCA visualization (always use scaled features)
    X = channels_data[
        ["views_per_video_scaled", "subscribers_per_video_scaled", "engagement_scaled"]
    ]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    channels_data["pca1"] = X_pca[:, 0]
    channels_data["pca2"] = X_pca[:, 1]

    # Shift advanced_popularity to be all positive for marker size
    channels_data['adv_popularity_plot'] = channels_data['advanced_popularity'] - channels_data['advanced_popularity'].min() + 1

    # Scatter plot
    fig_scatter = px.scatter(
        channels_data,
        x="pca1",
        y="pca2",
        color="popularity_level",
        size="adv_popularity_plot",
        hover_data={
            "title": True,
            "viewCount": True,
            "subscriberCount": True,
            "videoCount": True,
            "advanced_popularity": True,
            "pca1": False,
            "pca2": False,
        },
        color_discrete_map={
            "Low Popularity": "red",
            "Medium Popularity": "orange",
            "High Popularity": "green",
        },
        title="YouTube Channel Popularity Comparison",
    )

    fig_scatter.update_layout(
    legend_title="Popularity Level",
    xaxis_title="Overall Popularity Factor",
    yaxis_title="Secondary Popularity Factor",
    template="plotly_white",
    height=600,
    )
   
    fig_scatter.show()
    # Radar chart for top 3 channels by adjusted popularity
    display_categories = [
        'Avg Views per Video',
        'Avg Subs per Video',
        'Engagement Ratio'
    ]

    # Mapping of display names to DataFrame column names
    metric_map = {
        'Avg Views per Video': 'views_per_video_scaled',
        'Avg Subs per Video': 'subscribers_per_video_scaled',
        'Engagement Ratio': 'engagement_scaled'
    }

    top_channels = channels_data.sort_values(by='adjusted_popularity', ascending=False).head(3)
    channel_titles = top_channels['title'].tolist()
    df = channels_data  # Reference to full DataFrame for metrics

    fig = go.Figure()
    for title in channel_titles:
        row = df[df['title'] == title].iloc[0]
        fig.add_trace(go.Scatterpolar(
            r=[row[metric_map[cat]] for cat in display_categories],
            theta=display_categories,
            fill='toself',
            name=title
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Top 3 Channels Metrics Radar Chart"
    )
    fig.show()



# Run pipeline
if __name__ == "__main__":
    data = pd.read_csv("csv_files//channel_ids_detail_training_set.csv")
    data = data_prep(data)
    data = clustering(data)
    data = assign_rank_with_cluster_boost(data)
    data.to_csv("csv_files//clustered_data.csv", index=False)
    visualize(data)

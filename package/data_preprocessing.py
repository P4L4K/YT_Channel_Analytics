import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_prep(channels_data):
    # Check and drop duplicate channel IDs
    dupes = channels_data[channels_data.duplicated(subset=["channelId"], keep=False)]
    if not dupes.empty:
        print("Found duplicate channelIds:\n", dupes[["channelId", "title"]])
        channels_data = (
            channels_data
            .drop_duplicates(subset=["channelId"], keep="first")
            .reset_index(drop=True)
        )
    else:
        print("No duplicates found")

    # Convert columns to numeric
    for col in ["viewCount", "subscriberCount", "videoCount"]:
        channels_data[col] = pd.to_numeric(channels_data[col], errors="coerce").fillna(0)

    # Feature engineering
    channels_data["views_per_video"] = channels_data["viewCount"] / channels_data["videoCount"].replace(0, 1)
    channels_data["subs_per_video"] = channels_data["subscriberCount"] / channels_data["videoCount"].replace(0, 1)
    channels_data["engagement"] = channels_data["viewCount"] / channels_data["subscriberCount"].replace(0, 1)

    # Apply log transform to handle huge skews (YouTube counts vary a lot!)
    for col in ["viewCount", "subscriberCount", "videoCount", "views_per_video", "subs_per_video", "engagement"]:
        channels_data[f"log_{col}"] = np.log1p(channels_data[col])  # log1p handles log(0)

    # Normalize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        channels_data[
            ["log_viewCount", "log_subscriberCount", "log_videoCount",
             "log_views_per_video", "log_subs_per_video", "log_engagement"]
        ]
    )
    scaled_df = pd.DataFrame(
        scaled,
        columns=[
            "viewCount_scaled", "subscriberCount_scaled", "videoCount_scaled",
            "views_per_video_scaled", "subscribers_per_video_scaled", "engagement_scaled"
        ]
    )
    
    # Merge scaled data with original
    channels_data = pd.concat([channels_data, scaled_df], axis=1)
    # Advanced popularity (rebalanced weights)
    channels_data["advanced_popularity"] = (
        0.5 * channels_data["subscriberCount_scaled"] +   # subscribers weigh more for popularity
        0.3 * channels_data["viewCount_scaled"] +         # views still important
        0.1 * channels_data["engagement_scaled"] +        # engagement ratio adds nuance
        -0.1 * channels_data["videoCount_scaled"]          # penalize too many videos (quality over quantity)
    )

    return channels_data

from googleapiclient.discovery import build
import pandas as pd
import streamlit as st

api_key = st.secrets['yt_api_key']
youtube = build("youtube", "v3", developerKey=api_key)

def fetch_channel_details(channel_ids):
    try:
        channels_data = []
        # Process in batches of 50
        for i in range(0, len(channel_ids), 50):
            batch_ids = channel_ids[i:i+50]
            request = youtube.channels().list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch_ids)
            )
            response = request.execute()

            for channel in response.get("items", []):
                stats = channel.get("statistics", {})
                snippet = channel.get("snippet", {})
                data = {
                    "channelId": channel.get("id"),
                    "title": snippet.get("title", "Unknown"),
                    "viewCount": stats.get("viewCount", 0),
                    "subscriberCount": stats.get("subscriberCount", 0) if not stats.get("hiddenSubscriberCount", False) else 0,
                    "videoCount": stats.get("videoCount", 0)
                }
                channels_data.append(data)

        return pd.DataFrame(channels_data)

    except Exception as e:
        print(f" ERROR: {e}")
        return pd.DataFrame()

        

channels_ids = pd.read_csv("channel_ids.csv")["channelId"].dropna().astype(str)
channels_ids = [cid.strip() for cid in channels_ids if cid.strip() != ""]
print("Channel IDs being sent:", channels_ids)  # debug

data=fetch_channel_details(channels_ids)
# Create a DataFrame and save to CSV
data.to_csv('channel_ids_art.csv', index=False)
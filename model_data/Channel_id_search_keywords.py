from googleapiclient.discovery import build
import pandas as pd
import streamlit as st


api_key = st.secrets['yt_api_key']
api_service_name = 'youtube'
api_version = 'v3'
youtube = build(api_service_name, api_version, developerKey=api_key)


request = youtube.search().list(
    part="snippet",
    maxResults=50,
    q="art",
    type="channel"
)
response = request.execute()

# Extract channel IDs
channel_ids = [item['id']['channelId'] for item in response['items']]

# Create a DataFrame and save to CSV
df = pd.DataFrame(channel_ids, columns=['channelId'])
df.to_csv('channel_ids_file.csv', index=False)

print("Channel IDs saved to channel_ids.csv")

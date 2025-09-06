from googleapiclient.discovery import build
import pandas as pd
import streamlit as st

api_key=st.secrets['yt_api_key']
#channel_ids=["UCeVMnSShP_Iviwkknt83cww","UCWv7vMbMWH4-V0ZXdmDpPBA","UCBwmMxybNva6P_5VmxjzwqA","UC8butISFwT-Wl7EV0hUK0BQ"]
api_service_name='youtube'
api_version='v3'
youtube = build(api_service_name, api_version, developerKey=api_key)

def fetch_channel_details(channel_ids):
        try:
            request=youtube.channels().list(
                part="snippet,contentDetails,statistics", #required outputs
                id=",".join(channel_ids)
            )
            response=request.execute()
            channels_data=[]
            for channel in response['items']:
                channel_stats=channel['statistics']
                channel_title=channel['snippet']['title']
                channel_stats['title']=channel_title
                channel_stats[ "channelId"]= channel["id"]
                if channel_stats['hiddenSubscriberCount']==True:
                    print(f"Can't compare {channel_title} as content is hidden")
                else:
                    del channel_stats['hiddenSubscriberCount']
                    channels_data.append(channel_stats)
            return pd.DataFrame(channels_data)
        except Exception as e:
            print(f" ERROR: {e}")
            return pd.DataFrame()
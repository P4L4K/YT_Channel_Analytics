import streamlit as st
from channel_data import fetch_channel_details
from data_preprocessing import data_prep
from comparison import clustering, visualize
st.write('YT_Channel_Analytics: A Machine Learning Powered YouTube Popularity Analyzer')

# Input box for user channel IDs
channel_ids_input = st.text_area(
    "Enter YouTube Channel IDs (comma-separated):",
    placeholder="e.g. UCeVMnSShP_Iviwkknt83cww, UCWv7vMbMWH4-V0ZXdmDpPBA, UCBwmMxybNva6P_5VmxjzwqA",
    height=100
)

# Convert input to list
channel_ids = [cid.strip() for cid in channel_ids_input.split(",") if cid.strip()]

if st.button('Analyse'):
        data=fetch_channel_details(channel_ids)
        if data.empty:
            st.error("Data can't be fetched!")
        else:
            st.success("Data Fetched")
            data=data_prep(data)
            data=clustering(data)
            #show clustered data
            st.subheader("Channels with Popularity Levels")
            st.dataframe(data[['title','viewCount','subscriberCount','videoCount','popularity_level','channelId']])
            if len(data) >1:
                st.subheader("Cluster Summary")
                st.dataframe(data.groupby('popularity_level').size().reset_index(name='count'))
                visualize(data)

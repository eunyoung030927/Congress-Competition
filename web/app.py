import streamlit as st
from cluster import main as cluster_main
# from chat_with_documents import main as chat_with_documents_main
from top5_recommendation import main as top5_recommendation
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

st.set_page_config(page_title="Congress 신문고", page_icon="🏛️")

st.title("Congress 신문고")

st.write("Select a feature from the sidebar to get started.")

with st.sidebar:
    page = st.radio('Go to', ('Home', 'Top 5 Documentation Recommendation','cluster'))

if page == 'Home':
    st.write("Welcome to the Congress 신문고! Choose an option from the sidebar.")
elif page == 'Top 5 Documentation Recommendation':
    top5_recommendation()
elif page == 'cluster':
    cluster_main()
# elif page == 'Chat with Documents':
#     chat_with_documents_main()

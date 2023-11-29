import streamlit as st
from basic_streaming import main as basic_streaming_main
from chat_with_documents import main as chat_with_documents_main
from top5_recommendation import main as top5_recommendation
import sys
import os

# 현재 파일의 디렉토리 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Main page configuration
st.set_page_config(page_title="Congress 신문고", page_icon="🏛️")

# Main page title
st.title("Congress 신문고")

# Navigation instructions
st.write("Select a feature from the sidebar to get started.")

# Sidebar for navigation
with st.sidebar:
    page = st.radio('Go to', ('Home', 'Top 5 Documentation Recommendation', 'Basic Streaming', 'Chat with Documents'))

# Page navigation
if page == 'Home':
    st.write("Welcome to the Congress 신문고! Choose an option from the sidebar.")
elif page == 'Top 5 Documentation Recommendation':
    top5_recommendation()
elif page == 'Basic Streaming':
    basic_streaming_main()
elif page == 'Chat with Documents':
    chat_with_documents_main()

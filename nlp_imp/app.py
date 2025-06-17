import streamlit as st
from helpers import get_vid_path

st.title("NLP Based Sign Language GIF Generator")
st.write("Enter text to see its Sign language Translation")

user_input = st.text_input("Input Text", "A")

if user_input:
    vid_path = get_vid_path(user_input)
    if vid_path is not '':
        if vid_path.lower().endswith(('.mp4', '.webm', '.mov', '.avi')):
            st.video(vid_path)
        elif vid_path.lower().endswith(('.gif', '.jpg', '.jpeg', '.png')):
            st.image(vid_path)
        else:
            st.warning(f"Unsupported file type for: {vid_path}")
    else:
        st.error(f"No video found for: {user_input}")
        
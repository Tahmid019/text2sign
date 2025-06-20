import streamlit as st
from helpers import get_vid_path

st.title("NLP Based Sign Language GIF Generator")
st.write("Enter text to see its Sign language Translation")

user_input = st.text_input("Input Text", "")

reset = st.button("Reset")

if reset:
    user_input = ""
    # st.experimental_rerun()

if user_input and not reset:
    vid_path, score, matched = get_vid_path(user_input)
    if vid_path != '':
        if vid_path.lower().endswith(('.mp4', '.webm', '.mov', '.avi')):
            st.video(vid_path)
        elif vid_path.lower().endswith(('.gif', '.jpg', '.jpeg', '.png')):
            st.image(vid_path)
        else:
            st.warning(f"Unsupported file type for: {vid_path}")
        st.write(f"Score: {score}")
    else:
        st.error(f"No video found for: {user_input}")
    st.write(f"Matched: {matched}")
        
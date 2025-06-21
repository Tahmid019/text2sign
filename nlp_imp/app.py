import streamlit as st
import os
from helpers import get_vid_path, get_top_k_matches
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

st.title("NLP Based Sign Language GIF Generator")
st.write("Enter text to see its Sign language Translation")

user_input = st.text_input("Input Text", "")

reset = st.button("Reset")

if reset:
    user_input = ""
    # st.experimental_rerun()


if user_input and not reset:
    st.markdown("**More Options:**")
    suggestions = get_top_k_matches(user_input, k=5)
    
    options = [
        f"{sent} (score={score:.2f})"
        for sent, vid_id, score in suggestions
    ]
    choice = st.selectbox("", options)

    # Extract chosen sentence and score separately
    chosen_sent = choice.split(" (score")[0]
    chosen_score_str = choice.split("score=")[-1].rstrip(")")
    chosen_score = float(chosen_score_str)
    
    vid_path, score, matched, keyframe_path, keyframe_hand_path = get_vid_path(chosen_sent)
    if vid_path != '':
        if vid_path.lower().endswith(('.mp4', '.webm', '.mov', '.avi')):
            st.video(vid_path)
        elif vid_path.lower().endswith(('.gif', '.jpg', '.jpeg', '.png')):
            st.image(vid_path)
        else:
            st.warning(f"Unsupported file type for: {vid_path}")
        
        col1, col2 = st.columns(2)
        with col1:
            if keyframe_path is not None: st.video(keyframe_path)
        
        with col2:
            if keyframe_hand_path is not None: st.video(keyframe_hand_path)

        
        st.write(f"Score: {chosen_score}")
    else:
        st.error(f"No video found for: {user_input}")
    st.write(f"Matched: {matched}")
    # print(keyframe_path, vid_path)
        
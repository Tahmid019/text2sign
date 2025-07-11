from imports import *
from config import *
from pages.text2sign.t2sl import t2sl
# from pages.sign2text.func2 import s2t
from pages.sign2text.func3 import s2t
from pages.sign2sign.s2s import s2s
# from text2sign import *

# mp_client = MediaPipeClient(MP_URL)


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def intro():
    st.write("""
    ## Welcome to the Text2Sign Application

    In the sidebar, you will find three modules:
    - **T2SL (Text to Sign Language):** Convert written text into sign language representations.
    - **SL2T (Sign Language to Text):** Translate sign language gestures or videos into written text.
    - **S2S (Sign to Sign):** Transform one sign language format or style into another.

    Use the sidebar to select and access any of these modules as per your needs.
    """)
    

page_names = {
    "Home" : intro,
    "T2SL" : t2sl,
    "SL2T" : s2t,
    "S2S" : s2s,
}

demo_name = st.sidebar.selectbox("Choose Page", page_names.keys())
page_names[demo_name]()

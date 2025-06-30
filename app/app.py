from imports import *
from config import *
from pages.text2sign.t2sl import t2sl
# from pages.sign2text.MediaPipeClient import MediaPipeClient
from pages.sign2sign.s2s import s2s
# from text2sign import *

# mp_client = MediaPipeClient(MP_URL)


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def intro():
    st.write("Application")
    
# def sl2t_process():
#     result = mp_client.ping_sl2t()
#     st.json(result)

page_names = {
    "_" : intro,
    "T2SL" : t2sl,
    # "SL2T" : sl2t_process,
    "S2S" : s2s,
}

demo_name = st.sidebar.selectbox("Choose Page", page_names.keys())
page_names[demo_name]()

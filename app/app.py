from imports import *
from pages.text2sign.t2sl import t2sl
# from text2sign import *

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def intro():
    st.write("Application")

page_names = {
    "_" : intro,
    "T2SL" : t2sl,
}

demo_name = st.sidebar.selectbox("Choose Page", page_names.keys())
page_names[demo_name]()

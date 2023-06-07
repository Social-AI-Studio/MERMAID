import streamlit as st
import os
import json
import pandas as pd
from PIL import Image



st.title("TD Meme Viewer")

@st.cache_data
def load_all_relevant():
    with open("TDMEMES_predictions.json","r",encoding="utf-8") as predictionsfile:
        td_path_to_images = os.path.join("TDMEMES","TD_Memes")
        tdpreds = json.load(predictionsfile)
        tdpredlist = list(tdpreds.keys())
    
    # with open("normal_model_prediction.json","r",encoding="utf-8") as predictionsfile:
        # normal_path_to_images = os.path.join("parse_annotated_results","raw_data_jsons","image_dir")
        # normaljson = json.load(predictionsfile)    
    return td_path_to_images,tdpreds,tdpredlist
    



td_path_to_images,tdpreds,tdpredlist = load_all_relevant()

if not "target_num" in st.session_state:
    st.session_state.target_num = 0


target_number = st.number_input("Target value",min_value=0,max_value=len(tdpredlist)-1,key="target_num")

infodict = tdpreds[tdpredlist[st.session_state.target_num]]

st.image(Image.open(os.path.join(td_path_to_images,tdpredlist[st.session_state.target_num])))
heldlist = [] # for constant memory
for textbox in infodict:
    heldlist.append(st.subheader("Original Text"))
    heldlist.append(st.text(infodict[textbox]["Original_Text"]))
    heldlist.append(st.subheader("Extracted Entity"))
    heldlist.append(st.text(infodict[textbox]["Predicted_Entities"]))
    # infodict[textbox]["predicted_sigmoided"]
    # beginflaglist = []
    # stopflaglist = []
    # for i in z:
        # beginflaglist.append(i[0])
        # stopflaglist.append(i[1])
    st.dataframe(pd.DataFrame(infodict[textbox]["predicted_sigmoided"],columns=["Start Flag","Stop Flag"]))

    
    
    
# st.session_state.
    
    
# if st.button("Load Dataset Memes"):
    # pass


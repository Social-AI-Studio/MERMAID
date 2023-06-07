import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from PIL import Image



st.title("TD Meme Viewer")

@st.cache_data
def load_all_relevant():
    with open("TDMEMES_predictions_position_abalated.json","r",encoding="utf-8") as predictionsfile:
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

st.image(Image.open(os.path.join(td_path_to_images,tdpredlist[st.session_state.target_num].split("\\")[-1])))
heldlist = [] # for constant memory
predicted_entities_list = []
textboxes_set = set()
sigmoid_dict = {}

for predicted_entities in infodict[0]:
    predicted_entities_list.append(predicted_entities)
    print(infodict[0][predicted_entities])
    textboxes_set.add(infodict[0][predicted_entities]["Original_Text"])
    sigmoid_dict[infodict[0][predicted_entities]["Original_Text"]] = infodict[0][predicted_entities]["predicted_sigmoided"]

heldlist.append(st.subheader("Original Textboxes"))
for textbox in textboxes_set:
    heldlist.append(st.text(textbox))
    heldlist.append(st.divider())


heldlist.append(st.subheader("Extracted Entities"))

for predicted_entities in predicted_entities_list:
    heldlist.append(st.text(predicted_entities))
    heldlist.append(st.divider())
    # infodict[textbox]["predicted_sigmoided"]
    # beginflaglist = []
    # stopflaglist = []
    # for i in z:
        # beginflaglist.append(i[0])
        # stopflaglist.append(i[1])

st.subheader("Relations Extracted")
for idx in infodict[1]:
    # print(infodict[1])
    st.text("Sender: " + infodict[1][idx]["text (Sender)"])
    st.text("Selected Relation: " + infodict[1][idx]["selected_relation"])
    st.text("Receiver: " +infodict[1][idx]["text (Receiver)"])
    st.table(pd.DataFrame( np.array([infodict[1][idx]["Prediction (Logits)"]]).transpose(0,1),columns=["Superior", "Equal", "Upgrade", "Degrade", "Affirm/Favor", "Doubt/Disfavor", "Indifferent", "Inferior", "NULL"])) 

st.subheader("Raw Sigmoids")
for matchable in sigmoid_dict:
    st.text(matchable)
    st.dataframe(pd.DataFrame(sigmoid_dict[matchable],columns=["Start Flag","Stop Flag"]))
    
    
    
    
# st.session_state.
    
    
# if st.button("Load Dataset Memes"):
    # pass

# "TDMEMES\\TD_Memes\\img_4120.jpg": [
        # {
            # "a hdb flat finishes it ' s 99 - year lease": {
                # "predicted_sigmoided": [
                    # [
                        # 0.0027565171476453543,
                        # 0.0004948354908265173
                    # ],
                    # [
                        # 0.7827731966972351,
                        # 0.008421136066317558
                    # ],
                    # [
                        # 0.5517891645431519,
                        # 0.019519733265042305
                    # ],
                    # [
                        # 0.07075908035039902,
                        # 0.0006139228353276849
                    # ],
                    # [
                        # 0.002924407599493861,
                        # 0.0004482320509850979
                    # ],
                    # [
                        # 0.020396791398525238,
                        # 0.0003667041892185807
                    # ],
                    # [
                        # 0.0007138854707591236,
                        # 0.0067623071372509
                    # ],
                    # [
                        # 0.0017791095888242126,
                        # 0.000363662518793717
                    # ],
                    # [
                        # 0.0003775234508793801,
                        # 0.0006883882451802492
                    # ],
                    # [
                        # 0.0003420839784666896,
                        # 0.0010338017018511891
                    # ],
                    # [
                        # 0.2345847487449646,
                        # 0.00038246341864578426
                    # ],
                    # [
                        # 0.0007791852694936097,
                        # 0.0003095630672760308
                    # ],
                    # [
                        # 0.00531024020165205,
                        # 0.0003506634966470301
                    # ],
                    # [
                        # 0.012360000982880592,
                        # 0.5141112804412842
                    # ],
                    # [
                        # 0.0014201250160112977,
                        # 0.02436958998441696
                    # ],
                    # [
                        # 0.004669790156185627,
                        # 0.0006456434493884444
                    # ],
                    # [
                        # 0.007303264923393726,
                        # 0.0002828936849255115
                    # ],
                    # [
                        # 0.005827551707625389,
                        # 0.0008038607193157077
                    # ],
                    # [
                        # 0.14427535235881805,
                        # 0.01210661232471466
                    # ],
                    # [
                        # 0.03230104222893715,
                        # 0.007891529239714146
                    # ],
                    # [
                        # 0.10773176699876785,
                        # 0.002538814442232251
                    # ],
                    # [
                        # 0.012499331496655941,
                        # 0.001014744397252798
                    # ],
                    # [
                        # 0.04027402400970459,
                        # 0.00037110157427378
                    # ],
                    # [
                        # 0.036092609167099,
                        # 0.08589114993810654
                    # ],
                    # [
                        # 0.007753746584057808,
                        # 0.0005679643945768476
                    # ],
                    # [
                        # 0.4479921758174896,
                        # 0.0025244182907044888
                    # ],
                    # [
                        # 0.25373080372810364,
                        # 0.001894633867777884
                    # ],
                    # [
                        # 0.05691097304224968,
                        # 0.0002836112107615918
                    # ],
                    # [
                        # 0.007469854783266783,
                        # 0.5881640911102295
                    # ],
                    # [
                        # 0.003020619275048375,
                        # 0.34373727440834045
                    # ],
                    # [
                        # 0.00044108263682574034,
                        # 0.00015440650167874992
                    # ]
                # ],
                # "predicted_actual": [
                    # [
                        # -5.891026973724365,
                        # -7.610790252685547
                    # ],
                    # [
                        # 1.2819010019302368,
                        # -4.768553733825684
                    # ],
                    # [
                        # 0.207902193069458,
                        # -3.916616678237915
                    # ],
                    # [
                        # -2.575087070465088,
                        # -7.395027160644531
                    # ],
                    # [
                        # -5.831734657287598,
                        # -7.709751129150391
                    # ],
                    # [
                        # -3.871769905090332,
                        # -7.910588264465332
                    # ],
                    # [
                        # -7.244073867797852,
                        # -4.989605903625488
                    # ],
                    # [
                        # -6.329861640930176,
                        # -7.918920516967773
                    # ],
                    # [
                        # -7.881500244140625,
                        # -7.280468940734863
                    # ],
                    # [
                        # -7.980112075805664,
                        # -6.873477935791016
                    # ],
                    # [
                        # -1.1826015710830688,
                        # -7.868494987487793
                    # ],
                    # [
                        # -7.156482219696045,
                        # -8.080039024353027
                    # ],
                    # [
                        # -5.232793807983398,
                        # -7.9553327560424805
                    # ],
                    # [
                        # -4.380852699279785,
                        # 0.05646010488271713
                    # ],
                    # [
                        # -6.555589199066162,
                        # -3.6897478103637695
                    # ],
                    # [
                        # -5.361960411071777,
                        # -7.3446173667907715
                    # ],
                    # [
                        # -4.912103652954102,
                        # -8.170156478881836
                    # ],
                    # [
                        # -5.139313697814941,
                        # -7.125280380249023
                    # ],
                    # [
                        # -1.7802249193191528,
                        # -4.401823043823242
                    # ],
                    # [
                        # -3.3998215198516846,
                        # -4.834042549133301
                    # ],
                    # [
                        # -2.1141223907470703,
                        # -5.97351598739624
                    # ],
                    # [
                        # -4.369502067565918,
                        # -6.89210319519043
                    # ],
                    # [
                        # -3.170941114425659,
                        # -7.898663520812988
                    # ],
                    # [
                        # -3.284907102584839,
                        # -2.3648688793182373
                    # ],
                    # [
                        # -4.851795196533203,
                        # -7.472883701324463
                    # ],
                    # [
                        # -0.20878641307353973,
                        # -5.979217052459717
                    # ],
                    # [
                        # -1.0788124799728394,
                        # -6.266833305358887
                    # ],
                    # [
                        # -2.8076725006103516,
                        # -8.167622566223145
                    # ],
                    # [
                        # -4.8893818855285645,
                        # 0.35638099908828735
                    # ],
                    # [
                        # -5.7992682456970215,
                        # -0.6466835141181946
                    # ],
                    # [
                        # -7.725837230682373,
                        # -8.77576732635498
                    # ]
                # ],
                # "Original_Text": "When a HDB flat finishes it's 99-year lease: This is so sad can we hit a pedestrian with escooters?",
                # "Predicted_Entities": "a hdb flat finishes it ' s 99 - year lease",
                # "detection_count": 1,
                # "meme_type": "SG_Memes",
                # "textboxposition": false
            # }
        # },
        # {
            # "0": {
                # "text (Sender)": "a hdb flat finishes it ' s 99 - year lease",
                # "text (Receiver)": "MEME_CREATOR",
                # "Prediction": 0,
                # "Prediction (Logits)": [
                    # 0.9809771180152893,
                    # 0.921029806137085,
                    # 0.9625465869903564,
                    # 0.3601715564727783,
                    # 0.01595867983996868,
                    # 0.19393838942050934,
                    # 0.3119243383407593,
                    # 0.7171961069107056,
                    # 0.9278624057769775
                # ],
                # "selected_relation": "Superior"
            # },
            # "1": {
                # "text (Sender)": "MEME_CREATOR",
                # "text (Receiver)": "a hdb flat finishes it ' s 99 - year lease",
                # "Prediction": 4,
                # "Prediction (Logits)": [
                    # 0.0001589925232110545,
                    # 1.0098458915308584e-05,
                    # 0.000450662337243557,
                    # 0.026572339236736298,
                    # 0.9537124633789062,
                    # 0.6031327247619629,
                    # 0.00012919952860102057,
                    # 0.0031591092702001333,
                    # 0.12331569194793701
                # ],
                # "selected_relation": "Affirm/Favor"
            # }
        # }
    # ]
import os
import torch
import pprint
import time
import sys
import json
sys.path.append("..") # Adds higher directory 
import utils
import cProfile
import itertools
import PIL.Image
import sklearn.metrics
import numpy as np
from io import BytesIO    
from PIL import Image
from torchvision.transforms import ToTensor
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation
from attached_heads import trilinear_head,deeper_trilinear_head_relations
# torch.set_num_threads(2)
# torch.set_num_interop_threads(2)


# Run an already trained model on the entire TDMEMES dataset.
# Entity only.




def threshold_entity_extractor(input_vector,thresholdvalue):
    # input vector => SEQLEN, 2
    # 2 flags.
    extracted_seqlist = []
    tempholder = []
    
    for idx in range(len(input_vector)):
        if input_vector[idx][0]>thresholdvalue and input_vector[idx][1]>thresholdvalue: # startstops
            extracted_seqlist.append([idx])
        elif input_vector[idx][1]>thresholdvalue:
            if tempholder:
                tempholder.append(idx)
                extracted_seqlist.append(tempholder)
                # print(extracted_seqlist)
                tempholder = []
        elif input_vector[idx][0]>thresholdvalue:
            if tempholder:
                tempholder = [] # delete the phrase we detected.
            tempholder.append(idx)
        else: # 0 0
            if tempholder:
                tempholder.append(idx) # still in the middle of a phrase since a flag was indicated.
            continue
            
    return extracted_seqlist
            

if __name__=="__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prefix = "all_templates"    
    
    entity_threshold = 0.5
    data_dir = os.path.join("TDMEMES","TD_Memes")
    annotation_file = os.path.join("TDMEMES","annotation.json")

    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']
    
    entityheadname = "" # model torch files. Or just comment out the loads below if you're testing.
    entityname = ""
    
    # 76fv3h.jpg # example of multiple entities in a single statement
    
 
    with open(annotation_file,"r",encoding="utf-8") as annotationfile:
        annotations = json.load(annotationfile)
    print(annotations.keys())
    all_valid_image_files = annotations["SG_Memes"] + annotations["Non_SG_Memes"]
    
    class EmptyContext(object): #to ignore nograd if required
        def __init__(self, dummy=None):
            self.dummy = dummy
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    
    targetdumpfile = "TDMEMES_predictions.json"
    
    if os.path.exists(targetdumpfile):
        with open(targetdumpfile,"r",encoding="utf-8") as dumpfileopened:
            prediction_result_dict = json.load(dumpfileopened)
    else:
        prediction_result_dict = {}
    
    
    
    weightdecay = 1e-6
    entity_embed = BERT_preparation(prefix,device)
    # entity_embed = ROBERTA_preparation(prefix,device)
    entity_embed_head = trilinear_head(device,embed_dimension=768,targetsize = 2)
    entity_embed_head.to(entity_embed_head.device)
    entity_reference_vocab_dict = {i:k for k,i in entity_embed.texttokenizer.get_vocab().items()}
    entity_embed.model.load_state_dict(torch.load(entityname))
    entity_embed.model.eval()
    entity_embed_head.load_state_dict(torch.load(entityheadname))
    entity_embed_head.eval()

    
    texts_dict_organised = {}
    for item in annotations["Text"]:
        texts_dict_organised[list(item.keys())[0]] = item[list(item.keys())[0]]
    
    remapped_actionables_dict = {
        "Superior":0,
        "Equal":1,
        "Upgrade":2,
        "Degrade":3,
        "Affirm/Favor":4,
        "Doubt/Disfavor":5,
        "Indifferent":6,
        "Inferior":7,
        "NULL":8,
    }
    
    selected_context = EmptyContext()
    full_entity_report = {}
    
    overdupe_accounting = {}
    
    skipped_images_list = []
    with selected_context:
        for singlesample in all_valid_image_files: # no need to care about idx.
            # print("-"*30)
            # print("equivalent entities:",singlesample["equivalent_entities"])
            sample_entity_report_dict = {}
            
            imagesample = os.path.join(data_dir,singlesample)
            all_textboxes = []
            if singlesample in prediction_result_dict:
                continue
            dupedict = {}
            
            # print(vectorattachment[textname]["text"]) # still need to edit input to take something else instead...
            # textembeds,_ = entity_embed([vectorattachment[textname]["text"]])
            if singlesample in annotations["Non_SG_Memes"]:
                meme_type = "Non_SG_Memes"
            else:
                meme_type = "SG_Memes"
                
                
            annotated_text = texts_dict_organised[singlesample]
            print(annotated_text)
            
            textembeds,_ = entity_embed([annotated_text])
            tokenised_list = entity_embed.texttokenizer([annotated_text])
            print(tokenised_list)
            matched_entity_counter = 0
            correct_count = 0
            # pprint.pprint(singlesample)
            # print(dupedict)
            # print(annotated_text)
            for item in range(len(textembeds)):
                prediction = entity_embed_head(textembeds[item]["last_hidden_state"]).squeeze()
                indicators = entity_embed_head.sigmoid(prediction)
                # print("Tokenised OCR box:", tokenised_list["input_ids"][item])
                # print("OCR box:", annotated_text[item])
                # print("Sigmoided (CUT):",indicators[1:-1])
                # print("Sigmoided UNCUT",indicators)
                predicted_entity_indexes = threshold_entity_extractor(indicators[1:-1],entity_threshold) # predicted version
                print("predicted entity indexes:",predicted_entity_indexes)
                
                detected_entities = []
                for entityindexset in predicted_entity_indexes: 
                    wordlist = []
                    for wordpart in entityindexset:
                        # print(wordpart)
                        # print(tokenised_list["input_ids"][item])
                        wordlist.append(entity_reference_vocab_dict[tokenised_list["input_ids"][item][1:-1][wordpart]])
                    detected_entities.append(wordlist)
                print("detected entities:", detected_entities)
                predicted_entity_strings = []
                for entity in detected_entities:
                    finalstring = ""
                    for stringpart in entity:
                        if "##" in stringpart:
                            finalstring+=stringpart[2:] # instant attachment.
                        else:
                            finalstring+=" "
                            finalstring+=stringpart
                    finalstring = finalstring.strip()
                    predicted_entity_strings.append(finalstring)
                # print("predicted entity strings:",predicted_entity_strings)
                # print(annotated_text[item])
                
                for predicted_entities in predicted_entity_strings:
                    if not predicted_entities in sample_entity_report_dict:
                        sample_entity_report_dict[predicted_entities] = {
                            "predicted_sigmoided":indicators.detach().cpu().numpy().tolist(),
                            "predicted_actual":prediction.detach().cpu().numpy().tolist(),
                            "Original_Text":annotated_text,
                            "Predicted_Entities":predicted_entities,
                            "detection_count":1,
                            }
                        # matched_entity_counter+=1
                    else:
                        if predicted_entities in sample_entity_report_dict:
                            sample_entity_report_dict[predicted_entities]["detection_count"] +=1
                            # matched_entity_counter+=1 # valid dupe since a "dupe" is also available in the meme.

                
                # pprint.pprint(sample_entity_report_dict)
                # input()
            
            # now do accounting for how many we got right.
            # pprint.pprint(sample_entity_report_dict)
            prediction_result_dict[singlesample] = sample_entity_report_dict
            # input()
        

        prediction_result_dict["overdupe"] = overdupe_accounting
        with open(targetdumpfile,"w",encoding="utf-8") as dump_file:
            json.dump(prediction_result_dict,dump_file,indent=4)
    print("Skipped Images:")     
    print(skipped_images_list)
    
    print("Done")

        

    


    
    




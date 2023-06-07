import os
import torch
import pprint
import time
import json
import utils
import cProfile
import itertools
import PIL.Image
import random
import sklearn.metrics
import numpy as np
from io import BytesIO    
from PIL import Image
from torchvision.transforms import ToTensor
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation
from attached_heads import trilinear_head, trilinear_head_relations,deeper_trilinear_head_relations
# torch.set_num_threads(2)
# torch.set_num_interop_threads(2)



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
    text_box_positions_used = False
    prefix = "nil"
    wordonly = False
    entity_threshold = 0.5
    data_dir = os.path.join("TDMEMES","TD_Memes")
    annotation_file = os.path.join("TDMEMES","annotation.json")
    split_dump = False
    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']
    
    # 76fv3h.jpg # example of multiple entities in a single statement
    
 
    with open(annotation_file,"r",encoding="utf-8") as annotationfile:
        annotations = json.load(annotationfile)
    print(annotations.keys())
    all_valid_image_files = annotations["SG_Memes"] + annotations["Non_SG_Memes"]
    print("set length:",len(all_valid_image_files))
    class EmptyContext(object): #to ignore nograd if required
        def __init__(self, dummy=None):
            self.dummy = dummy
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    
    targetdumpfile = "TDMEMES_predictions_position_abalated"
    
    full_report = {}
    
    
    
    weightdecay = 1e-6
    entity_embed = BERT_preparation(prefix,device)
    # entity_embed = ROBERTA_preparation(prefix,device)
    entity_embed_head = trilinear_head(device,embed_dimension=768,targetsize = 2)
    entity_embed_head.to(entity_embed_head.device)
    entity_reference_vocab_dict = {i:k for k,i in entity_embed.texttokenizer.get_vocab().items()}
    entity_embed.model.load_state_dict(torch.load("fewshot_ENT0_5e-05_16_abalated.torch"))
    entity_embed.model.eval()
    entity_embed_head.load_state_dict(torch.load("fewshot_ENTHEAD_0_5e-05_16_abalated.torch"))
    entity_embed_head.eval()


    relation_embed = CLIP_preparation(prefix,device)
    relation_embed_head = deeper_trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=not text_box_positions_used)
    relation_internal_parameter = relation_embed_head.parameter # represents meme creator
    relation_embed_head.to(relation_embed_head.device)
    relation_embed.imagemodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_image_abalated.torch"))
    relation_embed.textmodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_text_abalated.torch"))
    relation_embed_head.load_state_dict(torch.load("fewshot_RLNHEAD_0_5e-05_16_abalated.torch"))
    
    texts_dict_organised = {}
    for item in annotations["Text"]:
        texts_dict_organised[list(item.keys())[0]] = item[list(item.keys())[0]]
    
    num_relationship_dict = {
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
    inverted_relationship_dict = {i:k for k,i in num_relationship_dict.items()}
    selected_context = EmptyContext()
    overdupe_accounting = {}
    report_idx=1
    current_count = 0
    for singlesample in all_valid_image_files: # no need to care about idx.
        with selected_context:
            print(current_count,":",singlesample)
            # ENTITY COMPONENT:
            current_count+=1
            # print("-"*30)
            # print("equivalent entities:",singlesample["equivalent_entities"])
            
            
            imagefilepath = os.path.join(data_dir,singlesample)
            if singlesample in full_report:
                continue
            dupedict = {}
            
            if singlesample in annotations["Non_SG_Memes"]:
                meme_type = "Non_SG_Memes"
            else:
                meme_type = "SG_Memes"
                
                
                
            input_text = [texts_dict_organised[singlesample]]    # note that this is a LIST of strings. in this case there is only one string always. So we place into a list.
            print(input_text)

            relations_out_dict = {}
            sample_entity_report_dict = {}
            
            
            
            textembeds,_ = entity_embed(input_text)
            tokenised_list = entity_embed.texttokenizer(input_text)
            # print(tokenised_list)
            matched_entity_counter = 0
            correct_count = 0
            # pprint.pprint(singlesample)
            # print(dupedict)
            # print(input_text)
            for item in range(len(textembeds)):
                prediction = entity_embed_head(textembeds[item]["last_hidden_state"]).squeeze()
                indicators = entity_embed_head.sigmoid(prediction)
                # print("Tokenised OCR box:", tokenised_list["input_ids"][item])
                # print("OCR box:", input_text[item])
                # print("Sigmoided (CUT):",indicators[1:-1])
                # print("Sigmoided UNCUT",indicators)
                predicted_entity_indexes = threshold_entity_extractor(indicators[1:-1],entity_threshold) # predicted version
                # print("predicted entity indexes:",predicted_entity_indexes)
                
                detected_entities = []
                for entityindexset in predicted_entity_indexes: 
                    wordlist = []
                    for wordpart in entityindexset:
                        # print(wordpart)
                        # print(tokenised_list["input_ids"][item])
                        wordlist.append(entity_reference_vocab_dict[tokenised_list["input_ids"][item][1:-1][wordpart]])
                    detected_entities.append(wordlist)
                # print("detected entities:", detected_entities)
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
                # print(input_text[item])
                
                for predicted_entities in predicted_entity_strings:
                    if not predicted_entities.lower() in sample_entity_report_dict:
                        sample_entity_report_dict[predicted_entities.lower()] = {
                            "predicted_sigmoided":indicators.detach().cpu().numpy().tolist(),
                            "predicted_actual":prediction.detach().cpu().numpy().tolist(),
                            "Original_Text":input_text[item],
                            "Predicted_Entities":predicted_entities.lower(),
                            "detection_count":1,
                            "meme_type":meme_type,
                            "textboxposition": False,
                            }
                        # matched_entity_counter+=1
                    else:
                        if predicted_entities.lower() in sample_entity_report_dict:
                            sample_entity_report_dict[predicted_entities.lower()]["detection_count"] +=1
                            # matched_entity_counter+=1 # valid dupe since a "dupe" is also available in the meme.

                
                # pprint.pprint(sample_entity_report_dict)
                # input()
            
            # now do accounting for how many we got right.
            # pprint.pprint(sample_entity_report_dict)
            
            # input()
            ##############################################################################################################
            ##############################################################################################################
            # END OF ENTITY, START OF RELATIONS.
            
            all_entities = []
            for entity_det in sample_entity_report_dict:
                if not text_box_positions_used and relation_embed.word_only:
                    outputs = relation_embed([sample_entity_report_dict[entity_det]["Predicted_Entities"]], imagefilepath)
                    textlist, _ = outputs
                    single_input = {"text":textlist}
                elif text_box_positions_used and relation_embed.word_only:
                    outputs = relation_embed([sample_entity_report_dict[entity_det]["Predicted_Entities"]])
                    textlist, _ = outputs
                    textlist = textlist[0]["pooler_output"]
                    single_input = {"text":textlist,"position":sample_entity_report_dict[entity_det]["textboxposition"]}
                else:
                    outputs = relation_embed([sample_entity_report_dict[entity_det]["Predicted_Entities"]],imagefilepath)
                    textlist, imageembed_outputs = outputs
                    textlist = textlist[0]["pooler_output"]
                    imageembed_outputs = imageembed_outputs["last_hidden_state"]
                    single_input = {"text":textlist,"image":imageembed_outputs.squeeze(),"position":sample_entity_report_dict[entity_det]["textboxposition"]}
                all_entities.append(["Detected_entity",single_input,sample_entity_report_dict[entity_det]["Predicted_Entities"],sample_entity_report_dict[entity_det]["textboxposition"]])
                # all_entities_detected[entity_det]["textboxposition"]
                # all_entities_detected[entity_det]["Predicted_Entities"]
            if not text_box_positions_used and relation_embed.word_only:
                single_input = {"text":relation_embed_head.text_parameter} 
            elif not text_box_positions_used and not relation_embed.word_only:
                single_input = {"text":relation_embed_head.text_parameter, "image":relation_embed_head.image_parameter} 
            else:
                single_input = {"text":relation_embed_head.text_parameter,"image":relation_embed_head.image_parameter,"position":relation_embed_head.position_parameter} 
            all_entities.append(["MEME_CREATOR",single_input,"MEME_CREATOR","MEME_CREATOR"])
                
            if len(all_entities)==1:
                            
                full_report[imagefilepath] = [sample_entity_report_dict,[]]
                continue
            
            all_possible_pairs = list(itertools.permutations(all_entities, 2))
            allinputvectors = {"image1":[],"position1":[],"text1":[], "image2":[],"position2":[],"text2":[]}
            for dual in all_possible_pairs:
                for dictionary_key in dual[0][1]:
                    allinputvectors[dictionary_key+str(1)].append(dual[0][1][dictionary_key])
                for dictionary_key in dual[1][1]:
                    allinputvectors[dictionary_key+str(2)].append(dual[1][1][dictionary_key])
            # pprint.pprint(allinputvectors)
            all_relation_outputs = relation_embed_head(allinputvectors["image1"],allinputvectors["position1"],allinputvectors["text1"], allinputvectors["image2"],allinputvectors["position2"],allinputvectors["text2"])
            all_sigmoided_outputs = relation_embed_head.sigmoid(all_relation_outputs)
            
            
            for pair_idxes in range(len(all_possible_pairs)):
                # print(all_possible_pairs[pair_idxes][0][2], all_possible_pairs[pair_idxes][1][2])
                # print(torch.argmax(all_sigmoided_outputs[pair_idxes]))
                # print(all_relation_outputs[pair_idxes])
                # print(all_sigmoided_outputs.cpu().tolist())
                
                
                
                # if all_possible_pairs[pair_idxes][1][2]=="MEME_CREATOR": # don't append receiver as meme creator stuff... though it is possible we didn't have such cases in our train actually.
                    # continue
                    
                relations_out_dict[pair_idxes] = {
                    "text (Sender)":all_possible_pairs[pair_idxes][0][2],
                    "text (Receiver)":all_possible_pairs[pair_idxes][1][2],
                    "Prediction":int(torch.argmax(all_sigmoided_outputs[pair_idxes].cpu())),
                    "Prediction (Logits)":all_sigmoided_outputs.cpu().tolist()[pair_idxes],
                    "selected_relation":inverted_relationship_dict[torch.argmax(all_sigmoided_outputs[pair_idxes]).cpu().tolist()],
                }
                
                
            full_report[imagefilepath] = [sample_entity_report_dict,relations_out_dict]
            pprint.pprint(full_report[imagefilepath])
            full_report["overdupe"] = overdupe_accounting
            if split_dump:
                if len(list(full_report.keys()))>500:
                    with open(targetdumpfile+str(report_idx)+".json","w",encoding="utf-8") as dump_file:
                        json.dump(full_report,dump_file,indent=4)
                    full_report = {}
                    report_idx+=1
                    print("Dumped: currentidx = ",report_idx)
            
            else:
                if random.randint(0,100)>98:
                    with open(targetdumpfile+".json","w",encoding="utf-8") as dump_file:
                        json.dump(full_report,dump_file,indent=4)
            
    if split_dump:
        with open(targetdumpfile+str(report_idx)+".json","w",encoding="utf-8") as dump_file:
            json.dump(full_report,dump_file,indent=4)
    else:
        with open(targetdumpfile+".json","w",encoding="utf-8") as dump_file:
                    json.dump(full_report,dump_file,indent=4)

    print("Done")

        

    


    
    




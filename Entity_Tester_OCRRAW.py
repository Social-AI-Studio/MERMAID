import os
import torch
import pprint
import time
import json
import utils
import cProfile
import itertools
import PIL.Image
import sklearn.metrics
import numpy as np
from io import BytesIO    
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation
from direct_OCR_result_vs_annotated_comparison import access_target_original_parse_textboxes
from model_dataset_class import text_vector_dataset_extractor ,dual_dataset
from attached_heads import trilinear_head, trilinear_head_relations,deeper_trilinear_head_relations
# torch.set_num_threads(2)
# torch.set_num_interop_threads(2)



def embed_dict_extraction(prep_model,outputs):
    # depending on the prep model, we "extract" the correct shapes we want accordingly into a dict.
    # makes things ever so slightly more readable.
    # mainly for documenting the shapes accordingly, not used in actual training loop
    
    # BLIP
    
    if prep_model.model_name== "BLIP_"+ prefix:
        textlists,imageembed_outputs = outputs
        return_dict = {"last_hidden_states":[], "pooler_outputs":[],"imageembed":imageembed_outputs}
        
        for textoutput in textlists:
        # produces 768 embeds. don't forget that 2 additional special characters are added in tokenisation.
            # print(textoutput.keys())
            # print(textoutput["text_model_output"]["last_hidden_state"].shape) # batchsize, hidden_size # one for each statement.  SEQLEN, 768
            # print(textoutput["text_model_output"]["pooler_output"].shape) # if it's relation part, only the relevant entities should be input    1,768
            return_dict["last_hidden_states"].append(textoutput["text_model_output"]["last_hidden_state"].squeeze())
            return_dict["pooler_outputs"].append(textoutput["text_model_output"]["pooler_output"].squeeze())
            
        # print(imageembed_outputs.shape) # [1, 768] # image is embedded into 768  placed into relation extraction portion.
        return return_dict

    # BERT
    # produces 768 embeds. don't forget that 2 additional special characters are added in tokenisation. though the dataloader accounts for that in the answers.
    elif prep_model.model_name== "BERT_"+ prefix:
        return_dict = {"last_hidden_states":[], "pooler_outputs":[],"imageembed":None}
        for textoutput in outputs:
            # print(textoutput["last_hidden_state"].squeeze().shape) # number of statements, sequence length, 768
            # print(textoutput["pooler_output"].squeeze().shape) # number of statements, 768
            return_dict["last_hidden_states"].append(textoutput["last_hidden_state"].squeeze())
            return_dict["pooler_outputs"].append(textoutput["pooler_output"].squeeze())
        return return_dict

    
    # RoBERTa
    # produces 768 embeds. don't forget that 2 additional special characters are added in tokenisation. though the dataloader accounts for that in the answers.
    elif prep_model.model_name== "RoBERTa_"+ prefix:
        return_dict = {"last_hidden_states":[], "pooler_outputs":[],"imageembed":None}
        for textoutput in outputs:
            # print(textoutput["last_hidden_state"].squeeze().shape) # number of statements, sequence length, 768
            # print(textoutput["pooler_output"].squeeze().shape) # number of statements, 768
            return_dict["last_hidden_states"].append(textoutput["last_hidden_state"].squeeze())
            return_dict["pooler_outputs"].append(textoutput["pooler_output"].squeeze())
        return return_dict
    


    #CLIP 
    # embeds in 768
    elif prep_model.model_name== "CLIP_"+ prefix:
        textlist, imageembed_outputs = outputs
        return_dict = {"last_hidden_states":[], "pooler_outputs":[],"imageembed":None}
        for textoutput in textlist:
            return_dict["last_hidden_states"].append(textoutput["last_hidden_state"].squeeze())
            return_dict["pooler_outputs"].append(textoutput["pooler_output"].squeeze())
            # print(text["last_hidden_state"].shape) #list of:  sequence length, 768. One for EACH item.
            # print(text["pooler_output"].shape) # number of statements, 768
        return_dict["imageembed"] = imageembed_outputs["pooler_output"].squeeze()
        # print(imageembed_outputs["last_hidden_state"].shape) # always 50,768 
        # print(imageembed_outputs["pooler_output"].shape) # always 1, 768
        return return_dict

    #data2vec
    # embeds in 768
    elif prep_model.model_name== "data2vec_"+ prefix:
        textlist, imageembed_outputs = outputs
        return_dict = {"last_hidden_states":[], "pooler_outputs":[],"imageembed":None}
        for textoutput in textlist:
            # print(text["last_hidden_state"].shape) #list of:  sequence length, 768. One for EACH item.
            # print(text["pooler_output"].shape) # number of statements, 768
            return_dict["last_hidden_states"].append(textoutput["last_hidden_state"].squeeze())
            return_dict["pooler_outputs"].append(textoutput["pooler_output"].squeeze())
        # print(imageembed_outputs)
        # print(imageembed_outputs["last_hidden_state"].shape) # always 50,768 
        # print(imageembed_outputs["pooler_output"].shape) # does not exist.
        return_dict["imageembed"] = imageembed_outputs["last_hidden_state"]
        return return_dict

    else:
        raise ValueError("Unknown model type. name missing?",prep_model.model_name)
        

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
    relation_threshold = 0.5
    data_dir = os.path.join("parse_annotated_results","raw_data_jsons","image_dir")
    labels_file = "final_dataset.json"
    
    entity_scheduler_stepsize = 5
    relation_scheduler_stepsize = 3
    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']

    # 76fv3h.jpg # example of multiple entities in a single statement
    
    results_table = []
    total_epochs = 30
    total_timer = time.time()
    

    
    class EmptyContext(object): #to ignore nograd if required
        def __init__(self, dummy=None):
            self.dummy = dummy
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    

    targetdumpfile = "rawOCR_textbox_prediction.json"
    if os.path.exists(targetdumpfile):
        with open(targetdumpfile,"r",encoding="utf-8") as dumpfileopened:
            prediction_result_dict = json.load(dumpfileopened)
        total_correct_entities = prediction_result_dict["correct"]
        total_entities_counter = prediction_result_dict["total"]
    else:
        prediction_result_dict = {}
        total_correct_entities = 0
        total_entities_counter = 0
    
    weightdecay = 1e-6
    entity_embed = BERT_preparation(prefix,device)
    # entity_embed = ROBERTA_preparation(prefix,device)
    entity_embed_head = trilinear_head(device,embed_dimension=768,targetsize = 2)
    entity_embed_head.to(entity_embed_head.device)
    entity_reference_vocab_dict = {i:k for k,i in entity_embed.texttokenizer.get_vocab().items()}
    entity_embed.model.load_state_dict(torch.load("fewshot_0_2e-05_8_innate_epoch_29_all_templates_no_fewshot_position_abalated_BERT_embedder_CLIP_all_templates_no_fewshot_position_abalated_partner.torch"))
    entity_embed.model.eval()
    entity_embed_head.load_state_dict(torch.load("fewshot_0_2e-05_8_innate_epoch_29_all_templates_no_fewshot_position_abalated_BERT_all_templates_no_fewshot_position_abalated_HEAD_CLIP_all_templates_no_fewshot_position_abalated_partner.torch"))
    entity_embed_head.eval()
    
    
    relation_embed = CLIP_preparation(prefix,device)
    relation_embed_head = deeper_trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=not text_box_positions_used)
    relation_internal_parameter = relation_embed_head.parameter # represents meme creator
    relation_embed_head.to(relation_embed_head.device)
    relation_embed.imagemodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_image_abalated.torch"))
    relation_embed.textmodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_text_abalated.torch"))
    relation_embed_head.load_state_dict(torch.load("fewshot_RLNHEAD_0_5e-05_16_abalated.torch"))
    
    with open("OCR_valid_images_list.json","r",encoding="utf-8") as has_entities_available_file:
        all_images = json.load(has_entities_available_file)
    OCR_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,"bert"],approved_images=all_images)
    OCR_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,"bert"])
    
    
    full_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,"bert"])
    
    
    
    OG_parses_dict = access_target_original_parse_textboxes(os.path.join("parse_annotated_results","label_studio_reference_output.json"))
    
    
    # total_possible_model_run_meme_entities = 0
    # total_impossible_model_run_meme_entities = 0
    
    
    # dict_possible_correct_OCR = {}
    # for item in full_dataset.savelist:
        # singleitem_dict = {}
        # itemkey = item["source_image"]
        # for true_textbox in item["correct_answers"]:
            # if "ACTUAL_TEXT" in item["correct_answers"][true_textbox]:
                # singleitem_dict[item["correct_answers"][true_textbox]["ACTUAL_TEXT"][0].lower()] = False
        # proposed_boxes = OG_parses_dict[itemkey]
        # possible_count = 0
        # impossible_count = 0
        # for item in singleitem_dict:
            # if item.lower() in proposed_boxes:
                # singleitem_dict[item] = True
                # possible_count+=1
            # else:
                # impossible_count+=1
        # singleitem_dict["possible_count"] = possible_count
        # singleitem_dict["impossible_count"] = impossible_count
        # total_possible_model_run_meme_entities += possible_count
        # total_impossible_model_run_meme_entities += impossible_count
        # dict_possible_correct_OCR[itemkey] = singleitem_dict
    
    
    
    # ocr_missed_entities = 0
    # for meme in full_dataset.savelist:
        # if not meme["source_image"] in all_images:
            # pure_correct_text_answers = []
            # for correct_ans_key in meme["correct_answers"]:
                # print(meme["correct_answers"][correct_ans_key])
                # if not "ACTUAL_TEXT" in meme["correct_answers"][correct_ans_key]:
                    # continue # textbox does not contain an entity.
                # for pure_entity in meme["correct_answers"][correct_ans_key]["ACTUAL_TEXT"]:
                    # pure_correct_text_answers.append(pure_entity.lower())
            # ocr_missed_entities+=len(pure_correct_text_answers)
    # print("ocr missed entities:",ocr_missed_entities)
    # obtain the proper dataset + it's answers.
    # print("meme dataset length:",len(OCR_meme_dataset))
    
    
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
    
    dataset_dataloader = DataLoader(OCR_meme_dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=OCR_meme_dataset.collate_fn)
    selected_context = EmptyContext()
    full_entity_report = {}
    
    overdupe_accounting = {}
    
    skipped_images_list = []
    with selected_context:
        for _, data_out in enumerate(dataset_dataloader): # no need to care about idx.
            for singlesample in data_out:
                # print("-"*30)
                # print("equivalent entities:",singlesample["equivalent_entities"])
                sample_entity_report_dict = {}
                vectorattachment = text_vector_dataset_extractor(singlesample,device)
                imagesample = os.path.join(data_dir,singlesample["source_image"])
                all_textboxes = []
                if singlesample["source_image"] in prediction_result_dict:
                    continue
                dupedict = {}
                
                print(imagesample)
                # pprint.pprint(singlesample)
                pure_correct_text_answers = []
                for correct_ans_key in singlesample["correct_answers"]:
                    # print(singlesample["correct_answers"][correct_ans_key])
                    # print(singlesample["correct_answers"][correct_ans_key])
                    if not "ACTUAL_TEXT" in singlesample["correct_answers"][correct_ans_key]:
                        continue # textbox does not contain an entity.
                    for pure_entity in singlesample["correct_answers"][correct_ans_key]["ACTUAL_TEXT"]:
                        pure_correct_text_answers.append(pure_entity.lower())
                
                correct_ans_tokenised = entity_embed.texttokenizer(pure_correct_text_answers)
                tokenised_correct_ans_dict = {}
                for pure_idx in range(len(pure_correct_text_answers)):
                    wordlist = []
                    for wordpart in correct_ans_tokenised["input_ids"][pure_idx][1:-1]:
                        # print(correct_ans_tokenised["input_ids"][pure_idx][1:-1])
                        wordlist.append(entity_reference_vocab_dict[wordpart])
                    finalstring = ""
                    for stringpart in wordlist:
                        if "##" in stringpart:
                            finalstring+=stringpart[2:] # instant attachment.
                        else:
                            finalstring+=" "
                            finalstring+=stringpart
                    finalstring = finalstring.strip()
                    if finalstring in tokenised_correct_ans_dict:
                        if not finalstring in dupedict:
                            dupedict[finalstring] = 1
                        dupedict[finalstring]+=1
                    else:
                        tokenised_correct_ans_dict[finalstring] = pure_correct_text_answers[pure_idx]
                
                pprint.pprint(tokenised_correct_ans_dict)
                
                # print(vectorattachment[textname]["text"]) # still need to edit input to take something else instead...
                # textembeds,_ = entity_embed([vectorattachment[textname]["text"]])
                OCR_identified_boxes = OG_parses_dict[singlesample["source_image"]]
                textembeds,_ = entity_embed(OCR_identified_boxes)
                tokenised_list = entity_embed.texttokenizer(OCR_identified_boxes)
                # print(tokenised_list)
                matched_entity_counter = 0
                correct_count = 0
                # pprint.pprint(singlesample)
                # print(dupedict)
                # print(len(textembeds))
                # print(OCR_identified_boxes)
                for item in range(len(textembeds)):
                    prediction = entity_embed_head(textembeds[item]["last_hidden_state"]).squeeze()
                    indicators = entity_embed_head.sigmoid(prediction)
                    # print("Tokenised OCR box:", tokenised_list["input_ids"][item])
                    # print("OCR box:", OCR_identified_boxes[item])
                    # print("Sigmoided (CUT):",indicators[1:-1])
                    # print("Sigmoided UNCUT",indicators)
                    predicted_entity_indexes = threshold_entity_extractor(indicators[1:-1],entity_threshold) # predicted version
                    # print("predicted entity indexes:",predicted_entity_indexes)
                    
                    detected_entities = []
                    
                    for entityindexset in predicted_entity_indexes: 
                        wordlist = []
                        for wordpart in entityindexset:
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
                    # print(OCR_identified_boxes[item])
                    
                    for predicted_entities in predicted_entity_strings:
                        for pure_entities in tokenised_correct_ans_dict:
                            if predicted_entities.lower()==pure_entities.lower():
                                if not predicted_entities.lower() in sample_entity_report_dict:
                                    sample_entity_report_dict[predicted_entities.lower()] = {
                                        "predicted_sigmoided":indicators.detach().cpu().numpy().tolist(),
                                        "predicted_actual":prediction.detach().cpu().numpy().tolist(),
                                        "OCR_extracted_text":OCR_identified_boxes[item],
                                        "Entity_extracted_text":pure_entities,
                                        "match":predicted_entities,
                                        "detection_count":1,
                                        "tokenised_version":predicted_entities.lower(),
                                        "actual_text_before_tokenise":tokenised_correct_ans_dict[predicted_entities.lower()],
                                        }
                                    # matched_entity_counter+=1
                                    correct_count+=1
                                else:
                                    if predicted_entities.lower() in dupedict:
                                        if sample_entity_report_dict[predicted_entities.lower()]["detection_count"]>  dupedict[predicted_entities.lower()]:
                                            if not singlesample["source_image"] in overdupe_accounting:
                                                overdupe_accounting[singlesample["source_image"]] = {}
                                            if not predicted_entities.lower() in overdupe_accounting[singlesample["source_image"]]:
                                                overdupe_accounting[singlesample["source_image"]][predicted_entities.lower()] = 0
                                            overdupe_accounting[singlesample["source_image"]][predicted_entities.lower()] +=1
                                        else:
                                            sample_entity_report_dict[predicted_entities.lower()]["detection_count"] +=1
                                            # matched_entity_counter+=1 # valid dupe since a "dupe" is also available in the meme.
                                            correct_count+=1 # valid dupe since a "dupe" is also available in the meme.
                                    else:
                                        if not singlesample["source_image"] in overdupe_accounting:
                                            overdupe_accounting[singlesample["source_image"]] = {}
                                            if not predicted_entities.lower() in overdupe_accounting[singlesample["source_image"]]:
                                                overdupe_accounting[singlesample["source_image"]][predicted_entities.lower()] = 0
                                            overdupe_accounting[singlesample["source_image"]][predicted_entities.lower()] +=1

                    
                    # pprint.pprint(sample_entity_report_dict)
                    # input()
                
                # now do accounting for how many we got right.
                total_number_of_entities = len(pure_correct_text_answers)
                if singlesample["source_image"] in overdupe_accounting:
                    pass
                total_correct_entities+=correct_count
                total_entities_counter+=total_number_of_entities
                # sample_entity_report_dict["possible_count"] = dict_possible_correct_OCR[singlesample["source_image"]]["possible_count"]
                # sample_entity_report_dict["impossible_count"] = dict_possible_correct_OCR[singlesample["source_image"]]["impossible_count"]
                # pprint.pprint(sample_entity_report_dict)
                prediction_result_dict[singlesample["source_image"]] = sample_entity_report_dict
                print("Total Correct entities:",total_correct_entities)
                print("Total Detection Entities:",total_entities_counter)
                # input()
            
            prediction_result_dict["correct"] = total_correct_entities
            prediction_result_dict["total"] = total_entities_counter
            prediction_result_dict["overdupe"] = overdupe_accounting
            with open(targetdumpfile,"w",encoding="utf-8") as dump_file:
                json.dump(prediction_result_dict,dump_file,indent=4)
    print("Skipped Images:")     
    print(skipped_images_list)
    
    print("Total Correct entities:",total_correct_entities)
    print("Total Incorrect/Missing Entities:",total_entities_counter)
    print("Done")

        

    


    
    




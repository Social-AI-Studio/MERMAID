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
from torch.utils.data import Dataset, DataLoader
from io import BytesIO    
from PIL import Image
from torchvision.transforms import ToTensor
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation
from attached_heads import trilinear_head,deeper_trilinear_head_relations
from model_dataset_class import text_vector_dataset_extractor, entity_error_analysis,dual_dataset,dataset_list_split
from direct_OCR_result_vs_annotated_comparison import access_target_original_parse_textboxes
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
    prefix = "nil"    
    wordonly = False # whether to ignore imageinputs
    noposition = False
    text_box_positions_used = not noposition # whether we used textbox positions. inferred from traditional noposition variable.

    entity_threshold = 0.5
    relation_threshold = 0.5
    data_dir = os.path.join("parse","raw_data_jsons","image_dir")
    labels_file = "final_dataset_cleared.json"
    
    entity_scheduler_stepsize = 5
    relation_scheduler_stepsize = 3
    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']

    # 76fv3h.jpg # example of multiple entities in a single statement
    
    results_table = []
    total_epochs = 30
    total_timer = time.time()
    
    targetdumpfile = "MERMAID_OCR_TEST_ONLY.json"
    
    class EmptyContext(object): #to ignore nograd if required
        def __init__(self, dummy=None):
            self.dummy = dummy
        def __enter__(self):
            return None
        def __exit__(self, *args):
            pass
    

    
    if os.path.exists(targetdumpfile):
        with open(targetdumpfile,"r",encoding="utf-8") as dumpfileopened:
            full_report = json.load(dumpfileopened)
        total_correct_entities = full_report["correctentity"]
        total_entities_counter = full_report["totalentity"]
        correctrelationcount = full_report["correctrelationcount"]
        totalrelationcount = full_report["totalrelationcount"]
        overdupe_accounting = full_report["overdupe"]
    else:
        full_report = {}
        total_correct_entities = 0
        total_entities_counter = 0
        correctrelationcount = 0
        totalrelationcount = 0
        overdupe_accounting = {}
    
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
    relation_embed_head = deeper_trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=noposition)
    relation_internal_parameter = relation_embed_head.parameter # represents meme creator
    relation_embed_head.to(relation_embed_head.device)
    relation_embed.imagemodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_image_abalated.torch"))
    relation_embed.textmodel.load_state_dict(torch.load("fewshot_RLN_0_5e-05_16_text_abalated.torch"))
    relation_embed_head.load_state_dict(torch.load("fewshot_RLNHEAD_0_5e-05_16_abalated.torch"))
    

    original_train_imglist,original_test_imglist  = dataset_list_split(labels_file,0.6,[])
    full_test_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type],approved_images=original_test_imglist)
    # full_test_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type])
    print("test numbers:",len(full_test_dataset))
    OG_parses_dict = access_target_original_parse_textboxes(os.path.join("parse","label_studio_reference_input_OCR_INITIAL.json"))
    
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
    inverted_relationship_dict = {i:k for k,i in remapped_actionables_dict.items()}
    dataset_dataloader = DataLoader(full_test_dataset, batch_size=32, shuffle=True, num_workers=0,collate_fn=full_test_dataset.collate_fn)
    selected_context = EmptyContext()
    
    skipped_images_list = []
    with selected_context:
        for _, data_out in enumerate(dataset_dataloader): # no need to care about idx.
            for singlesample in data_out:
                # print("-"*30)
                # print("equivalent entities:",singlesample["equivalent_entities"])
                sample_entity_report_dict = {}
                vectorattachment = text_vector_dataset_extractor(singlesample,device)
                imagefilepath = os.path.join(data_dir,singlesample["source_image"])
                all_textboxes = []
                if singlesample["source_image"] in full_report:
                    continue
                dupedict = {}
                meme_type = singlesample["archetype"]
                print(imagefilepath)
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
                    finalstring = finalstring.strip().replace(" ' ","'")
                    
                    
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
                    # print(OCR_identified_boxes[item])
                    
                    for predicted_entities in predicted_entity_strings:
                        for pure_entities in tokenised_correct_ans_dict:
                            if predicted_entities.lower()==pure_entities.lower():
                                if not predicted_entities.lower() in sample_entity_report_dict:
                                    sample_entity_report_dict[predicted_entities.lower()] = {
                                        "predicted_sigmoided":indicators.detach().cpu().numpy().tolist(),
                                        "predicted_actual":prediction.detach().cpu().numpy().tolist(),
                                        "OCR_extracted_text":OCR_identified_boxes[item],
                                        "detection_count":1,
                                        "meme_type":meme_type,
                                        "textboxposition": text_box_positions_used,
                                        "Predicted_Entities":predicted_entities.lower(),
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
                print("Total Correct entities:",total_correct_entities)
                print("Total Detection Entities:",total_entities_counter)
                # input()
            
                full_report["correctentity"] = total_correct_entities
                full_report["totalentity"] = total_entities_counter
                full_report["overdupe"] = overdupe_accounting
                
                
                # input()
                # print(entity_prediction_result_dict)
                print("Predicted entities dict:")
                pprint.pprint(sample_entity_report_dict)
                correct_answers = {}
                correct_entity_set = set()
                has_meme_creator = False
                for entity_answer in singlesample["actual_entities"]:
                    if entity_answer[2]=="MEME_CREATOR":
                        has_meme_creator = True
                        correct_answers[entity_answer[2]] = entity_answer[0]
                        correct_entity_set.add(entity_answer[2])
                    else:
                        correct_answers[entity_answer[2].lower()] = entity_answer[0]
                        correct_entity_set.add(entity_answer[2].lower())
                
                relationshipans_dict = singlesample["relationship_num"]
                
                inverted_relationshipans_dict = {}
                for key in relationshipans_dict:
                    for pair_entity in relationshipans_dict[key]:
                        inverted_relationshipans_dict[tuple(pair_entity)] = key
                
                ##############################################################################################################
                ##############################################################################################################
                # END OF ENTITY, START OF RELATIONS.
                
                print("Correct Answers:",correct_answers)
                print("Correct Entity set:",correct_entity_set)
                
                all_entities = []
                for entity_det in sample_entity_report_dict:
                    if not sample_entity_report_dict[entity_det]["Predicted_Entities"].lower() in correct_entity_set: #ignore instances where the entity isn't actually "CORRECT"
                        continue
                    print("Successful Entity extraction above:",sample_entity_report_dict[entity_det]["Predicted_Entities"].lower())
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
                
                if has_meme_creator and sample_entity_report_dict: # check to ensure that there were actual detected entities.
                    all_entities.append(["MEME_CREATOR",single_input,"MEME_CREATOR","MEME_CREATOR"])
                
                
                
                print("Number of entities in relations consideration:",len(all_entities))
                
                
                if not sample_entity_report_dict or len(list(all_entities))<=1: # we can't do pairs on a single term!
                    for _ in inverted_relationshipans_dict:
                        totalrelationcount+=1
                    print("Failed to detect any entities in ",singlesample["source_image"])
                    full_report[singlesample["source_image"]] = [sample_entity_report_dict,[]]
                else:
                    
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
                    
                    relation_prediction_results = []
                    
                    for pair_idxes in range(len(all_possible_pairs)):
                        # print(all_possible_pairs[pair_idxes][0][2], all_possible_pairs[pair_idxes][1][2])
                        # print(torch.argmax(all_sigmoided_outputs[pair_idxes]))
                        # print(all_relation_outputs[pair_idxes])
                        # print(all_sigmoided_outputs.cpu().tolist())
                        
                        
                        
                        # if all_possible_pairs[pair_idxes][1][2]=="MEME_CREATOR": # don't append receiver as meme creator stuff... though it is possible we didn't have such cases in our train actually.
                            # continue
                        print("target pair:",(correct_answers[all_possible_pairs[pair_idxes][0][2]],correct_answers[all_possible_pairs[pair_idxes][1][2]]))
                        print("inverted relationship ans dict:")
                        pprint.pprint(inverted_relationshipans_dict)
                        selected_relation_idx = int(torch.argmax(all_sigmoided_outputs[pair_idxes].cpu()))
                        print("selected relation idx:",selected_relation_idx)
                        if (correct_answers[all_possible_pairs[pair_idxes][0][2]],correct_answers[all_possible_pairs[pair_idxes][1][2]]) in inverted_relationshipans_dict:
                            correct_relation_idx = inverted_relationshipans_dict[(correct_answers[all_possible_pairs[pair_idxes][0][2]],correct_answers[all_possible_pairs[pair_idxes][1][2]])]
                        else:
                            correct_relation_idx = 8
                        
                        if correct_relation_idx == selected_relation_idx:
                            correctrelationcount+=1
                            
                        relation_prediction_results.append({
                            "text (Sender)":all_possible_pairs[pair_idxes][0][2],
                            "text (Receiver)":all_possible_pairs[pair_idxes][1][2],
                            "Prediction":int(torch.argmax(all_sigmoided_outputs[pair_idxes].cpu())),
                            "Prediction (Logits)":all_sigmoided_outputs.cpu().tolist()[pair_idxes],
                            "selected_relation":inverted_relationship_dict[selected_relation_idx],
                            "correct_relation":inverted_relationship_dict[correct_relation_idx],
                        })
                        
                        pprint.pprint(relation_prediction_results)
                        
                        
                    full_report[singlesample["source_image"]] = [sample_entity_report_dict,relation_prediction_results]
                    pprint.pprint(full_report[singlesample["source_image"]])
                    
                
            full_report["overdupe"] = overdupe_accounting
            full_report["correctentity"] = total_correct_entities
            full_report["totalentity"] = total_entities_counter
            full_report["correctrelationcount"] = correctrelationcount
            full_report["totalrelationcount"] = totalrelationcount
            
            with open(targetdumpfile,"w",encoding="utf-8") as dump_file:
                json.dump(full_report,dump_file,indent=4)
        
        # quit()
    print("Skipped Images:")     
    print(skipped_images_list)
    
    print("Done")

        

    


    
    




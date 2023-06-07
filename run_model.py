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
from zipfile import ZipFile
from io import BytesIO    
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation, GPT_NEO_preparation


# dataset class
from model_dataset_class import text_vector_dataset_extractor,dual_dataset,entity_error_analysis,dataset_list_split
# trilinear
from attached_heads import trilinear_head ,deeper_trilinear_head_relations
torch.set_num_threads(2)
torch.set_num_interop_threads(2)


        

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
    prefix = "normal_pos_"
    
    
    
    entity_threshold = 0.5
    relation_threshold = 0.5
    data_dir = os.path.join("parse","raw_data_jsons","image_dir")
    labels_file = "final_dataset_cleared.json"

    noposition = False
    guidance = True
    deeper_head = True
    
    test_batchsize = 56
    fewshot_internalepochs = 30
    fewshot_targetks = [0,5,10,20]
    
    optimizerlrlist = [1e-5,2e-5,5e-5]
    batch_sizelist = [4,8,16,32]
    
    

    entity_scheduler_stepsize = 5
    relation_scheduler_stepsize = 3
    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']
    target_fewshot_templates = ['Cuphead-Flower','Soyboy-Vs-Yes-Chad','Running-Away-Balloon','Mother-Ignoring-Kid-Drowning-In-A-Pool','Hide-the-Pain-Harold']
    non_involved_templates = []
    
    if target_fewshot_templates:
        additional_name_prefix = "fewshot_"
    else:
        additional_name_prefix = "regular"
    
    for item in all_archetypes:
        if item in target_fewshot_templates:
            continue
        non_involved_templates.append(item)
    
    if not target_fewshot_templates: # we are not performing fewshot.
        fewshot_targetks = [0] # force set to only run once.
    
    print("Target fewshot:")
    print(target_fewshot_templates)
    print("Non_involved Templates set:")
    print(non_involved_templates)
    
    
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
    

    

    
    for optimizerlr in optimizerlrlist:
        for batch_size in batch_sizelist:
            for target_n in fewshot_targetks:
                weightdecay = 1e-6
                entity_embed = BERT_preparation(prefix,device)
                # entity_embed = GPT_NEO_preparation(prefix,device)
                # entity_embed = ROBERTA_preparation(prefix,device)
                entity_embed_head = trilinear_head(device,embed_dimension=entity_embed.embedsize,targetsize = 2)
                entity_embed_head.to(entity_embed_head.device)
                entityheadoptimizer = torch.optim.Adam(entity_embed.optimizer_parameters+list(entity_embed_head.parameters()),lr=optimizerlr,weight_decay = weightdecay)
                entity_reference_vocab_dict = {i:k for k,i in entity_embed.texttokenizer.get_vocab().items()}


                
                # calculated by proportion of total number of that relationship.
                
                binary_sample_loss = torch.nn.BCEWithLogitsLoss() # we should use this in reality.
                # cross_entropy_loss = torch.nn.CrossEntropyLoss()
                
                relation_embed = CLIP_preparation(prefix,device)
                # relation_embed = data2vec_preparation(prefix,device)
                # relation_embed = FLAVA_preparation(prefix,device)
                # relation_embed = BLIP_preparation(prefix,device)
                
                
                wordonly = relation_embed.word_only
                relation_embed_head = deeper_trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=noposition)
   
                relation_internal_parameter = relation_embed_head.parameter # represents meme creator
                relation_embed_head.to(relation_embed_head.device)
                relationheadoptimizer = torch.optim.Adam(list(relation_embed.optimizer_parameters)+list(relation_embed_head.parameters()),lr=optimizerlr,weight_decay = weightdecay)


                
                # fewshottest_constructor = [i for i in list(standard_imagesize_dict.keys()) if not i in leaveouts]
                
                relationscheduler = StepLR(relationheadoptimizer, step_size=relation_scheduler_stepsize, gamma=0.5) # every x epochs, We REDUCE learning rate.
                entityscheduler = StepLR(entityheadoptimizer, step_size=entity_scheduler_stepsize, gamma=0.5) # every x epochs, We REDUCE learning rate.
                
                original_train_imglist,original_test_imglist  = dataset_list_split(labels_file,0.6,[])
                
            
                print("Current target n:", target_n, "   RELOADING WEIGHTS")
                
                
                if target_n==0:
                    if non_involved_templates:
                        fewshot_removal_list1, fewshot_removal_list2 = dataset_list_split(labels_file,0.6,non_involved_templates,minimal_n=target_n,approved_images=original_train_imglist)
                        for removed_instance in fewshot_removal_list1:
                            original_train_imglist.remove(removed_instance)
                        for removed_instance in fewshot_removal_list2:
                            original_train_imglist.remove(removed_instance)
                    else:
                        pass
                else:
                    fewshot_train_img_candidates, fewshot_removal_list  = dataset_list_split(labels_file,0.6,non_involved_templates,minimal_n=target_n,approved_images=original_train_imglist)
                    for removed_instance in fewshot_removal_list:
                        original_train_imglist.remove(removed_instance)
                
                
                
                # fewshottrain_img_list,fewshottest_img_list  = dataset_list_split("ZZZZ_completed_processed_fullabels_shaun.json",1,non_involved_templates,minimal_n=target_n,approved_images=original_train_imglist)
                
                # print("Fewshot train image list:",len(fewshottrain_img_list))
                # print("Fewshot test image list:",len(fewshottest_img_list))
                fewshottrain_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type],approved_images=original_train_imglist)
                fewshottest_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type],approved_images=original_test_imglist)
                print("fewshot train length:",len(fewshottrain_meme_dataset),"fewshot test length:",len(fewshottest_meme_dataset))
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
                
                
                itemcount = {}
                maxseen = 0
                classwise_weights = []
                for rlnname in fewshottrain_meme_dataset.relationships_counter:
                    matched_number = remapped_actionables_dict[rlnname]
                    itemcount[matched_number] = fewshottrain_meme_dataset.relationships_counter[rlnname]
                    itemcount[matched_number] += fewshottest_meme_dataset.relationships_counter[rlnname]
                    if itemcount[matched_number]>maxseen:
                        maxseen = itemcount[matched_number]
                    classwise_weights.append(0)
                
                for rlnindex in itemcount:
                    classwise_weights[rlnindex] = maxseen/itemcount[rlnindex]
                    
                    
                print("Classwise Weights:",classwise_weights)
                relational_binary_sample_loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(classwise_weights).to(device)) # we should use this in reality.

                saved_weightsfilenames = []
                saved_resultsfilenames = []
                
                fewshottrain_meme_dataloader = DataLoader(fewshottrain_meme_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=fewshottest_meme_dataset.collate_fn)
                fewshottest_meme_dataloader = DataLoader(fewshottest_meme_dataset, batch_size=test_batchsize, shuffle=True, num_workers=0,collate_fn=fewshottest_meme_dataset.collate_fn)
            
                
                for innate_epoch in range(fewshot_internalepochs):
                    
                    
                    fewshottotal_train_loss = 0
                    fewshottotal_test_loss = 0
                    fewshotstart_epoch_time = time.time()
                    fewshotrelation_classwise_train = {"Superior":0,"Equal":1,"Upgrade":2,"Degrade":3,"Affirm/Favor":4,"Doubt/Disfavor":5,"Indifferent":6,"Inferior":7,"NULL":8,0:[0,0],1:[0,0],2:[0,0],3:[0,0],4:[0,0],5:[0,0],6:[0,0],7:[0,0],8:[0,0]}
                    fewshotrelation_classwise_test = {"Superior":0,"Equal":1,"Upgrade":2,"Degrade":3,"Affirm/Favor":4,"Doubt/Disfavor":5,"Indifferent":6,"Inferior":7,"NULL":8,0:[0,0],1:[0,0],2:[0,0],3:[0,0],4:[0,0],5:[0,0],6:[0,0],7:[0,0],8:[0,0]}
                    fewshotentity_recorder_train = {"total_entities":0, "correct_entities_startstops":0, "pure_correct_entities":0}
                    fewshotentity_recorder_test = {"total_entities":0, "correct_entities_startstops":0, "pure_correct_entities":0}
                    fewshottrain_predcomparedict = {}
                    fewshottest_predcomparedict = {}
                    for selected_dataloader,istrain,interaction_record,prediction_dict,entity_correctness_record in [[fewshottrain_meme_dataloader,True,fewshotrelation_classwise_train,fewshottrain_predcomparedict,fewshotentity_recorder_train],[fewshottest_meme_dataloader,False,fewshotrelation_classwise_test,fewshottest_predcomparedict,fewshotentity_recorder_test]]:
                        
                        if istrain:
                            for part in [entity_embed, entity_embed_head, relation_embed, relation_embed_head]:
                                part.train()
                            selected_context = EmptyContext()
                        else:
                            for part in [entity_embed, entity_embed_head, relation_embed, relation_embed_head]:
                                part.eval()
                            selected_context = torch.no_grad()
                        
                        # note that the parameter which acts as embedding for the MEME CREATOR doesn't need to have .train() called. it's bundled in the relation embed head.

                        
                        with selected_context:
                            for _, data_out in enumerate(selected_dataloader): # no need to care about idx.
                                total_loss = 0
                                entity_correctness_counter = 0
                                total_entities = 0
                                entity_pure_correctness_counter = 0
                                # print(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated())
                                batchtimer = time.time()
                                all_input_format = []
                                for singlesample in data_out:
                                    # print("-"*30)
                                    fewshotrelation_loss_sum = 0
                                    fewshotentity_loss_sum = 0
                                    # print("equivalent entities:",singlesample["equivalent_entities"])
                                    entity_source_pair = []
                                    sample_entity_report_dict = {}
                                    vectorattachment = text_vector_dataset_extractor(singlesample,device)
                                    imagesample = os.path.join(data_dir,singlesample["source_image"])
                                    for textname in vectorattachment:
                                        # print(vectorattachment[textname]["text"])
                                        textembeds,_ = entity_embed([vectorattachment[textname]["text"]])
                                        vectorattachment[textname][entity_embed.model_name] = textembeds[0]
                                        vectorattachment[textname]["entity_head_output"] = entity_embed_head(vectorattachment[textname][entity_embed.model_name]["last_hidden_state"]).squeeze()
                                        # vectorattachment[textname]["entity_head_output"] = torch.ones([len(singlesample["correct_answers"][textname][entity_embed.model_type]),2]).to(device)
                                        
                                        # wordlist = []
                                        # for entitypart in _["input_ids"][0]:
                                            # wordlist.append(entity_reference_vocab_dict[int(entitypart)])
                                        # print(wordlist)
                                        # print("0"*30)
                                        
                                        # print(len(vectorattachment[textname]["entity_head_output"]), len(singlesample["correct_answers"][textname][entity_embed.model_type]),len(singlesample["span_answer"][textname][entity_embed.model_type]))
                                        # print(entity_embed.model_type)
                                        if entity_embed.model_type=="gpt_neo_SINGLISH":
                                            # print(len(vectorattachment[textname]["entity_head_output"]))
                                            # print(vectorattachment[textname]["entity_head_output"].shape)
                                            if len(vectorattachment[textname]["entity_head_output"].shape)==1:
                                                vectorattachment[textname]["entity_head_output"] = vectorattachment[textname]["entity_head_output"].unsqueeze(0)
                                            # print(len(singlesample["correct_answers"][textname][entity_embed.model_type][1:-1]))
                                            entity_report, entity_loss = entity_error_analysis(vectorattachment[textname]["entity_head_output"],singlesample["correct_answers"][textname][entity_embed.model_type][1:-1],singlesample["span_answer"][textname][entity_embed.model_type][1:-1],entity_threshold,device,loss_entity=binary_sample_loss)
                                        else:
                                            entity_report, entity_loss = entity_error_analysis(vectorattachment[textname]["entity_head_output"],singlesample["correct_answers"][textname][entity_embed.model_type],singlesample["span_answer"][textname][entity_embed.model_type],entity_threshold,device,loss_entity=binary_sample_loss)
                                        fewshotentity_loss_sum += entity_loss
                                        for entity_item in entity_report:
                                            total_entities+=1
                                            if entity_report[entity_item]["start"] and entity_report[entity_item]["stop"]: # correct entity counter.
                                                entity_correctness_counter += 1
                                            if entity_report[entity_item]["start"] and entity_report[entity_item]["stop"] and not entity_report[entity_item]["midfailure"]: # correct entity counter.
                                                entity_pure_correctness_counter +=1
                                        # continue
                                        
                                        indicators = entity_embed_head.sigmoid(vectorattachment[textname]["entity_head_output"])
                                        # print(indicators)
                                        
                                        sample_entity_report_dict[textname] = {"entity_report":entity_report,
                                                "loss":entity_loss.item(),
                                                "predicted_sigmoided":indicators.tolist(),
                                                "predicted_actual":vectorattachment[textname]["entity_head_output"].tolist(),
                                                "actual":singlesample["span_answer"][textname][entity_embed.model_type],
                                                "text":vectorattachment[textname]["text"]}
                                        
                                        
                                        # print("predicted version")
                                        predicted_entity_indexes = threshold_entity_extractor(vectorattachment[textname]["entity_head_output"][1:-1],entity_threshold) # predicted version
                                        # print("actual version")
                                        # print(singlesample["correct_answers"][textname][entity_embed.model_type][1:-1])
                                        correct_entity_indexes = threshold_entity_extractor(singlesample["correct_answers"][textname][entity_embed.model_type][1:-1],entity_threshold) # true version
                                        # print(correct_entity_indexes)
                                        
                                        # print(singlesample["correct_answers"][textname][entity_embed.model_type])
                                        detected_entities = []
                                        # print(singlesample["tokenised_strings"][textname]["input_text"])
                                        for entityindexset in correct_entity_indexes:
                                            wordlist = []
                                            for entitypart in entityindexset:
                                                wordlist.append(entity_reference_vocab_dict[singlesample["tokenised_strings"][textname][entity_embed.model_type][entitypart]])
                                            detected_entities.append(wordlist)
                                        # print(detected_entities)
                                        corrected_entity_string = []
                                        for entity in detected_entities:
                                            finalstring = ""
                                            for stringpart in entity:
                                                if "##" in stringpart:
                                                    finalstring+=stringpart[2:] # instant attachment.
                                                else:
                                                    finalstring+=" "
                                                    finalstring+=stringpart
                                            corrected_entity_string.append(finalstring)
                                        # print(corrected_entity_string)
                                        entity_source_pair.append([corrected_entity_string,textname])
                                    # print(vectorattachment)                    
                                    # print(vectorattachment[textname][entity_embed.model_name]["last_hidden_state"].shape)
                                    # print(vectorattachment[textname]["entity_head_output"].shape)
                                        
                                    all_input_format.append(vectorattachment)
                                    # continue
                                    
                                    all_entities = []
                                    # pprint.pprint(singlesample)
                                    if guidance:
                                        for sourcetrio in singlesample["actual_entities"]:   # id, which box, actual text # use id to track what the correct relationship is.
                                            targetid, originbox, entity_text = sourcetrio
                                            # print(entity_text)
                                            if originbox==None: # is MEME_CREATOR
                                                if noposition:
                                                    single_input = {"text":relation_embed_head.text_parameter,"image":relation_embed_head.image_parameter}
                                                else:
                                                    single_input = {"text":relation_embed_head.text_parameter,"image":relation_embed_head.image_parameter,"position":relation_embed_head.position_parameter}
                                                # single_input = {"text":relation_embed_head.text_parameter,"image":relation_embed_head.image_parameter,"position":relation_embed_head.position_parameter}
                                                all_entities.append(["MEME_CREATOR",single_input,"MEME_CREATOR","MEME_CREATOR"])
                                                continue
                                                
                                            elif relation_embed.word_only and not noposition:
                                                outputs = relation_embed([entity_text])
                                                textlist, _ = outputs
                                                # print(textlist.shape) 
                                                # print(imageembed_outputs.shape)
                                                textlist = textlist[0]["pooler_output"]
                                                positional_vector = singlesample["text_locs"][originbox][0]["vector"]
                                                single_input = {"text":textlist,"position":positional_vector}
                                                
                                            elif relation_embed.word_only and noposition:
                                                outputs = relation_embed([entity_text])
                                                textlist, _ = outputs
                                                textlist = textlist[0]["pooler_output"]
                                                # print(textlist.shape) 
                                                # print(imageembed_outputs.shape)
                                                positional_vector = singlesample["text_locs"][originbox][0]["vector"]
                                                single_input = {"text":textlist}
                                            
                                            else: 
                                                outputs = relation_embed([entity_text], os.path.join(data_dir, singlesample["source_image"]))
                                                textlist, imageembed_outputs = outputs
                                                # print(imageembed_outputs)
                                                # print(textlist)
                                                textlist = textlist[0]["pooler_output"]
                                                imageembed_outputs = imageembed_outputs["last_hidden_state"]
                                                positional_vector = singlesample["text_locs"][originbox][0]["vector"]
                                                single_input = {"text":textlist,"image":imageembed_outputs.squeeze(),"position":positional_vector}

                                                # print(textlist.shape) 
                                                # print(imageembed_outputs.shape)
                                            
                                            all_entities.append([targetid,single_input,entity_text,originbox])
                                            
                                    else:
                                        for sourcepair in entity_source_pair:
                                            entity_text, originbox = sourcepair
                                            if originbox==None: # is MEME_CREATOR
                                                single_input = {"text":relation_embed_head.text_parameter,"image":relation_embed_head.image_parameter,"position":relation_embed_head.position_parameter}
                                                all_entities.append(["MEME_CREATOR",single_input,"MEME_CREATOR","MEME_CREATOR"])
                                                continue
                                            
                                            
                                            if relation_embed.word_only:
                                                outputs = relation_embed(entity_text)
                                            else:
                                                outputs = relation_embed(entity_text, os.path.join(data_dir, singlesample["source_image"]))
                                            textlist, imageembed_outputs = outputs
                                            textlist = textlist[0]["pooler_output"]
                                            imageembed_outputs = imageembed_outputs["last_hidden_state"]
                                            positional_vector = singlesample["text_locs"][originbox][0]["vector"]
                                            # print(textlist.shape) 
                                            # print(imageembed_outputs.shape)
                                            
                                            if noposition and relation_embed.word_only:
                                                single_input = {"text":textlist}
                                            elif not noposition and relation_embed.word_only:
                                                single_input = {"text":textlist,"position":positional_vector}
                                            else:
                                                single_input = {"text":textlist,"image":imageembed_outputs.squeeze(),"position":positional_vector}
                                            
                                            all_entities.append(["NO_ID",single_input,entity_text,originbox])
                                        # print(relation_internal_parameter.shape)
                                        # print(single_input.shape)
                                        
                                    
                                    all_possible_pairs = list(itertools.permutations(all_entities, 2))
                                    # print(all_possible_pairs)
                                    
                                    
                                    # print("entity list:",all_entities)
                                    # print("inverted_numerical_relationships:",singlesample["inverted_numerical_relationships"])
                                    # print("all possible pairs:",all_possible_pairs)
                                    allinputvectors = {"image1":[],"position1":[],"text1":[], "image2":[],"position2":[],"text2":[]}
                                    for dual in all_possible_pairs:
                                        for dictionary_key in dual[0][1]:
                                            allinputvectors[dictionary_key+str(1)].append(dual[0][1][dictionary_key])
                                        for dictionary_key in dual[1][1]:
                                            allinputvectors[dictionary_key+str(2)].append(dual[1][1][dictionary_key])
                                    # pprint.pprint(allinputvectors)
                                    all_relation_outputs = relation_embed_head(allinputvectors["image1"],allinputvectors["position1"],allinputvectors["text1"], allinputvectors["image2"],allinputvectors["position2"],allinputvectors["text2"])
                                    relation_prediction_result_dict = {}
                                    for pair_idxes in range(len(all_possible_pairs)):
                                        
                                        # print("checked a pair...")
                                        # print(pair_output) # [targetid,single_input,entity_text,originbox]
                                        
                                        pair_output = all_possible_pairs[pair_idxes]
                                        relation_prediction = all_relation_outputs[pair_idxes]
                                        
                                        meme_creator_id = None # check for presence of MEME_CREATOR
                                        for actual_entity_trio in singlesample["actual_entities"]:
                                            if actual_entity_trio[2]=="MEME_CREATOR":
                                                meme_creator_id = actual_entity_trio[0]
                                        
                                        try:
                                            if pair_output[0][0]=="MEME_CREATOR":
                                                # print("MEME1")
                                                correct_target = singlesample["inverted_numerical_relationships"][(meme_creator_id,pair_output[1][0])]
                                            elif pair_output[1][0]=="MEME_CREATOR":
                                                # print("failure in dataset")
                                                continue # we don't have any memes that have a reaction TOWARDS the meme creator. it's always the other way round.
                                                # correct_target = singlesample["inverted_numerical_relationships"][(pair_output[0][0],meme_creator_id)]
                                            else:
                                                # print("reg")
                                                correct_target = singlesample["inverted_numerical_relationships"][(pair_output[0][0],pair_output[1][0])]
                                        except KeyError:
                                            print((pair_output[0][0],meme_creator_id,pair_output[1][0]))
                                            print(singlesample["actual_entities"])
                                            print(singlesample["relationship_num"])
                                            print(singlesample["inverted_numerical_relationships"])
                                            print("@"*1234)
                                            input()
                                        
                                        if guidance:
                                        
                                            targettensor = torch.zeros(relation_prediction.shape).unsqueeze(0)
                                            # print(targettensor.shape)
                                            # input()
                                            # print(correct_target)
                                            for relationtype in correct_target:
                                                targettensor[0][relationtype] = 1
                                            sigmoided_version = relation_embed_head.sigmoid(relation_prediction)

                                            # print("prediction:",relation_prediction.shape)
                                            # print("target:",targettensor.shape)
                                            # input()
                                            relationloss = relational_binary_sample_loss(relation_prediction.unsqueeze(0),targettensor.to(device))
                                            fewshotrelation_loss_sum+=relationloss
                                            
                                            selected_relations = []
                                            for idx in range(len(sigmoided_version)):
                                                if sigmoided_version[idx]> relation_threshold:
                                                    selected_relations.append(idx)
                                            # print(selected_relation,correct_target)
                                            # input()
                                            for targetidx in range(len(targettensor.squeeze())):
                                                if targettensor.squeeze()[targetidx]==1: # if there's supposed to be arelation here...
                                                    if sigmoided_version[targetidx]>relation_threshold:
                                                        # print("Correct Relation")
                                                        interaction_record[targetidx][0] = interaction_record[targetidx][0] + 1
                                                    else:
                                                        # print("Wrong Relation")
                                                        interaction_record[targetidx][1] = interaction_record[targetidx][1] + 1
                                        
                                            relation_prediction_result_dict[str(pair_output[0][0])+"_;_"+str(pair_output[1][0])] = {
                                                "idSENDER":pair_output[0][0],
                                                "idRECEIVER":pair_output[1][0],
                                                "text (Sender)":pair_output[0][2],
                                                "Origin (Sender)":pair_output[0][3],
                                                "text (Receiver)":pair_output[1][2],
                                                "Origin (Receiver)":pair_output[1][3],
                                                "Prediction":relation_prediction.cpu().tolist(),
                                                "Prediction (Logits)":sigmoided_version.cpu().tolist(),
                                                "loss":relationloss.item(),
                                                "selected_relations":selected_relations,
                                                "correct_relation":correct_target,
                                            }
                                        else:
                                            sigmoided_version = relation_embed_head.sigmoid(relation_prediction)
                                            relation_prediction_result_dict[str(pair_output[0][0])+"_;_"+str(pair_output[1][0])] = {
                                                "text (Sender)":pair_output[0][1],
                                                "Origin (Sender)":pair_output[0][2],
                                                "text (Receiver)":pair_output[1][1],
                                                "Origin (Receiver)":pair_output[1][2],
                                                "Prediction":relation_prediction.cpu().tolist(),
                                                "Prediction (Logits)":sigmoided_version.cpu().tolist(),
                                                "selected_relation":selected_relation,
                                            }
                                            pprint.pprint(relation_prediction_result_dict[str(pair_output[0][0])+"_;_"+str(pair_output[1][0])])
                                        # print(relation_prediction_result_dict[(pair_output[0][0],pair_output[1][0])])
                                    prediction_dict[singlesample["source_image"]] = {"archetype":singlesample["archetype"], "relations":relation_prediction_result_dict,"entities":sample_entity_report_dict}
                                        
                                        
                                    
                                if type(fewshotentity_loss_sum)!=int:
                                    total_loss += fewshotentity_loss_sum.item()
                                if type(fewshotrelation_loss_sum)!=int:
                                    total_loss += fewshotrelation_loss_sum.item()
                                if istrain:
                                    if type(fewshotentity_loss_sum)!=int:
                                        fewshotentity_loss_sum.backward()
                                    if type(fewshotrelation_loss_sum)!=int and guidance:
                                        fewshotrelation_loss_sum.backward()
                                        
                                
                                
                                
                                 # contains all the different textbox entity detections.

                                # print(prediction_dict[singlesample["source_image"]])
                                    
                                    
                                
                                
                                if istrain:
                                    # backproptimer = time.time()
                                    relationheadoptimizer.step()
                                    entityheadoptimizer.step()
                                    
                                # print("backprop time:",time.time()-backproptimer)
                            
                                relationheadoptimizer.zero_grad()
                                entityheadoptimizer.zero_grad()
                                
                                
                                # print(list_of_positionals) # {"text1":{ "height"   "width"   "x"    "y"   "text"   "vector" }.... "text2":{  } ....}
                                # note that vector refers to the input positional data, for relationship extraction
                                
                                
                                
                                print("Batch Loss:",total_loss,"Batch timer:",time.time()-batchtimer)
                                # quit()
                            
                                entity_correctness_record["total_entities"] += total_entities
                                entity_correctness_record["correct_entities_startstops"]+= entity_correctness_counter
                                entity_correctness_record["pure_correct_entities"]+= entity_pure_correctness_counter
                                # break
                                if istrain:
                                    fewshottotal_train_loss += total_loss
                                else:
                                    fewshottotal_test_loss += total_loss
                                
                                
                        
                    if fewshotentity_recorder_train["total_entities"]==0:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Train: ",0, " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
                    else:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Train: ",100*fewshotentity_recorder_train["correct_entities_startstops"]/fewshotentity_recorder_train["total_entities"], " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
                        
                    if fewshotentity_recorder_train["total_entities"]==0:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Train: ",0, " % of total entities perfectly detected (not just startstop but also the middle)")
                    else:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Train: ",100*fewshotentity_recorder_train["pure_correct_entities"]/fewshotentity_recorder_train["total_entities"], " % of total entities perfectly detected (not just startstop but also the middle)")
                    
                        
                    if fewshotentity_recorder_test["total_entities"]==0:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Test: ",0, " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
                    else:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Test: ",100*fewshotentity_recorder_test["correct_entities_startstops"]/fewshotentity_recorder_test["total_entities"], " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
                    
                    
                    if fewshotentity_recorder_test["total_entities"]==0:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Test: ",0, " % of total entities perfectly detected (not just startstop but also the middle)")
                    else:
                        print("fewshot_"+str(target_n)+"_innate_epoch_"+str(innate_epoch)+"Test: ",100*fewshotentity_recorder_test["pure_correct_entities"]/fewshotentity_recorder_test["total_entities"], " % of total entities perfectly detected (not just startstop but also the middle)")
                    
                    
                    print("relation_classwise: Train:",fewshotrelation_classwise_train)
                    print("relation_classwise: Test:",fewshotrelation_classwise_test)
                    print("Total train Loss:",fewshottotal_train_loss)
                    print("Total test Loss:",fewshottotal_test_loss)
                        
                    
                    
                    end_epoch_time = time.time()
                    fewshotepoch_time_taken = end_epoch_time - fewshotstart_epoch_time
                    overall_time_taken = time.time() - total_timer
                    print("fewshot epoch time taken - Innate epoch "+ str(innate_epoch)+" ("+str(target_n)+"):",fewshotepoch_time_taken)
                    
                    
                        
                        
                    with open("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_"+"_innate_epoch_"+str(innate_epoch)+"_"+"results_train",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json","w",encoding="utf-8") as dumpfile:
                        json.dump({"entity":fewshotentity_recorder_train,"relation":fewshotrelation_classwise_train},dumpfile,indent=4)
                        saved_resultsfilenames.append("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_"+"_innate_epoch_"+str(innate_epoch)+"_"+"results_train",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json")
                        
                    with open("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"results_test",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json","w",encoding="utf-8") as dumpfile:
                        json.dump({"entity":fewshotentity_recorder_test,"relation":fewshotrelation_classwise_test},dumpfile,indent=4)
                        saved_resultsfilenames.append("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"results_test",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json")
                    
                    with open("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"predictiondict_train",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json","w",encoding="utf-8") as dumpfile:
                        json.dump(fewshottrain_predcomparedict,dumpfile,indent=4)
                        saved_resultsfilenames.append("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"predictiondict_train",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json")
                        
                    with open("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"predictiondict_test",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json","w",encoding="utf-8") as dumpfile:
                        json.dump(fewshottest_predcomparedict,dumpfile,indent=4)
                        saved_resultsfilenames.append("_".join([additional_name_prefix+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+"predictiondict_test",entity_embed.model_name,"_",relation_embed.model_name,prefix])+".json")
                    
                    if innate_epoch==29 or innate_epoch%5==0:
                        for save_statedict,savename in entity_embed.return_statedicts():
                            torch.save(save_statedict,"_".join([additional_name_prefix+"ENT"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,savename,"embedder",relation_embed.model_name,"partner"])+".torch" )
                            saved_weightsfilenames.append("_".join([additional_name_prefix+"ENT"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,savename,"embedder",relation_embed.model_name,"partner"])+".torch" )
                            
                        torch.save(entity_embed_head.state_dict(),"_".join([additional_name_prefix+"ENTHEAD_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,entity_embed.model_name,"HEAD",relation_embed.model_name,"partner"])+".torch" )
                        saved_weightsfilenames.append("_".join([additional_name_prefix+"ENTHEAD_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,entity_embed.model_name,"HEAD",relation_embed.model_name,"partner"])+".torch" )
                        
                        for save_statedict,savename in relation_embed.return_statedicts():
                            torch.save(save_statedict,"_".join([additional_name_prefix+"RLN_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,savename,relation_embed.model_name,"embedder",entity_embed.model_name,"partner"])+".torch" )
                            saved_weightsfilenames.append("_".join([additional_name_prefix+"RLN_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,savename,relation_embed.model_name,"embedder",entity_embed.model_name,"partner"])+".torch")
                            
                        torch.save(relation_embed_head.state_dict(),"_".join([additional_name_prefix+"RLNHEAD_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,relation_embed.model_name,"HEAD",entity_embed.model_name,"partner"])+".torch" )
                        saved_weightsfilenames.append("_".join([additional_name_prefix+"RLNHEAD_"+str(target_n)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_innate_epoch_"+str(innate_epoch)+"_"+prefix,relation_embed.model_name,"HEAD",entity_embed.model_name,"partner"])+".torch")
                
                if target_fewshot_templates:
                    target_additional = "_"+str(target_n)+"_"
                else:
                    target_additional = ""
                with ZipFile(additional_name_prefix+target_additional+"batch"+str(batch_size)+"_"+ str(optimizerlr)[:2] +"_weights"+".zip", 'a',compression=zipfile.ZIP_DEFLATED) as weights_zipfile:
                    for target_dump_filename in saved_weightsfilenames:
                        weights_zipfile.write(target_dump_filename)
                        os.remove(target_dump_filename)
                        
                with ZipFile(additional_name_prefix+target_additional+"batch"+str(batch_size)+"_"+ str(optimizerlr)[:2] +"_results"+".zip", 'a',compression=zipfile.ZIP_DEFLATED) as results_zipfile: # always 'a' to prevent deleting old stuff.
                    for target_dump_filename in saved_resultsfilenames:
                        results_zipfile.write(target_dump_filename)
                        os.remove(target_dump_filename)
                
                print("Completed fewshot cycle. Reloaded to main state.")
            
    print("Done")

        

    


    
    




import os
import torch
import pprint
import time
import json
import cProfile
import itertools
import PIL.Image
import sys
sys.path.append("..") # Adds higher directory 
import utils
import copy
import sklearn.metrics
import numpy as np
import random
import zipfile
from zipfile import ZipFile
from io import BytesIO    
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation, GPT_NEO_preparation

# ALTCLIP
# from transformers import AutoProcessor, AltCLIPTextModel, AltCLIPVisionModel

# dataset class
from model_dataset_class import text_vector_dataset_extractor, entity_error_analysis

# trilinear
from attached_heads import trilinear_head,deeper_trilinear_head_relations
torch.set_num_threads(2)
torch.set_num_interop_threads(2)


class dual_dataset(Dataset):

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
    
    relationship_mirror = {
        "Upgrade":"Degrade",
        "Degrade":"Upgrade", # we can add superior here!
        "Superior":"Inferior",
        "Inferior":"Superior",
        "Equal":"Equal"
    }

    
    def __init__(self, importfile, dumpdir = None,target_tokeniser=False,approved_images=[],verbose=False,use_templates=False,template_dir="templates"):
        with open(importfile,"r",encoding="utf-8") as datasetfile:
            temp_savelist = json.load(datasetfile)
        
        relationships_counter = {}
        entitycounter = {}
        textcounter = {}
        self.savelist = []
        
        for item in temp_savelist:
            if approved_images:
                if not item["source_image"] in approved_images:
                    continue
                self.savelist.append(item)
            else:
                self.savelist.append(item)
        
                
                
        for item in self.savelist:
            # pprint.pprint(item)
            held_dict = {}
            for relationshippart in item["relationship_num"]:
                held_dict[int(relationshippart)] = item["relationship_num"][relationshippart]
            item["relationship_num"] = held_dict
            
            
            numerical_relationships = item["relationship_num"]
            # print(numerical_relationships) # generate inverted method
            
                
            inverted_numerical_relationships = {}
            for k in numerical_relationships:
                for pairinstance in numerical_relationships[k]:
                    if not tuple(pairinstance) in inverted_numerical_relationships:
                        inverted_numerical_relationships[tuple(pairinstance)] = []
                    inverted_numerical_relationships[tuple(pairinstance)].append(k)
                
        
            textbox_counter = len(list(item["span_answer"].keys()))
            
            if not textbox_counter in textcounter:
                textcounter[textbox_counter] = 0
            textcounter[textbox_counter] += 1
                    
            
            if use_templates:
                item["source_image"] = os.path.join(template_dir, item["archetype"]+".png")
            # print(item["relationships_read"])
            for i in item["relationships_read"]: # relationships counter
                if not i in relationships_counter:
                    relationships_counter[i] = 0
                relationships_counter[i] += len(item["relationships_read"][i])
            item["inverted_numerical_relationships"] = inverted_numerical_relationships
            # pprint.pprint(singleinstance)
            # print(numerical_relationships)
            # print(inverted_numerical_relationships)
            # input()
            
            
            # singleinstance = {
                # "text_locs":all_items[imagekey]["textbox_locations"],
                # "correct_answers":boxes_answer,
                # "span_answer":span_registration,
                # "equivalent_entities":simplified_equivalents,
                # "relationships_read":relationships,
                # "relationship_num":numerical_relationships,
                # "inverted_numerical_relationships":inverted_numerical_relationships,
                # "source_image":imagekey,
                # "image_addr":os.path.join(dumpdir,imagekey),
                # "meme_creator":meme_creatorpresence,
                # "tokenised_strings": all_items[imagekey]["tokenised_strings"],
                # "actual_entities": all_TRUE_entities,
                # "archetype": all_items[imagekey]["archetype"]
                # Inverted is manually processed as you can see above.
            # }
    
    
        
        print("Entity Counter:",entitycounter)
        print("Relationships:",relationships_counter)
        print("Text:",textcounter)
        print(self.remapped_actionables_dict)
        self.relationships_counter = relationships_counter
        # input()
    
    def __getitem__(self,idx):
        return self.savelist[idx]
    
    def __len__(self):
        return len(self.savelist)
            
    def collate_fn(self,input_data):
        return copy.deepcopy(input_data)



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
    prefix = "abalated_b16l5_SGMEMES"
    
    
    # fewshot_targetks = [0,5,10,20]
    # optimizerlrlist = [1e-5,2e-5,5e-5]
    # batch_sizelist = [4,8,16,32]
    
    splitratio = 0.6
    optimizerlr = 5e-5
    batch_size = 16
    
    entity_threshold = 0.5
    relation_threshold = 0.5
    data_dir = os.path.join("TDMEMES","TD_Memes")
    labels_file = "SGMEMES_dataset_processed_final.json"
    split_number_cap = 2 # 0,1,2
    
    all_imageslist = []
    with open(labels_file,"r",encoding="utf-8") as labelfile:
        loaded_labelsfile_dict = json.load(labelfile)
        for item in loaded_labelsfile_dict:
            all_imageslist.append(item["source_image"])
    
    entityheadname = "" # model .torch files. Or just comment out the loads below if you're testing.
    entityname = ""
    relationheadname = ""
    relationname_img = ""
    relationname_text = ""
    
    print("Total in SGMEMES:",len(all_imageslist))
            
    if not os.path.exists("SG_splitlists.json"):
        splitslist = []
        with open("SG_splitlists.json","w",encoding="utf-8") as sgsplitlistfile:
            z = int(len(all_imageslist)*splitratio)
            for _ in range(split_number_cap):
                random.shuffle(all_imageslist)
                splitslist.append([all_imageslist[:z],all_imageslist[z:]])
            json.dump(splitslist,sgsplitlistfile,indent=4)
    else:
        with open("SG_splitlists.json","r",encoding="utf-8") as sgsplitlistfile:
            splitslist = json.load(sgsplitlistfile)


    noposition = True
    guidance = True
    deeper_head = True
    
    if noposition:
        zipfile_abbr = "abalated"
    else:
        zipfile_abbr = "normal"
    
    test_batchsize = 56
    fewshot_internalepochs = 30
    

    entity_scheduler_stepsize = 5
    relation_scheduler_stepsize = 3

    


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
    if deeper_head:
        relation_embed_head = deeper_trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=noposition)
    else:
        relation_embed_head = trilinear_head_relations(device,image_embed1=relation_embed.imagembed_size1,image_embed2=relation_embed.imagembed_size2,text_embed=relation_embed.textembed_size,targetsize = 9,parameter=True,wordonly=wordonly,noposition=noposition)
        
    relation_internal_parameter = relation_embed_head.parameter # represents meme creator
    relation_embed_head.to(relation_embed_head.device)
    relationheadoptimizer = torch.optim.Adam(list(relation_embed.optimizer_parameters)+list(relation_embed_head.parameters()),lr=optimizerlr,weight_decay = weightdecay)


    
    # fewshottest_constructor = [i for i in list(standard_imagesize_dict.keys()) if not i in leaveouts]
    
    relationscheduler = StepLR(relationheadoptimizer, step_size=relation_scheduler_stepsize, gamma=0.5) # every x epochs, We REDUCE learning rate.
    entityscheduler = StepLR(entityheadoptimizer, step_size=entity_scheduler_stepsize, gamma=0.5) # every x epochs, We REDUCE learning rate.
    
    entity_embed.model.load_state_dict(torch.load(entityname))
    entity_embed_head.load_state_dict(torch.load(entityheadname))
    relation_embed.imagemodel.load_state_dict(torch.load(relationname_img))
    relation_embed.textmodel.load_state_dict(torch.load(relationname_text))
    relation_embed_head.load_state_dict(torch.load(relationheadname))


    # print("Fewshot train image list:",len(fewshottrain_img_list))
    # print("Fewshot test image list:",len(fewshottest_img_list))
    for split_number in range(split_number_cap):
        trainimglist,testimglist  = splitslist[split_number]

        fewshottrain_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type],approved_images=trainimglist)
        fewshottest_meme_dataset = dual_dataset(labels_file,target_tokeniser=[entity_embed.model_type,relation_embed.model_type],approved_images=testimglist)
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

        
        
        fewshottrain_meme_dataloader = DataLoader(fewshottrain_meme_dataset, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=fewshottest_meme_dataset.collate_fn)
        fewshottest_meme_dataloader = DataLoader(fewshottest_meme_dataset, batch_size=test_batchsize, shuffle=True, num_workers=0,collate_fn=fewshottest_meme_dataset.collate_fn)
        saved_weightsfilenames = []
        saved_resultsfilenames = []
    
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
                            prediction_dict[singlesample["source_image"]] = { "relations":relation_prediction_result_dict,"entities":sample_entity_report_dict}
                                
                                
                            
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
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Train: ",0, " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
            else:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Train: ",100*fewshotentity_recorder_train["correct_entities_startstops"]/fewshotentity_recorder_train["total_entities"], " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
                
            if fewshotentity_recorder_train["total_entities"]==0:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Train: ",0, " % of total entities perfectly detected (not just startstop but also the middle)")
            else:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Train: ",100*fewshotentity_recorder_train["pure_correct_entities"]/fewshotentity_recorder_train["total_entities"], " % of total entities perfectly detected (not just startstop but also the middle)")
            
                
            if fewshotentity_recorder_test["total_entities"]==0:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Test: ",0, " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
            else:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Test: ",100*fewshotentity_recorder_test["correct_entities_startstops"]/fewshotentity_recorder_test["total_entities"], " % of total entities start and stops detected. Note it doesn't account for in between true start and stop span errors")
            
            
            if fewshotentity_recorder_test["total_entities"]==0:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Test: ",0, " % of total entities perfectly detected (not just startstop but also the middle)")
            else:
                print(str(split_number)+"_innate_epoch_"+str(innate_epoch)+"Test: ",100*fewshotentity_recorder_test["pure_correct_entities"]/fewshotentity_recorder_test["total_entities"], " % of total entities perfectly detected (not just startstop but also the middle)")
            
            
            print("relation_classwise: Train:",fewshotrelation_classwise_train)
            print("relation_classwise: Test:",fewshotrelation_classwise_test)
            print("Total train Loss:",fewshottotal_train_loss)
            print("Total test Loss:",fewshottotal_test_loss)
                
            
            
            end_epoch_time = time.time()
            fewshotepoch_time_taken = end_epoch_time - fewshotstart_epoch_time
            overall_time_taken = time.time() - total_timer
            print("fewshot epoch time taken - Innate epoch "+ str(innate_epoch)+" ("+str(split_number)+"):",fewshotepoch_time_taken)
            
            
                
                
            with open("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_"+"epoch_"+str(innate_epoch)+"_"+"results_train",prefix])+".json","w",encoding="utf-8") as dumpfile:
                json.dump({"entity":fewshotentity_recorder_train,"relation":fewshotrelation_classwise_train},dumpfile,indent=4)
                saved_resultsfilenames.append("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"_"+"epoch_"+str(innate_epoch)+"_"+"results_train",prefix])+".json")
                
            with open("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"results_test",prefix])+".json","w",encoding="utf-8") as dumpfile:
                json.dump({"entity":fewshotentity_recorder_test,"relation":fewshotrelation_classwise_test},dumpfile,indent=4)
                saved_resultsfilenames.append("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"results_test",prefix])+".json")
            
            with open("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"predictiondict_train",prefix])+".json","w",encoding="utf-8") as dumpfile:
                json.dump(fewshottrain_predcomparedict,dumpfile,indent=4)
                saved_resultsfilenames.append("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"predictiondict_train",prefix])+".json")
                
            with open("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"predictiondict_test",prefix])+".json","w",encoding="utf-8") as dumpfile:
                json.dump(fewshottest_predcomparedict,dumpfile,indent=4)
                saved_resultsfilenames.append("_".join([entity_embed.model_name,"split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+"predictiondict_test",prefix])+".json")
            
            if innate_epoch==29 or innate_epoch%5==0:
                for save_statedict,savename in entity_embed.return_statedicts():
                    torch.save(save_statedict,"_".join([savename,"ENTsplit"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch" )
                    saved_weightsfilenames.append("_".join([savename,"ENTsplit"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch")
                    
                torch.save(entity_embed_head.state_dict(),"_".join([entity_embed.model_name,"ENTHEAD_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch")
                saved_weightsfilenames.append("_".join([entity_embed.model_name,"ENTHEAD_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch")
                
                
                for save_statedict,savename in relation_embed.return_statedicts():
                    torch.save(save_statedict,"_".join([relation_embed.model_name,"RLN_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix,savename])+".torch" )
                    saved_weightsfilenames.append("_".join([relation_embed.model_name,"RLN_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix,savename])+".torch")
                    
                torch.save(relation_embed_head.state_dict(),"_".join([relation_embed.model_name+"RLNHEAD_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch" )
                saved_weightsfilenames.append("_".join([relation_embed.model_name+"RLNHEAD_split"+str(split_number)+"_"+str(optimizerlr)+"_"+str(batch_size)+"epoch_"+str(innate_epoch)+"_"+prefix])+".torch")
                
        with ZipFile("SGbatch"+zipfile_abbr+str(batch_size)+"_"+ str(optimizerlr)[:2] +"_weights"+".zip", 'a',compression=zipfile.ZIP_DEFLATED) as weights_zipfile:
            for target_dump_filename in saved_weightsfilenames:
                weights_zipfile.write(target_dump_filename)
                os.remove(target_dump_filename)
                    
        with ZipFile("SGbatch"+zipfile_abbr+str(batch_size)+"_"+ str(optimizerlr)[:2] +"_results"+".zip", 'a',compression=zipfile.ZIP_DEFLATED) as results_zipfile: # always 'a' to prevent deleting old stuff.
            for target_dump_filename in saved_resultsfilenames:
                results_zipfile.write(target_dump_filename)
                os.remove(target_dump_filename)
        
        print("Completed cycle.")
    
            
    print("Done")

        

    


    
    




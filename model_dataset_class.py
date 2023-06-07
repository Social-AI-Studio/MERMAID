import os
import torch
import pprint
import json
import copy
import PIL.Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from parse.mass_tokenizer_class import dataset_information_generation_class

def text_vector_dataset_extractor(single_sample,device="cpu",tokeniser=False):
        # given a single output from the dataset class (unmorphed by collatefn), return text, and additional_data_vector in the form of a dictionary.
        #   Note that textboxname is text1, text2, etc.
        #         trackeritem[textboxname+"text"] -> either torch.tensor or a string.
        #         trackeritem[textboxname+"vector"] -> [height, width, x, y] in terms of %.   returns a torch tensor here.
        # if a tokeniser was provided, the text is tokenised, and the input_id vector is returned accordingly. (not the rest of the stuff.)
        
        
        # note that you should probably shuffle the input order of textboxes for a model. 
        # random.shuffle will suffice since there's a chance no change occurs from the original input.
        
        
        external_tracker = {}
        # print(single_sample)
        for textboxname in single_sample["text_locs"]:
            external_tracker[textboxname] = single_sample["text_locs"][textboxname][0]
            if not tokeniser:
                external_tracker[textboxname]["text"] = single_sample["text_locs"][textboxname][1]
            else:
                external_tracker[textboxname]["text"] = tokeniser(single_sample["text_locs"][textboxname][1],return_tensors="pt")["input_ids"][0]
            external_tracker[textboxname]["vector"] = torch.tensor([external_tracker[textboxname]["height"]*0.01,
                external_tracker[textboxname]["width"]*0.01,
                external_tracker[textboxname]["x"]*0.01,
                external_tracker[textboxname]["y"]*0.01]).to(device)
        return copy.deepcopy(external_tracker) # deepcopy here severs the link between the dictionary and the dataset completely, preventing randomly saving garbage.
        # this is due to the way our dataloader returns a string directly. Used in training loops.


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
        archcounter = {}
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
                
            
            if not item["archetype"] in archcounter:
                archcounter[item["archetype"]] = 0
            archcounter[item["archetype"]] +=1
        
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
    
    
        print("Archetype counter:",archcounter)
        
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




def entity_error_analysis(given,target,span_answer,threshold,device="cpu",loss_entity=False,transcendence=False):
    # transcendence indicates if the dataset has span transcending entities.
    if len(given) != len(target):
        print("******"*30)
        print("Given:",given)
        print("Target:",target)
        raise ValueError("given versus target is different.")
        
    all_entities = {}
    for item in span_answer:
        if item:
            if transcendence:
                all_entities[item] = {"start":False,"stop":False,"flag":False,
                    "falsenegstart":False,"falsenegstop":False,"falsenegflag":False,
                    "falseposstart":False,"falseposstop":False,"falseposflag":False, "name":item, "midfailure": False
                    }
            else:
                all_entities[item] = {"start":False,"stop":False,
                    "falsenegstart":False,"falsenegstop":False,
                    "falseposstart":False,"falseposstop":False, "name":item, "midfailure": False
                    }
    # pprint.pprint(span_answer)
    # print(given)
    # print(target)
    # pprint.pprint(all_entities)
                
    if loss_entity:
        # print("0"*30)
        # print(given.shape)
        # print(torch.tensor(target).shape)
        # print("0"*30)
        total_loss = loss_entity(given.float(),torch.tensor(target).float().to(device))
    else:
        total_loss = False
        
    if transcendence:
        target_counter = 3
    else:
        target_counter = 2
    
    mid_of_entity=False
    currententity=False
    tempsigmoid = torch.nn.Sigmoid()
    sigmoided_given = tempsigmoid(given.cpu().detach())
    
    for idx in range(len(target)):
        # print(given[idx],target[idx])
        

        for type_identifier in range(target_counter):
            if type_identifier==0:
                flagword = "start"
            elif type_identifier==1:
                flagword = "stop"
            else:
                flagword = "flag"
            # print(idx,type_identifier,flagword,span_answer[idx])
                
            if sigmoided_given[idx][type_identifier]<=threshold and target[idx][type_identifier]<=threshold:
                # nothing happens. no one should be "Targeted".
                # print("passed")
                pass
                
            elif sigmoided_given[idx][type_identifier]<=threshold and target[idx][type_identifier]>threshold:
                all_entities[span_answer[idx]]["falseneg"+flagword] = False # note that we screwed up the tag of an entity. we didn't flag when we should have.
                if mid_of_entity: # if you are in the middle of an entity, or you double flag something, we record it as a "mid failure". where you screw up in the middle of an entity
                    all_entities[span_answer[idx]]["midfailure"] = True
                
                
                if flagword=="start":
                    mid_of_entity = True
                    currententity = span_answer[idx]
                elif flagword=="stop":
                    mid_of_entity = False
                    currententity=False
                # print("false negatived:",span_answer[idx])
                
                # this is considered a FALSE NEGATIVE
                
            elif sigmoided_given[idx][type_identifier]>threshold and target[idx][type_identifier]>threshold:
                all_entities[span_answer[idx]][flagword] = True # we got it correct.
                if flagword=="start":
                    mid_of_entity = True
                    currententity = span_answer[idx]
                elif flagword=="stop":
                    mid_of_entity = False
                    currententity = False
                # print("Correct:",span_answer[idx])
                
            elif sigmoided_given[idx][type_identifier]>threshold and target[idx][type_identifier]<=threshold:
                if currententity:
                    all_entities[currententity]["falsepos"+flagword] = False # note that we screwed up the tag of an entity. we flagged, but it wasn't correct.
                if mid_of_entity: # if you are in the middle of an entity, or you double flag something, we record it as a "mid failure". where you screw up in the middle of an entity
                    all_entities[currententity]["midfailure"] = True
                # print("false positived:",span_answer[idx])
                # This is considered a FALSE POSITIVE
    # A report has been produced.
    return all_entities,total_loss
    

def dataset_list_split(importfile,trainsplit_ratio,disallowed_archetypes,dumpdir=None,verbose=False,return_smartstrike=False,minimal_n=None,report_archetype_counts=False,approved_images=False):


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



    # parser function. Spits out the number of relationship combinations that are present in each meme.
    if dumpdir==None:
        dumpdir = os.path.join("parse","raw_data_jsons","image_dir")
    with open(importfile,"r",encoding="utf-8") as opened_file:
        all_items = json.load(opened_file)

    
    archetypes = {}
    meme_creatorpresence = False
    
    # print("Total Images (before removing textbox spanning entity instances:",len(list(all_items.keys())))
    
    for image_item in all_items:
        if approved_images:
            if not image_item["source_image"] in approved_images: # brute force image check. Only approved images can be considered now.
                continue
                
                
        with Image.open(os.path.join(dumpdir,image_item["source_image"])) as checkimage: 
            # Cause an error if the image doesn't exist in dumpdir.
            # if your dataset is missing images, this will prevent time wastage (except the importing models bit which will eat time regardless.)
        
            ref_image_size = checkimage.size
        
        if not image_item["archetype"] in archetypes:
            archetypes[image_item["archetype"]] = {}
        
        boxes_answer = {}
        
            
        equivalents = image_item["equivalent_entities"]
        
        
        
        
        relationcounter = []
        numerical_relationships = image_item["relationship_num"]
        
        for relationtype in sorted(list(numerical_relationships.keys())):
            # relationcounter.append((relationtype,len(numerical_relationships[relationtype])))
            
            relationcounter.append(relationtype)
            
        if not tuple(relationcounter) in archetypes[image_item["archetype"]]:
            archetypes[image_item["archetype"]][tuple(relationcounter)] = []
        archetypes[image_item["archetype"]][tuple(relationcounter)].append(image_item["source_image"])
            
        
    if verbose:
        print(remapped_actionables_dict)
        print(relationship_mirror)
    
    
    trainsplit = {}
    testsplit = {}
    total_arches = 0
    
    archetype_data_record = {}
    
    
    if minimal_n==None:
        for arch in archetypes:
            # sort from largest number of samples for a relation combination to smallest.
            for subarch in archetypes[arch]:
                total_arches += len(archetypes[arch][subarch])
            # for relationcombi in list(archetypes[arch].keys()):
                # print(arch,relationcombi,len(archetypes[arch][relationcombi]))
            archetypes[arch] ={k: v for k, v in sorted(archetypes[arch].items(), key=lambda item: len(item[1]),reverse=True)}
            # for relationcombi in list(archetypes[arch].keys()):
                # print(arch,relationcombi,len(archetypes[arch][relationcombi]))
        print("Total Images",total_arches)
        
        final_archtypes = {}
        for arch in archetypes:
            if arch in disallowed_archetypes: # skip disallowed archetypes
                continue
            trainsplit[arch] = {}
            testsplit[arch] = {}
            train_seen_types = set()
            seen_rl_combi = []
            total_samples_arch = 0
            for relationcombi in list(archetypes[arch].keys()):
                
                if verbose:
                    print(arch,relationcombi,len(archetypes[arch][relationcombi]))
                trainsplit[arch][relationcombi] = {}
                testsplit[arch][relationcombi] = {}
                total_samples_arch += len(archetypes[arch][relationcombi])
                seen_rl_combi.append(relationcombi)
                if len(archetypes[arch][relationcombi])>5:
                    for relation in relationcombi:
                        train_seen_types.add(relation)
                    splitidx = int(len(archetypes[arch][relationcombi])*trainsplit_ratio)
                    trainsplit[arch][relationcombi] = archetypes[arch][relationcombi][:splitidx] 
                    testsplit[arch][relationcombi] = archetypes[arch][relationcombi][splitidx:]
                    if verbose:
                        print("splitidx:",splitidx)
                        print("original length:",len(archetypes[arch][relationcombi]))
                        print("trainsplit added:",len(trainsplit[arch][relationcombi]))
                        print("testsplit added:",len(testsplit[arch][relationcombi]))
                    
                elif len(archetypes[arch][relationcombi])>1:
                    for relation in relationcombi:
                        train_seen_types.add(relation)
                    trainsplit[arch][relationcombi] = archetypes[arch][relationcombi][:1] 
                    testsplit[arch][relationcombi] = archetypes[arch][relationcombi][1:]
                    if verbose:
                        print(archetypes[arch][relationcombi])
                        print(trainsplit[arch][relationcombi])
                        print(testsplit[arch][relationcombi])
                        print("original length:",len(archetypes[arch][relationcombi]))
                        print("trainsplit added:",len(trainsplit[arch][relationcombi]))
                        print("testsplit added:",len(testsplit[arch][relationcombi]))
                else:
                    # There is only a single sample of such a combination...
                    all_in_sets = True
                    for relation in relationcombi:
                        if not relation in train_seen_types:
                            all_in_sets = False
                    # are all relations currently seen in the train set already?
                    
                    if all_in_sets:
                        if verbose:
                            print("^ To Test")
                        # print(arch,relationcombi,archetypes[arch][relationcombi][0]) # for visibility.
                        testsplit[arch][relationcombi] = archetypes[arch][relationcombi][0:]
                    else:
                        if verbose:
                            print("^ To Train")
                        # print(arch,relationcombi,archetypes[arch][relationcombi][0]) # for visibility.
                        trainsplit[arch][relationcombi] = archetypes[arch][relationcombi][0:]
                        for relation in relationcombi:
                            train_seen_types.add(relation)
                if verbose:
                    print("0"*30)
            # Split requirements: At least one of each relation type visible WITHIN training split. Number of entities with said relation does not matter.
            if verbose:
                print("-"*90)
                
            final_archtypes[arch] = train_seen_types
            archetype_data_record[arch] = {}
            archetype_data_record[arch]["all_relationships"] = list(train_seen_types)
            archetype_data_record[arch]["relationship_combis"] = seen_rl_combi
            archetype_data_record[arch]["samplecount"] = total_samples_arch
                
        
    else:
        print("----fewshot_variation triggered----")
        for arch in archetypes:
            # sort from largest number of samples for a relation combination to smallest.
            for subarch in archetypes[arch]:
                total_arches += len(archetypes[arch][subarch])
            # for relationcombi in list(archetypes[arch].keys()):
                # print(arch,relationcombi,len(archetypes[arch][relationcombi]))
            archetypes[arch] ={k: v for k, v in sorted(archetypes[arch].items(), key=lambda item: len(item[1]),reverse=True)}
            # for relationcombi in list(archetypes[arch].keys()):
                # print(arch,relationcombi,len(archetypes[arch][relationcombi]))
        print("Total arches",total_arches)
        
        final_archtypes = {}
        for arch in archetypes:
            if arch in disallowed_archetypes: # skip disallowed archetypes
                continue
            trainsplit[arch] = {}
            testsplit[arch] = {}
            train_seen_types = set()
            seen_rl_combi = []
            total_samples_arch = 0
            # fish unique relns. and then pool the rest.
            # fish to max number of samples afterwards.
            fillerpool = {}
            for relationcombi in list(archetypes[arch].keys()):
                if verbose or report_archetype_counts:
                    print(arch,relationcombi,len(archetypes[arch][relationcombi]))
                trainsplit[arch][relationcombi] = []
                testsplit[arch][relationcombi] = []
                total_samples_arch += len(archetypes[arch][relationcombi])
                seen_rl_combi.append(relationcombi)
                
                new_relation_combi_flag = False
                for relation in relationcombi:
                    if not relation in train_seen_types:
                        new_relation_combi_flag= True
                
                if not new_relation_combi_flag:
                    fillerpool[relationcombi] = archetypes[arch][relationcombi]
                    continue
                
                if len(archetypes[arch][relationcombi])>2: # >1, lesser than minimal_n. Prioritise placing only ONE sample into minimal.
                    for relation in relationcombi:
                        train_seen_types.add(relation)
                    trainsplit[arch][relationcombi] = [archetypes[arch][relationcombi][0]]
                    testsplit[arch][relationcombi] = [archetypes[arch][relationcombi][1]]
                    fillerpool[relationcombi] = archetypes[arch][relationcombi][2:]
                    if verbose:
                        print(archetypes[arch][relationcombi])
                        print(trainsplit[arch][relationcombi])
                        print(testsplit[arch][relationcombi])
                        print("original length:",len(archetypes[arch][relationcombi]))
                        print("trainsplit added:",len(trainsplit[arch][relationcombi]))
                        print("testsplit added:",len(testsplit[arch][relationcombi]))
                        print("to filler:",len(fillerpool[relationcombi]))
                        
                elif len(archetypes[arch][relationcombi])==2: # 2. one each.
                    for relation in relationcombi:
                        train_seen_types.add(relation)
                    trainsplit[arch][relationcombi] = [archetypes[arch][relationcombi][0]]
                    testsplit[arch][relationcombi] = [archetypes[arch][relationcombi][1]]

                    if verbose:
                        print(archetypes[arch][relationcombi])
                        print(trainsplit[arch][relationcombi])
                        print(testsplit[arch][relationcombi])
                        print("original length:",len(archetypes[arch][relationcombi]))
                        print("trainsplit added:",len(trainsplit[arch][relationcombi]))
                        print("testsplit added:",len(testsplit[arch][relationcombi]))
                
                else:
                    # There is only a single sample of such a combination...
                    all_in_sets = True
                    for relation in relationcombi:
                        if not relation in train_seen_types:
                            all_in_sets = False
                    # are all relations currently seen in the train set already?
                    
                    if all_in_sets:
                        if verbose:
                            print("^ To Test")
                        # print(arch,relationcombi,archetypes[arch][relationcombi][0]) # for visibility.
                        testsplit[arch][relationcombi] = archetypes[arch][relationcombi][0:]
                    else:
                        if verbose:
                            print("^ To Train")
                        # print(arch,relationcombi,archetypes[arch][relationcombi][0]) # for visibility.
                        trainsplit[arch][relationcombi] = archetypes[arch][relationcombi][0:]
                        for relation in relationcombi:
                            train_seen_types.add(relation)
                
                if verbose:
                    print("0"*30)
            
            summer = 0
            for relationcombi in trainsplit[arch]:
                summer += len(trainsplit[arch][relationcombi])
            # print("summer:",summer)
            
            exit_padminimal_n=False
            while summer<minimal_n:
                for relationcombi in fillerpool:
                    startlen = len(fillerpool[relationcombi])
                    if not relationcombi in trainsplit[arch]:
                        trainsplit[arch][relationcombi] = []
                    if len(fillerpool[relationcombi])>0:
                        trainsplit[arch][relationcombi].append(fillerpool[relationcombi].pop())
                        total_samples_arch+=1
                        summer+=1
                    # print("startlen:",startlen,"endlen:",len(fillerpool[relationcombi]),"newsummer:",summer)
                    if summer>=minimal_n:
                        exit_padminimal_n=True
                        break
                leftover_tests = 0
                for relationcombi in fillerpool:
                    leftover_tests+=len(fillerpool[relationcombi])
                    
                if exit_padminimal_n or leftover_tests==0:
                    break
            # print(arch,fillerpool)
            # if leftover_tests==0:  
                # print("***"*30,arch,"***"*30)
            
            
            
            for relationcombi in fillerpool:
                if not relationcombi in testsplit[arch]:
                    testsplit[arch][relationcombi] = []
                # print(testsplit[arch][relationcombi])
                testsplit[arch][relationcombi].extend(fillerpool[relationcombi])
            
                
               
            final_archtypes[arch] = train_seen_types
            archetype_data_record[arch] = {}
            archetype_data_record[arch]["all_relationships"] = list(train_seen_types)
            archetype_data_record[arch]["relationship_combis"] = seen_rl_combi
            archetype_data_record[arch]["samplecount"] = total_samples_arch
                
    train_img_list = []
    test_img_list = []
    for arch in trainsplit:
        for relationcombi in trainsplit[arch]:
            train_img_list.extend(trainsplit[arch][relationcombi])
    for arch in testsplit:
        for relationcombi in testsplit[arch]:
            test_img_list.extend(testsplit[arch][relationcombi])
    
    print("Split results")
    print("train:",len(train_img_list))
    print("test:",len(test_img_list))
    
    if report_archetype_counts:
        pprint.pprint(archetype_data_record,width=100)
        print("-"*100)
        print("-"*100)
        print("-"*100)
        for arch in archetype_data_record:
            print(arch)
            print("number of unique relationship types", len(archetype_data_record[arch]["all_relationships"]))
            print("number of unique relationship combis", len(archetype_data_record[arch]["relationship_combis"]))
            print("Samples:",archetype_data_record[arch]["samplecount"])
        
    if return_smartstrike:
        return final_archtypes
    else:
        return train_img_list, test_img_list


if __name__=="__main__":
   
    # given the label file here:
    target_file = os.path.join("parse","ZZZZ_completed_processed_fullabels_shaun.json")
    


    # now try loading your dataset.
    dataset = dual_dataset("final_dataset_cleared.json")
    
    
    train_img_list, test_img_list = dataset_list_split("final_dataset_cleared.json",0.6,verbose=False)
    # normal split test
    
    
    # fewshot exclude certain archetypes test.
    all_archetypes = ['Buff-Doge-vs-Cheems', 'Cuphead-Flower', 'Drake-Hotline-Bling', 'Mr-incredible-mad', 'Soyboy-Vs-Yes-Chad', 'Spongebob-Burning-Paper', 'Squidward', 'Teachers-Copy', 'Tuxedo-Winnie-the-Pooh-grossed-reverse', 'Arthur-Fist', 'Distracted-Boyfriend', 'Moe-throws-Barney', 'Types-of-Headaches-meme', 'Weak-vs-Strong-Spongebob', 'This-Is-Brilliant-But-I-Like-This', 'Running-Away-Balloon', 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask', 'Mother-Ignoring-Kid-Drowning-In-A-Pool', 'kermit-window', 'Is-This-A-Pigeon', 'If-those-kids-could-read-theyd-be-very-upset', 'Hide-the-Pain-Harold', 'Feels-Good-Man', 'Clown-Applying-Makeup', 'Both-Buttons-Pressed', 'Anime-Girl-Hiding-from-Terminator', 'Epic-Handshake', 'Disappointed-Black-Guy', 'Blank-Nut-Button', 'Tuxedo-Winnie-The-Pooh', 'Ew-i-stepped-in-shit', 'Two-Paths', 'This-is-Worthless', 'They-are-the-same-picture', 'The-Scroll-Of-Truth', 'Spider-Man-Double', 'Skinner-Out-Of-Touch', 'Left-Exit-12-Off-Ramp', 'Fancy-pooh']
    target_fewshot_exclusions = ['Cuphead-Flower','Soyboy-Vs-Yes-Chad','Running-Away-Balloon','Mother-Ignoring-Kid-Drowning-In-A-Pool','Hide-the-Pain-Harold']
    non_fewshot_list = []

    for item in all_archetypes:
        if item in target_fewshot_exclusions:
            continue
        non_fewshot_list.append(item)
    
    target_n=10
    
    # test more splits.
    
    normal_train_split_imglist,normal_test_split_imglist  = dataset_list_split("final_dataset_cleared.json",0.6,target_fewshot_exclusions,minimal_n=target_n,verbose=True)
    
    fewshot_train_split_imglist, fewshot_test_split_imglist  = dataset_list_split("final_dataset_cleared.json",0.6,non_fewshot_list,minimal_n=target_n,verbose=True)
    
    quit()
    
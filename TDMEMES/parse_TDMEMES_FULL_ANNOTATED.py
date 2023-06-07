import json
import os
import pprint
import datetime
import urllib.parse
import random
import urllib.parse
from transformers import AutoTokenizer, ViltProcessor, AutoProcessor, Blip2Processor, GPT2Tokenizer
import PIL
from PIL import Image
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

def straighten(targetfile):
    with open(targetfile,"r",encoding="utf-8") as ocrfile:
        loaded_out = json.load(ocrfile)

    with open(targetfile,"w",encoding="utf-8") as ocrfile:
        json.dump(loaded_out,ocrfile,indent=4)


def all_sequence_extraction(a):
    outy = set()
    for n in range(len(a)):
        outy|=set(list(zip(*[a[i:] for i in range(n)])))
    return outy



def subsequence_searcher(subseq, seq):
    # https://stackoverflow.com/questions/425604/best-way-to-determine-if-a-sequence-is-in-another-sequence
    i, n, m = -1, len(seq), len(subseq)
    ranges_accepted = []
    try:
        while True:
            i = seq.index(subseq[0], i + 1, n - m + 1)
            if subseq == seq[i:i + m]:
               ranges_accepted.append(i)
    except ValueError:
        if ranges_accepted:
            return ranges_accepted
        else:
            return -1


class mass_tokenizer:
    def __init__(self, berttokenizer=None,robertatokenizer=None,cliptokenizer=None,data2vectokenizer=None,vilttokenizer=None,vilberttokenizer=None,bliptokenizer=None,blip2tokenizer=None,gpt_neotokenizer=None):
        
        # teststring = "This is a test string. Return me the various tokenised versions of this sentence, and show me your special token numbers in your tokenizer."
        
        
        
        """
        class vilt_method:
            
            def __init__(self,falseimage,actualprocessor):
                self.falseimage = falseimage
                self.processor = actualprocessor
            
            def __call__(self,input_text,return_tensors,add_special_tokens=False):
                return self.processor(self.falseimage,input_text,return_tensors=return_tensors,add_special_tokens=add_special_tokens)
        """
        
        
        if berttokenizer!=None:
            self.bert = berttokenizer
        else:
            self.bert = AutoTokenizer.from_pretrained("bert-base-uncased")
            # print(bert_tokenizer(teststring,return_tensors="pt",add_special_tokens=False)["input_ids"][0])
            # BERT -> 101 = CLS, 102 = SEP
        
        
        
        if robertatokenizer!=None:
            self.roberta = robertatokenizer
        else:
        
            self.roberta = AutoTokenizer.from_pretrained("roberta-base")
            # print(roberta_tokenizer(teststring,return_tensors="pt",add_special_tokens=False)["input_ids"][0])
            # ROBERTA -> 0 = CLS, 2 = SEP
        
        
        if cliptokenizer!=None:
            self.clip = cliptokenizer
        else:
            self.clip = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            # print(clip_tokenizer(text = teststring,return_tensors="pt",add_special_tokens=False)["input_ids"][0])
            # CLIP -> 49406 = CLS     49407 = SEP. Or so it would appear.

        
        
        if data2vectokenizer!=None:
            self.data2vec = data2vectokenizer
        else:
            self.data2vec = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
            # (data2vec_tokenizer(teststring, return_tensors="pt",add_special_tokens=False)["input_ids"][0])
            # data2vec -> 0 = CLS,   2 = SEP. or so it would appear.
        
        

        if vilttokenizer!=None:
            self.vilt = vilttokenizer
        else:
            vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm") 
            self.vilt = vilt_processor.tokenizer # this should operate the same as any tokenizer since we extract the text component.
            
            # we don't KNOW which berttokenizerFAST it is... but at least the tokenizer and image processor in this case is separate.
            # vilt_tokenizer = vilt_method(setviltimage,vilt_processor)
            # print(vilt_tokenizer(teststring,return_tensors="pt", add_special_tokens=False)["input_ids"][0])
            # VILT -> CLS ->101, 102 = SEP
            
        
        
        if vilberttokenizer!=None:
            self.vilbert = vilberttokenizer
        else:
            self.vilbert = AutoTokenizer.from_pretrained("bert-base-uncased")
        # print(vilbert_tokenizer(teststring, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        # BERT -> 101 = CLS, 102 = SEP
        
        
        
        
        if bliptokenizer!=None:
            self.blip = bliptokenizer
        else:
            self.blip = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base").tokenizer
        # print(blip_processor(text=teststring, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        # BLIP -> 101= CLS, 102 = SEP
        
        
        
        if blip2tokenizer!=None:
            self.blip2 = blip2tokenizer
        else:
            self.blip2 = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b").tokenizer
        # print(blip2_processor(text=teststring, return_tensors="pt", add_special_tokens=False)["input_ids"][0])
        # BLIP 2 -> 2 = CLS, 4 = SEP
        
        if gpt_neotokenizer!=None:
            self.gptneo = gpt_neotokenizer
        else:
            self.gptneo = self.texttokenizer = GPT2Tokenizer.from_pretrained("storage-temp/gpt-neo-2.7B-singlish",use_auth_token="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")
        
        
        self.reference = {
                    "bert": self.bert,
                    "roberta":self.roberta,
                    "clip": self.clip,
                    "data2vec":self.data2vec,
                    "vilt": self.vilt,
                    "vilbert":self.vilbert,
                    "blip": self.blip,
                    "blip2":self.blip2,
                    "gpt_neo_SINGLISH": self.gptneo,
                }
    
    
    def __call__(self,input_text,includespecialtokens=False):
        # by default, special tokens are not included.
    
        output_dict = {
                "bert": self.bert(input_text,return_tensors="pt",add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "roberta": self.roberta(input_text,return_tensors="pt",add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "clip": self.clip(text = input_text,return_tensors="pt",add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "data2vec": self.data2vec(input_text, return_tensors="pt",add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "vilt": self.vilt(input_text,return_tensors="pt", add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "vilbert": self.vilbert(input_text, return_tensors="pt", add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "blip": self.blip(text=input_text, return_tensors="pt", add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "blip2": self.blip2(text=input_text, return_tensors="pt", add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "gpt_neo_SINGLISH": self.gptneo(text=input_text, return_tensors="pt", add_special_tokens=includespecialtokens)["input_ids"][0].tolist(),
                "input_text":input_text
            }
        return output_dict



class dataset_information_generation_class():

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

    
    
    def __init__(self, importfile=None, dumpdir = None,target_tokeniser=False,approved_images=[],verbose=False,use_templates=False,template_dir="templates"):
        if dumpdir==None:
            dumpdir = os.path.join("parse_annotated_results","raw_data_jsons","image_dir")
        if importfile==None:
            importfile = os.path.join("parse_annotated_results","ZZZZ_completed_processed_fullabels_shaun.json")
        with open(importfile,"r",encoding="utf-8") as opened_file:
            all_items = json.load(opened_file)

        relationships_counter = {}
        textcounter = {}
        entitycounter = {}
        self.savelist = []
        meme_creatorpresence = False
        for imagekey in all_items:
            if approved_images:
                if not imagekey in approved_images: # skip non approved images if an approved list has been provided.
                    continue
                    
            with Image.open(os.path.join(dumpdir,imagekey)) as checkimage: # conveniently causes an error if the image doesn't exist.
                ref_image_size = checkimage.size
            
            
            boxes_answer = {}
            
            
            equivalents = all_items[imagekey]["equivalent_ids"]
            # all_items[imagekey]["relationmapper"]
            
            span_registration = {} # we will keep a separate dictionary for accounting purposes.
            textboxcounter = 0
            
            for textboxname in all_items[imagekey]["tokenised_strings"]:
                if not textboxname in span_registration:
                    span_registration[textboxname] = {}
                if not textboxname in boxes_answer:
                    boxes_answer[textboxname] = {}
                
                for tokeniser in all_items[imagekey]["tokenised_strings"][textboxname]:
                    if tokeniser=="input_text":
                        continue
                    boxes_answer[textboxname][tokeniser] = []
                    span_registration[textboxname][tokeniser] = []
                    for k in range(len(all_items[imagekey]["tokenised_strings"][textboxname][tokeniser])+2): # include Special characters in the front and back.
                        # boxes_answer[textboxname][tokeniser].append([0,0,0]) # for version with a span transcendence flag
                        boxes_answer[textboxname][tokeniser].append([0,0])
                        span_registration[textboxname][tokeniser].append(False)
                
                textboxcounter+=1

            seen_entitynum = {}
            unique_ids = set()
    
    
    
            # Rejects entities that span several lines
            # print(all_items[imagekey]["entity_spans_labels"].keys())
            # if set()
            manual_reject_seen_entities = set()  # Detect instances of multiple entities spanning several lines.
            # we will leave code that allows for handling multi-span entities in. However, they will be removed from the dataset. (Scope issue)
            sample_ignore_flag = False
            
            for entityids in all_items[imagekey]["entity_spans_labels"]: # we check
                if all_items[imagekey]["entity_spans_labels"][entityids]=="MEME_CREATOR":
                    continue # meme creator instance. ignore.
                if all_items[imagekey]["entity_spans_labels"][entityids]["EntityNum"] in manual_reject_seen_entities:
                    sample_ignore_flag = True
                manual_reject_seen_entities.add(all_items[imagekey]["entity_spans_labels"][entityids]["EntityNum"])
                
            if sample_ignore_flag:
                if verbose:
                    print(all_items[imagekey]["entity_spans_labels"])
                    print("Ignored a sample (Contains single entity spanning multiple lines):",imagekey)
                continue
            ###############################################################################
            
            
            
            

            for entityids in all_items[imagekey]["entity_spans_labels"]: # we check
                seen_entity = False
                if all_items[imagekey]["entity_spans_labels"][entityids]=="MEME_CREATOR":
                    meme_creatorpresence = entityids
                    unique_ids.add(entityids)
                    continue # not span
                    
                if not entityids in unique_ids:
                    unique_ids.add(entityids)

                
                origin_box = all_items[imagekey]["entity_spans_labels"][entityids]["box"]
                entity_number = all_items[imagekey]["entity_spans_labels"][entityids]["EntityNum"]
                
                
                
                if entity_number in seen_entitynum:
                    seen_entity=True
                    seen_entitynum[entity_number].append(entityids)
                else:
                    seen_entitynum[entity_number] = []
                    seen_entitynum[entity_number].append(entityids) 
                    # it also registers the link between an entity number and an id.
                    # everyone who shares the same entity number are linked here
                    # this one documents if an entity spans multiple different input spans.
                    
                
                    
                for tokeniser in all_items[imagekey]["entity_spans_labels"][entityids]:
                    if tokeniser=="box" or tokeniser=="EntityNum" or tokeniser=="ACTUAL_TEXT":
                        continue
                    # Now we start loading the expected labels.
                    textboxname = all_items[imagekey]["entity_spans_labels"][entityids]["box"]
                    
                    if sum(all_items[imagekey]["entity_spans_labels"][entityids][tokeniser])==1:  # Case where the entire entity is contained in a SINGLE token
                        for idx_answer in range(len(all_items[imagekey]["entity_spans_labels"][entityids][tokeniser])):
                            if all_items[imagekey]["entity_spans_labels"][entityids][tokeniser][idx_answer]==1:
                                if seen_entity:
                                    raise ValueError("Span transcendence sample occurred. This should not happen.")
                                    boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,1,1] # singular spanner.. has a flag to continue.
                                    span_registration[textboxname][tokeniser][idx_answer+1] = entityids

                                else:
                                    # boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,1,0] # singular spanner # version where span transcendence can occur
                                    boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,1] # singular spanner
                                    span_registration[textboxname][tokeniser][idx_answer+1] = entityids
                        
                    else: # case where entity is NOT contained in a single token
                        _idxbegun = False
                        for idx_answer in range(len(all_items[imagekey]["entity_spans_labels"][entityids][tokeniser])):
                            if all_items[imagekey]["entity_spans_labels"][entityids][tokeniser][idx_answer]==1:
                                if not _idxbegun:
                                    _idxbegun=True 
                                    if seen_entity:
                                        raise ValueError("Span transcendence sample occurred. This should not happen.")
                                        boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,0,1] # begin the start of the span,with flag
                                        span_registration[textboxname][tokeniser][idx_answer+1] = entityids
                                    else:
                                        # boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,0,0] # begin the start of the span. # version where span transcendence can occur
                                        boxes_answer[origin_box][tokeniser][idx_answer+1] = [1,0] # begin the start of the span. 
                                        span_registration[textboxname][tokeniser][idx_answer+1] = entityids
                                else:
                                    pass # do nothing...
                                    # boxes_answer[origin_box][tokeniser][idx_answer+1] = [0,0,0] # Center of the span
                            else:
                                if _idxbegun:
                                    _idxbegun=False # turn off flag
                                    if seen_entity:
                                        raise ValueError("Span transcendence sample occurred. This should not happen.")
                                        boxes_answer[origin_box][tokeniser][idx_answer] = [0,1,1] # end of the span,with flag 
                                        span_registration[textboxname][tokeniser][idx_answer] = entityids
                                    else:
                                        # boxes_answer[origin_box][tokeniser][idx_answer] = [0,1,0] # end of the span. # version where span transcendence can occur
                                        boxes_answer[origin_box][tokeniser][idx_answer] = [0,1] # end of the span.
                                        span_registration[textboxname][tokeniser][idx_answer] = entityids
                                    # print(all_items[imagekey]["entity_spans_labels"][entityids]["ACTUAL_TEXT"],entityids,tokeniser,origin_box,boxes_answer[origin_box][tokeniser])
                                    break # completed span. No spans are non-continuous (except between different statements)
                                else:
                                    # do nothing.
                                    # boxes_answer[origin_box][tokeniser][idx_answer+1] = [0,0,0] # not part of the span
                                    pass
                        
                        if _idxbegun:            # after all loops. catch a case where the last token is actually the last of that entity.
                            if seen_entity:
                                raise ValueError("Span transcendence sample occurred. This should not happen.")
                                boxes_answer[origin_box][tokeniser][idx_answer+1] = [0,1,1] # end of the span,with flag
                                span_registration[textboxname][tokeniser][idx_answer+1] = entityids
                            else:
                                boxes_answer[origin_box][tokeniser][idx_answer+1] = [0,1] # end of the span.
                                span_registration[textboxname][tokeniser][idx_answer+1] = entityids
                            # print(all_items[imagekey]["entity_spans_labels"][entityids]["ACTUAL_TEXT"],entityids,tokeniser,origin_box,boxes_answer[origin_box][tokeniser])

                            
                            
                                # +1 because there's the CLS or start token at the start.
                if not "ACTUAL_TEXT" in boxes_answer[origin_box]:
                    boxes_answer[origin_box]["ACTUAL_TEXT"] = []
                boxes_answer[origin_box]["ACTUAL_TEXT"].append(all_items[imagekey]["entity_spans_labels"][entityids]["ACTUAL_TEXT"])
                    
                    # completed marking out a SPAN in a textbox.
            # produced the following:
            boxes_answer # [which box][tokeniser][labels idx]
            # this is the thing we backprop with.
            
            span_registration #[which box][tokeniser][entityid? idx] # since we have no overlaps. 
            # span_registration can be used to track if the entirety of a particular span was CORRECT.
                    
            if not len(list(seen_entitynum.keys())) in entitycounter:
                entitycounter[len(list(seen_entitynum.keys()))] = 0
            entitycounter[len(list(seen_entitynum.keys()))] +=1
            
            if not textboxcounter in textcounter:
                textcounter[textboxcounter] = 0
            textcounter[textboxcounter] += 1
            
            
            simplified_equivalents = []
            for equal_set in equivalents: # tells you which 2 pairs are equivalent. Highlighting even one of them is sufficient. They can then replace each other.
                holder_set = []
                for identification_entity in equal_set:
                    holder_set.append(identification_entity[0])
                simplified_equivalents.append(holder_set) # we only need one of the two of these ids to be reported. 
                
                
            
            relationships = all_items[imagekey]["relationmapper"]
            
            

            
            # Numerical version of reference, + Mirroring.
            numerical_relationships = {}
            # print("-BEFORE-")
            # print(relationships)
            # print("-----------")
            rln_holder = {}
            # input()
            
            for relation in relationships:
                toadd = []
                for equivalent1,equivalent2 in simplified_equivalents:
                    for pairwise in relationships[relation]:
                        if pairwise[0] == equivalent1:
                            toadd.append([equivalent2, pairwise[1]])
                        if pairwise[1] == equivalent1:
                            toadd.append([pairwise[0],equivalent2])
                        
                        if pairwise[0] == equivalent2:
                            toadd.append([equivalent1, pairwise[1]])
                        if pairwise[1] == equivalent2:
                            toadd.append([pairwise[0],equivalent1])
                relationships[relation].extend(toadd) # ensure you only add once by processing everything for matches first.
            
            # print("-AFTER SIMILAR-")
            # print(relationships)
            # print("-----------")    
            # input()
            
            
            trig = False
            for relation in list(relationships.keys()): # this ensures we won't be using a non constant number of "keys".
                # i.e on relationship mirror, we don't call the newly created mirrored versions.
                if relation in self.relationship_mirror:
                    trig = True
                    new_rn = self.relationship_mirror[relation]
                    if not new_rn in relationships:
                        relationships[new_rn] = []
                    rln_holder[new_rn] = []
                    for pairwise in relationships[relation]:
                        rln_holder[new_rn].append([pairwise[1],pairwise[0]]) # inversion triggers.
                    for pairwise in relationships[new_rn]:
                        rln_holder[new_rn].append([pairwise[1],pairwise[0]]) # inversion triggers.
                    
                    
                    new_rn_numerical_relation_mapping = self.remapped_actionables_dict[new_rn]
                    numerical_relationships[new_rn_numerical_relation_mapping] = rln_holder[new_rn]
                
            relationships.update(rln_holder)
              
            for relation in list(relationships.keys()): # this ensures we won't be using a non constant number of "keys".        
                numerical_relation_mapping = self.remapped_actionables_dict[relation]
                numerical_relationships[numerical_relation_mapping] = relationships[relation]
            # if trig:
                # print(relationships)
                # input()
            # numerical version of relationships.
            
            # Generate the NULL variations using the list of IDs that are present in the annotation.
            general_map = {}
            
            id_mapped = set()
            
            for relation in relationships:
                for pairwise in relationships[relation]:
                    if not pairwise[0] in general_map:
                        general_map[pairwise[0]] = set()
                    if not pairwise[1] in general_map:
                        general_map[pairwise[1]] = set()
                    id_mapped.add(pairwise[0])
                    id_mapped.add(pairwise[1])
                    
                    general_map[pairwise[0]].add(pairwise[1])
            
            
            floating_entity = False       # Floating entity detection (REMOVAL?)
            for id_seen in list(unique_ids):
                if not id_seen in id_mapped:
                    seen_equal = False
                    for pairitem in equivalents:
                        if pairitem[0][0]==id_seen or pairitem[1][0]==id_seen:
                            seen_equal=True
                    if not seen_equal:
                        if verbose:
                            print(equivalents)
                            print(id_seen)
                        floating_entity = True
            if floating_entity:
                if verbose:
                    print("-"*30)
                    print("ids_seen",id_seen)
                    print("unique ids", unique_ids)
                    print(relationships)
                    print("FLOATING ENTITY IN ", imagekey, " -> Has an entity that has no relation attached to it. (pointless entity)")
                    print("-"*30)
                    # input()
                continue # skip this image...
                # input()
            ###############################################################################
            
            
            for unique in unique_ids:
                if not unique in general_map.keys():
                    general_map[unique] = set()
            
            relationships["NULL"] = []
            
            
            for source in general_map:
                for unique in unique_ids:
                    if all_items[imagekey]["entity_spans_labels"][unique]=="MEME_CREATOR": # target should never be meme creator.
                        continue
                    if not unique in general_map[source] and source!=unique:
                        equivalent_detected = False
                        for equal_pair in equivalents:
                            if source in equal_pair and unique in equal_pair:
                                equivalent_detected = True
                        
                        if not equivalent_detected: 
                        # don't test the equivalents and against self, ban  <ENTITY> -> MEME CREATOR from being constructed since we don't have samples of something having an opinion towards meme creator.
                            relationships["NULL"].append([source,unique])
            

            
            
            if not relationships["NULL"]:
                # all entities are directly related to each other no matter what.
                del relationships["NULL"]
            else:
                numerical_relationships[8] = relationships["NULL"]


            # print(numerical_relationships) # generate inverted method
            inverted_numerical_relationships = {}
            for k in numerical_relationships:
                for pairinstance in numerical_relationships[k]:
                    if not tuple(pairinstance) in inverted_numerical_relationships:
                        inverted_numerical_relationships[tuple(pairinstance)] = []
                    inverted_numerical_relationships[tuple(pairinstance)].append(k)
                
            
                    
            # print("-AFTER-")
            # print(relationships)
            # print("----------")
            # pprint.pprint(numerical_relationships)
            # print(inverted_numerical_relationships)
            # for item in inverted_numerical_relationships:
                # if len(inverted_numerical_relationships[item])>1:
                    # input()
            # print("---------------------")
            # input()



            for i in relationships: # relationships counter
                if not i in relationships_counter:
                    relationships_counter[i] = 0
                relationships_counter[i] += len(relationships[i])

            
            if target_tokeniser:
                for originbox in boxes_answer:
                    for listed_tokeniser in list(boxes_answer[originbox].keys()):
                        if not listed_tokeniser in target_tokeniser:
                            del boxes_answer[originbox][listed_tokeniser]
                    for listed_tokeniser in list(span_registration[originbox].keys()):
                        if not listed_tokeniser in target_tokeniser:
                            del span_registration[originbox][listed_tokeniser]
            
            all_TRUE_entities = []    
            for entityid in all_items[imagekey]["entity_spans_labels"]:
                if all_items[imagekey]["entity_spans_labels"][entityid]=="MEME_CREATOR":
                    all_TRUE_entities.append([entityid,None,"MEME_CREATOR"])
                else:   
                    all_TRUE_entities.append([entityid,all_items[imagekey]["entity_spans_labels"][entityid]["box"],all_items[imagekey]["entity_spans_labels"][entityid]["ACTUAL_TEXT"]])
            
            singleinstance = {
                    "text_locs":all_items[imagekey]["textbox_locations"],
                    "correct_answers":boxes_answer,
                    "span_answer":span_registration,
                    "equivalent_entities":simplified_equivalents,
                    "relationships_read":relationships,
                    "relationship_num":numerical_relationships,
                    # "inverted_numerical_relationships":inverted_numerical_relationships,
                    "source_image":imagekey,
                    "image_addr":os.path.join(dumpdir,imagekey),
                    "meme_creator":meme_creatorpresence,
                    "tokenised_strings": all_items[imagekey]["tokenised_strings"],
                    "actual_entities": all_TRUE_entities,
                    }
            
            if use_templates:
                singleinstance["source_image"] = os.path.join(template_dir, self.map_templates_names[all_items[imagekey]["archetype"]])
                    
            # pprint.pprint(singleinstance)
            # print(numerical_relationships)
            # print(inverted_numerical_relationships)
            # input()
            self.savelist.append(singleinstance)
    
        
        print("Entity Counter:",entitycounter)
        print("Relationships:",relationships_counter)
        print("Text:",textcounter)
        print(self.remapped_actionables_dict)
        self.relationships_counter = relationships_counter
        # input()
     
    def dump_dataset(self,targetfile = "final_dataset.json"):
        with open(targetfile,"w",encoding="utf-8") as datasetpeekfile:
            json.dump(self.savelist,datasetpeekfile,indent=4)

  

if __name__=="__main__":
    targetfile = "SGMEMES_labelout_Annotated_Relation_completed.json"
    straighten(targetfile)
    target_savefile = "SGMEMES_PROCESSED_COMPLETED_LABELS.json"
    
    with open(targetfile,"r",encoding="utf-8") as targetfile_opened:
        loaded_json = json.load(targetfile_opened)
    
    errorlist = []
    
    
    
    byte_or_wordpiece = {
        "bert":"wordpiece",
        "blip2":"bytepair",  # it is sentence piece, but i suspect it's bytepair encoding
        "blip":"wordpiece",
        "clip":"bytepair",
        "data2vec":"bytepair",
        "vilt":"wordpiece",
        "vilbert":"wordpiece",
        "roberta":"bytepair",
        "gpt_neo_SINGLISH":"bytepair",
    }
    
    
    
    all_tokenisers = mass_tokenizer()
   
    
        
        
    accepted_instances = {}
    accepted_counter = 0
    rejected_counter = 0
    spell_counter = 0
    
    if os.path.exists(target_savefile):
        with open(target_savefile,"r",encoding="utf-8") as savepointfile:
            accepted_instances = json.load(savepointfile)
        
        

    for labelinstance in loaded_json:
    
        if labelinstance["data"]["backup"]["source_image"] in accepted_instances:
            continue
        rejection_indicator = False

        if labelinstance["annotations"][0]["was_cancelled"]:
            continue

        for annotation_result in labelinstance["annotations"][0]["result"]:  # check rejection, and spelling ticks.
            
            if "choices" in annotation_result["type"]:
                if "REJECT" == annotation_result["value"]["choices"][0]:
                    rejection_indicator=True
                    break
                    
        if rejection_indicator: # rejected sample, be it repetition, or some other major bounding box issue.
            rejected_counter+=1
            continue
        

        
        
        # DATA PROCESSING PORTION.
        # double spaces are CONDENSED to single space by label studio.
        # basically they mutate our data.
        # If there are no links to SAID entity, we should FILTER it.
        # split the words apart and check for MATCHES from there
        # for single letter or words anyway.
        # you'll need to iterate to check for phrase matches.
        # this will be slow though
        
        
        
        
        tokenised_strings = {}
        
        
        for k in range(1,6):
            if labelinstance["data"]["text"+str(k)]: # non empty string.
                tokenised_strings["text"+str(k)] = all_tokenisers(" ".join(labelinstance["data"]["text"+str(k)].split())) # standardise strings.
        


        
        idmatchup = [] # save IDs that have no relation, as they are equivalent.
        relationmapper = {}  # save Relations between IDs.
        entity_matchup_dict = {} #get the exact entity spans and expected labels for each model output.
        
        
        for annotation_result in labelinstance["annotations"][0]["result"]:
            if annotation_result["type"] == "relation":
                if not annotation_result["labels"]:
                    # empty relation so string is equivalent.
                    idmatchup.append((annotation_result["from_id"],annotation_result["to_id"]))
                    # Save the equivalence pair.
                else:
                    if not annotation_result["labels"][0] in relationmapper:
                        relationmapper[annotation_result["labels"][0]] = []
                    relationmapper[annotation_result["labels"][0]].append((annotation_result["from_id"],annotation_result["to_id"]))
                    
                # relationships and equivalence are processed later.
                    
            elif annotation_result["type"] == "labels":
                # Span.
                # print(annotation_result["id"])
                origin_statement = annotation_result["from_name"]
                if annotation_result["value"]["text"]=="MEME_CREATOR":
                    entity_embedded_versions = False
                else:
                    entity_embedded_versions = all_tokenisers(" ".join(annotation_result["value"]["text"].split()))
                
                if annotation_result["to_name"]=="creatortext":
                    entity_matchup_dict[annotation_result["id"]] = "MEME_CREATOR"
                    
                else:
                    full_string_embed = tokenised_strings[annotation_result["to_name"]]
                    for specific_tokeniser in entity_embedded_versions:
                        # print("-"*30)
                        # print("embedtype:",specific_tokeniser)
                        # print("original text")
                        # print(labelinstance["data"][annotation_result["to_name"]])
                        # print("entity text")
                        # print(annotation_result["value"]["text"])
                        # print("full original")
                        # print(full_string_embed[specific_tokeniser])
                        # print("full extract")
                        # print(entity_embedded_versions[specific_tokeniser])
                        if specific_tokeniser =="input_text": # storage. not actually a tokeniser.
                            continue
                        
                        cropped_first = 0
                        cropped_last = 0    
                        if byte_or_wordpiece[specific_tokeniser]=="wordpiece":
                        
                            subseqnum = subsequence_searcher(entity_embedded_versions[specific_tokeniser],full_string_embed[specific_tokeniser])
                            if subseqnum==-1:
                                raise ValueError("MISSING FROM STRING")
                                
                            else:
                                if len(subseqnum)>1:
                                    # print(subseqnum)
                                    # find the START, of the detected item, and the START of the possible ones, select lowest dist.
                                    # in tie, auto select the back one.
                                    apparent_start = annotation_result["value"]["start"]
                                    reconstructed_sliced_target_string = ""
                                    token_vocab_dict = {v: k for k, v in all_tokenisers.reference[specific_tokeniser].get_vocab().items()}
                                    for croppable_part in entity_embedded_versions[specific_tokeniser]:
                                        # print(token_vocab_dict[croppable_part])
                                        reconstructed_sliced_target_string+=token_vocab_dict[croppable_part]
                                    # print("reconstructed_sliced_string:",reconstructed_sliced_target_string)
                                    if specific_tokeniser=="bert":
                                        source_string = labelinstance["data"][annotation_result["to_name"]].lower()
                                    # completelyjoined_substring = " ".join(list(reconstructed_sliced_target_string))
                                    # print(completelyjoined_substring)
                                    all_start_indices = [index for index in range(len(source_string)) if source_string.startswith(reconstructed_sliced_target_string, index)]
                                    currentsmallest = 9999
                                    selected_indice = False
                                    # print("all start indices:",all_start_indices)
                                    # print("reported start:",apparent_start)
                                    for indice_target in all_start_indices:
                                        if abs(apparent_start-indice_target)<currentsmallest:
                                            currentsmallest = abs(apparent_start-indice_target)
                                            selected_indice = indice_target
                                    if selected_indice==False and type(selected_indice)==bool:
                                        # print("errorlisted")
                                        errorlist.append(labelinstance["data"]["backup"]["source_image"])
                                        break
                                    subseqnum = subseqnum[all_start_indices.index(selected_indice)]
                                    # print(subseqnum) # i.e the subseq starts here.
                                    # raise ValueError("Should not occur")
                                else:
                                    subseqnum = subseqnum[0]
                                start = subseqnum
                                end = subseqnum + len(entity_embedded_versions[specific_tokeniser])
                                            
                        else: # byte style.. assuming it doesn't 
                            # first, check the front and back embeds. 
                            #This is of course with an assumption that in our dataset we don't have cases where several words are CONJOINED.
                            
                            # if the entire bracket isn't inside, we can't track it and we're screwed.
                            
                            croppable = entity_embedded_versions[specific_tokeniser]
                            
                            has_valid =False
                            for k in croppable:
                                if k in full_string_embed[specific_tokeniser]:
                                    has_valid=True
                                    break
                            
                            if has_valid: # at least ONE matching entity token is present.
                                
                                
                                if full_string_embed[specific_tokeniser]==croppable:
                                    start = 0
                                    end = len(croppable)
                                else:
                                    # print(all_sequence_extraction(full_string_embed[specific_tokeniser]))
                                    all_sequences_extracted = all_sequence_extraction(full_string_embed[specific_tokeniser])

                                    
                                    
                                    while not tuple(croppable) in all_sequences_extracted:
                                        cropped_first += 1
                                        croppable = croppable[1:]
                                        # print("aftercrop:",croppable)
                                        if cropped_first >=len(entity_embedded_versions[specific_tokeniser]):
                                            raise ValueError("Cropped till end from front. failed. check inputs")
                                            
                                        
                                    
                                    
                                    while not tuple(croppable) in all_sequences_extracted:
                                        cropped_last += 1
                                        croppable = croppable[:1]
                                        # print("aftercrop:",croppable)
                                        if cropped_last >=len(entity_embedded_versions[specific_tokeniser]):
                                            raise ValueError("Cropped till end from back. failed. check inputs")
                                            
                                    
                                    
                                    subseqnum = subsequence_searcher(croppable,full_string_embed[specific_tokeniser])
                                    if subseqnum==-1:
                                        raise ValueError("MISSING FROM STRING")
                                    else:
                                        if len(subseqnum)>1:
                                            # print(subseqnum)
                                            # find the START, of the detected item, and the START of the possible ones, select lowest dist.
                                            # in tie, auto select the back one.
                                            apparent_start = annotation_result["value"]["start"]
                                            reconstructed_sliced_target_string = ""
                                            token_vocab_dict = {v: k for k, v in all_tokenisers.reference[specific_tokeniser].get_vocab().items()}
                                            for croppable_part in croppable:
                                                # print(token_vocab_dict[croppable_part])
                                                reconstructed_sliced_target_string+=token_vocab_dict[croppable_part]
                                            # print("reconstructed_sliced_string:",reconstructed_sliced_target_string)
                                            if specific_tokeniser=="clip":
                                                reconstructed_sliced_target_string = reconstructed_sliced_target_string.replace("</w>","")
                                                source_string = labelinstance["data"][annotation_result["to_name"]].lower()
                                            elif specific_tokeniser=="roberta" or specific_tokeniser=="data2vec" or specific_tokeniser=="blip2":
                                                reconstructed_sliced_target_string = reconstructed_sliced_target_string.replace("Ġ","")
                                                source_string = labelinstance["data"][annotation_result["to_name"]]
                                            else:
                                                source_string = labelinstance["data"][annotation_result["to_name"]]
                                            # completelyjoined_substring = " ".join(list(reconstructed_sliced_target_string))
                                            all_start_indices = [index for index in range(len(source_string)) if source_string.startswith(reconstructed_sliced_target_string, index)]
                                            currentsmallest = 9999
                                            selected_indice = False
                                            # print("all start indices:",all_start_indices)
                                            # print("reported start:",apparent_start)
                                            for indice_target in all_start_indices:
                                                if abs(apparent_start-indice_target)<currentsmallest:
                                                    currentsmallest = abs(apparent_start-indice_target)
                                                    selected_indice = indice_target
                                            if not selected_indice:
                                                errorlist.append(labelinstance["data"]["backup"]["source_image"])
                                                break    
                                            subseqnum = subseqnum[all_start_indices.index(selected_indice)]
                                            # print(subseqnum) # i.e the subseq starts here.
                                            # raise ValueError("Should not occur")
                                        else:
                                            subseqnum = subseqnum[0]
                                        start = subseqnum - cropped_first
                                        end = start + len(croppable) + cropped_last + cropped_first # +1 because of how indexing works in python.
                                    # print("cropped ver:")
                                    # print(croppable)
                                    # print("cropped_first",cropped_first)
                                    # print("cropped_last",cropped_last)
                                    # print("start",start)
                                    # print("end",end)
                                    # print("what we identify inside.")
                                    # print(full_string_embed[specific_tokeniser][start:end])

                            
                            
                            else: # there are NO Matching entity tokens. We will default to BRUTE force string match
                                # all_sequences_extracted = all_sequence_extraction(full_string_embed[specific_tokeniser])
                                
                                new_dict =  {v:k for k,v in all_tokenisers.roberta.get_vocab().items()} 
                                # print(labelinstance["data"][annotation_result["to_name"]].split())
                                all_possible_sequences = all_sequence_extraction(labelinstance["data"][annotation_result["to_name"]].split())
                                # print(all_possible_sequences)
                                word_matched=False
                                lendict = {}
                                
                                recall_dict = {}
                                
                                
                                # strict token based word match. i.e we look for what substring combinations can possibly map to the token seen.
                                
                                for subsequence_string in all_possible_sequences:
                                    subsequence_tokenised = all_tokenisers(" ".join(subsequence_string))[specific_tokeniser]
                                    recall_dict[subsequence_string] = subsequence_tokenised
                                    # print("-"*30)
                                    # print(tuple(entity_embedded_versions[specific_tokeniser]))
                                    # print(subsequence_tokenised)
                                    # print(" ".join(subsequence_string))
                                    # print(tuple(subsequence_tokenised))
                                    # print(tuple(entity_embedded_versions[specific_tokeniser]))
                                    if tuple(entity_embedded_versions[specific_tokeniser]) == tuple(subsequence_tokenised):
                                        continue # not what we're looking for. it isn't the direct match.
                                        
                                    for embedded_entity_part in tuple(entity_embedded_versions[specific_tokeniser]):
                                        
                                        if embedded_entity_part in subsequence_tokenised:
                                            word_matched = True
                                            if not len(list(subsequence_tokenised)) in lendict:
                                                lendict[len(list(subsequence_tokenised))] = []
                                            if not (subsequence_string) in lendict[len(list(subsequence_tokenised))]:
                                                lendict[len(list(subsequence_tokenised))].append((subsequence_string))
                                
                                            
                                
                                
                                
                                
                                
                                
                                
                                if not word_matched:
                                    # no choice. Try brute forcing.
                                    # this is a special case, like 'IS THIS GOD', vs 'GOD' "
                                    if annotation_result["value"]["text"] in labelinstance["data"][annotation_result["to_name"]].split() \
                                    and len(labelinstance["data"][annotation_result["to_name"]].split()) == len(full_string_embed[specific_tokeniser]):
                                        start = labelinstance["data"][annotation_result["to_name"]].split().index(annotation_result["value"]["text"])
                                        end = start+1
                                        # we basically conclude 
                                        #1) annotation text is INDEED within the actual text
                                        #2) the tokenised version is a one to one mapping.
                                        #3) this runs on the assumption that it's a SINGLE token. Multitoken can't be solved easily.
                                        # print("force selected start", start)
                                        # print("Selected string part:",labelinstance["data"][annotation_result["to_name"]].split()[start])
                                        
                                    elif labelinstance["data"][annotation_result["to_name"]].split()[-1]== annotation_result["value"]["text"]:
                                        
                                        start = len(labelinstance["data"][annotation_result["to_name"]].split())-1
                                        end = start+1
                                        # print("low accuracy was forced.")
                                        # example: "NOW I HATE YOU" and "YOU". 
                                        
                                        
                                    else:
                                        raise ValueError("Apparently not in any of the subsequences, nor bruteforceable.")
                                else:
                                    # print(lendict)
                                    
                                    lowestkey = min(list(lendict.keys()))
                                    
                                    if len(lendict[lowestkey])>1:
                                        # print(lendict[lowestkey])
                                        raise ValueError("More than one possibility. Suggestion: Manual Override.")
                                    
                                    
                                    targetted_string = lendict[lowestkey][0]  # since there is only one entry.
                                    
                                    parts_of_text = annotation_result["value"]["text"].split()
                                    
                                    # print("Original label:",parts_of_text)
                                    # print("Adjusted Label:",targetted_string)
                                    
                                    croppable = recall_dict[targetted_string] # remap to the CLOSEST possible pair
                                    
                                    all_sequences_extracted = all_sequence_extraction(full_string_embed[specific_tokeniser])
                                    
                                    # print(all_sequences_extracted)
                                    while not tuple(croppable) in all_sequences_extracted:
                                        cropped_first += 1
                                        croppable = croppable[1:]
                                        # print("aftercrop:",tuple(croppable))
                                        if cropped_first >len(entity_embedded_versions[specific_tokeniser]):
                                            raise ValueError("Cropped till end from front. failed. check inputs")
                                            
                                        
                                    
                                    
                                    while not tuple(croppable) in all_sequences_extracted:
                                        cropped_last += 1
                                        croppable = croppable[:1]
                                        print("aftercrop:",croppable)
                                        if cropped_last >len(entity_embedded_versions[specific_tokeniser]):
                                            raise ValueError("Cropped till end from back. failed. check inputs")
                                                
                                                
                                    subseqnum = subsequence_searcher(croppable,full_string_embed[specific_tokeniser])
                                    if subseqnum==-1:
                                        raise ValueError("MISSING FROM STRING")
                                    else:
                                        if len(subseqnum)>1:
                                            # print(subseqnum)
                                            # find the START, of the detected item, and the START of the possible ones, select lowest dist.
                                            # in tie, auto select the back one.
                                            apparent_start = annotation_result["value"]["start"]
                                            reconstructed_sliced_target_string = ""
                                            token_vocab_dict = {v: k for k, v in all_tokenisers.reference[specific_tokeniser].get_vocab().items()}
                                            for croppable_part in croppable:
                                                print(token_vocab_dict[croppable_part])
                                                reconstructed_sliced_target_string+=token_vocab_dict[croppable_part]
                                            # print("reconstructed_sliced_string:",reconstructed_sliced_target_string)
                                            if specific_tokeniser=="clip":
                                                reconstructed_sliced_target_string = reconstructed_sliced_target_string.replace("</w>","")
                                                source_string = labelinstance["data"][annotation_result["to_name"]].lower()
                                            elif specific_tokeniser=="roberta" or specific_tokeniser=="data2vec" or specific_tokeniser=="blip2":
                                                reconstructed_sliced_target_string = reconstructed_sliced_target_string.replace("Ġ","")
                                                source_string = labelinstance["data"][annotation_result["to_name"]]
                                            else:
                                                source_string = labelinstance["data"][annotation_result["to_name"]]
                                            # completelyjoined_substring = " ".join(list(reconstructed_sliced_target_string))
                                            all_start_indices = [index for index in range(len(source_string)) if source_string.startswith(reconstructed_sliced_target_string, index)]
                                            currentsmallest = 9999
                                            selected_indice = False
                                            # print("all start indices:",all_start_indices)
                                            # print("reported start:",apparent_start)
                                            for indice_target in all_start_indices:
                                                if abs(apparent_start-indice_target)<currentsmallest:
                                                    currentsmallest = abs(apparent_start-indice_target)
                                                    selected_indice = indice_target
                                                
                                            if not selected_indice:
                                                errorlist.append(labelinstance["data"]["backup"]["source_image"])
                                                break    

                                            subseqnum = subseqnum[all_start_indices.index(selected_indice)]
                                            # print(subseqnum) # i.e the subseq starts here.
                                            # raise ValueError("Should not occur")
                                        else:
                                            subseqnum = subseqnum[0]
                                        start = subseqnum - cropped_first
                                        end = start + len(croppable) + cropped_last + cropped_first # +1 because of how indexing works in python.
                                # print("cropped ver:")
                                # print(croppable)
                                # print("cropped_first",cropped_first)
                                # print("cropped_last",cropped_last)
                                # print("start",start)
                                # print("end",end)
                                # print("what we identify inside.")
                                # print(full_string_embed[specific_tokeniser][start:end])
                                
                        
                        expected_labels = [0]*len(full_string_embed[specific_tokeniser])
                        # print(expected_labels)
                        # print(start,end)
                        for flipped_idx in range(start,end): # end point is omitted...
                            # print(flipped_idx)
                            expected_labels[flipped_idx] = 1
                        if not annotation_result["id"] in entity_matchup_dict:
                            entity_matchup_dict[annotation_result["id"]] = {"box":annotation_result["to_name"]}
                        
                        # print("expected labels:",expected_labels)
                        # print("full string:",full_string_embed[specific_tokeniser])
                        
                        if sum(expected_labels)!=len(full_string_embed[specific_tokeniser][start:end]):
                            raise ValueError("something went wrong with the way we count our expected labels")
                            
                        entity_matchup_dict[annotation_result["id"]][specific_tokeniser] = expected_labels
                        entity_matchup_dict[annotation_result["id"]]["EntityNum"] = int(annotation_result["value"]["labels"][0].replace("Entity","").strip())
                        entity_matchup_dict[annotation_result["id"]]["ACTUAL_TEXT"] = annotation_result["value"]["text"]
                        # input()

                
                
        original_textboxstrings = {}
        
        
        final_idmatchup = []
        # print(idmatchup)
        # print(entity_matchup_dict)
        for idpair in idmatchup:
            single_pair = []
            for singular_id in idpair:
                if entity_matchup_dict[singular_id]=="MEME_CREATOR":
                    errorlist.append([labelinstance["data"]["true_filename"],"MEME_CREATOR LINKED TO INTERNAL ENTITY NEEDLESSLY"])
                    break
                single_pair.append((singular_id,entity_matchup_dict[singular_id]["EntityNum"]))
            final_idmatchup.append(single_pair)

        
            
        
        textbox_locations = {}
        for textbox_accepted in tokenised_strings:
            textbox_locations[textbox_accepted] = [ {"height":labelinstance["data"]["rect"+textbox_accepted[-1]+"height"],"width":labelinstance["data"]["rect"+textbox_accepted[-1]+"y"],"x":labelinstance["data"]["rect"+textbox_accepted[-1]+"x"],"y":labelinstance["data"]["rect"+textbox_accepted[-1]+"y"]}, labelinstance["data"][textbox_accepted] ]
            


        final_dataset_output = {
            "equivalent_ids":final_idmatchup, # save IDs that have no relation, as they are equivalent.
            "relationmapper":relationmapper,  # save Relations between IDs.
            "entity_spans_labels":entity_matchup_dict, #get the exact entity spans and expected labels for each model output.      
            "tokenised_strings":tokenised_strings, # all tokenised versions of the strings.
            "filename":labelinstance["data"]["backup"]["source_image"],
            "textbox_locations":textbox_locations
        }
        
        accepted_instances[labelinstance["data"]["backup"]["source_image"]] = final_dataset_output # something...?
        
        accepted_counter+=1
        if random.randint(0,10)>9: # rng based saving.
            with open(target_savefile,"w",encoding="utf-8") as completefile:
                    json.dump(accepted_instances,completefile,indent=4)
        
    with open(target_savefile,"w",encoding="utf-8") as completefile:
        json.dump(accepted_instances,completefile,indent=4)
    print("spell_counter",spell_counter)
    print("accepted_counter", accepted_counter)
    print("rejected_counter",rejected_counter)
    
        
    print("success")
    with open(target_savefile,"w",encoding="utf-8") as completefile:
        json.dump(accepted_instances,completefile,indent=4)
    print("error files:",errorlist)
    
    
    s = dataset_information_generation_class(target_savefile,dumpdir=os.path.join("TDMEMES","TD_Memes"),verbose=True)
    s.dump_dataset(targetfile = "SGMEMES_dataset_processed_final.json")
    
import json
import pprint
import math
import json
import os
import datetime
import urllib.parse
import random
import PIL
import pprint
from PIL import Image
from transformers import AutoTokenizer, ViltProcessor, AutoProcessor, Blip2Processor, GPT2Tokenizer




"""
Don't forget to set the relevant environmental variables for Labelstudio to LOAD your images.
# in mac it's not set, it's some other keyword, export.
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true



|This file is designed to hold the following:

1) A mass tokenizer class, which loads all tokenizers (word only), - Where all contains: BLIP, CLIP, Data2vec, vilt, vilbert, BERT, RoBERTa.
    - While FLAVA is not explicitly utilised here, its text tokeniser is equal BERT
    - Where The following tokenizers are equivalent, though loaded multiple times.
        - 
        - VILT,FLAVA, VILBERT, BLIP uses BERT (Wordpiece)
        
        - RoBERTa uses Bytepair-encoding (unique to self)
        - BLIP2 uses T5's Sentencepiece-encoding (very similar to byte-pair encoding)
        - Data2vec uses Bytepair-encoding (unique to self, cannot be shared with others.)
    __call__ ing  the mass tokenizer class will return the following after it is given an input string:
        output_dict = {
                "bert"
                "roberta"
                "clip"
                "data2vec"
                "vilt"
                "vilbert"
                "blip" 
                "blip2"
                "input_text"
            }
        where each of the items in the dict is the encoded version of the string.
        Note that by default,  includespecialtokens=False 
        
2) Additional Annotation Interface and other setup classes.
"""






class EntityA():
    pass

class EntityB():
    pass

class EntityC():
    pass

class EntityD():
    pass

class EntityE():
    pass

    

master_dict = {
    "Anime-Girl-Hiding-from-Terminator":[4,15],
    "Arthur-Fist":[11],
    "Blank-Nut-Button":[28,29,33],
    "Both-Buttons-Pressed":[8,9,34],
    "Buff-Doge-vs-Cheems":[32],
    "Clown-Applying-Makeup":[22,23],
    "Cuphead-Flower":[12,13],
    "Disappointed-Black-Guy":[10,35],
    "Distracted-Boyfriend":[26],
    "Drake-Hotline-Bling":[1,2],
    "Epic-Handshake":[25],
    "Ew-i-stepped-in-shit":[15,16],
    "Fancy-pooh":[30,31],
    "Feels-Good-Man":[7],
    "Hide-the-Pain-Harold":[3],
    "If-those-kids-could-read-theyd-be-very-upset":[5,6,17],
    "Is-This-A-Pigeon":[27],
    "kermit-window":[11],
    "Moe-throws-Barney":[19,20,36],
    "Mother-Ignoring-Kid-Drowning-In-A-Pool":[21],
    "Mr-incredible-mad":[3,5,17],
    "Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask":[3],
    "Running-Away-Balloon":[24],
    "Skinner-Out-Of-Touch":[5,6],
    "Soyboy-Vs-Yes-Chad":[2,11],
    "Spider-Man-Double":[8,9],
    "Spongebob-Burning-Paper":[4,11],
    "Squidward":[10],
    "Teachers-Copy":[10,35],
    "They-are-the-same-picture":[8,9],
    "This-is-Worthless":[4,11],
    "Tuxedo-Winnie-the-Pooh-grossed-reverse":[1,10],
    "Tuxedo-Winnie-The-Pooh":[2,12],
    "Two-Paths":[10,18,38],
    "Weak-vs-Strong-Spongebob":[12,14,2,1],
    "The-Scroll-Of-Truth":[3,4,5],
    "Left-Exit-12-Off-Ramp":[1,2],
    "Types-of-Headaches-meme":[3,4],
    "This-Is-Brilliant-But-I-Like-This":[1,2],
}



entity_crop_areas = {
    "Anime-Girl-Hiding-from-Terminator":[(0,0,500,272),(0,272,500,544)],
    "Arthur-Fist":[(0,0,666,360)],
    "Blank-Nut-Button":[(0,0,600,110),(250,110,600,446),(0,223,350,436)],
    "Both-Buttons-Pressed":[(0,0,235,354),(235,0,500,354),(0,354,500,730)],
    "Buff-Doge-vs-Cheems":[(0,0,375,325),(375,0,650,325)],
    "Clown-Applying-Makeup":[(0,0,250,133),(0,133,250,266),(0,266,250,399),(0,399,250,520),(250,0,500,532)],
    "Cuphead-Flower":[(250,0,500,250),(250,250,500,500)],
    "Disappointed-Black-Guy":[(0,0,516,250),(0,250,516,500)],
    "Distracted-Boyfriend":[(0,281,350,500),(350,0,560,500),(560,0,750,500)],
    "Drake-Hotline-Bling":[(250,0,500,250),(250,250,500,500),(0,0,250,350)],
    "Epic-Handshake":[(0,250,300,490),(300,220,500,490),(0,0,500,220)],
    "Ew-i-stepped-in-shit":[(0,0,500,405),(0,350,500,700)],
    "Fancy-pooh":[(200,0,500,176),(200,176,500,352),(200,352,500,527)],
    "Feels-Good-Man":[(0,0,541,500)],
    "Hide-the-Pain-Harold":[(0,0,590,500)],
    "If-those-kids-could-read-theyd-be-very-upset":[(0,0,500,280),(0,280,500,505)],
    "Is-This-A-Pigeon":[(0,0,278,300),(278,0,556,300),(0,300,556,490)],
    "kermit-window":[(0,0,700,459)],
    "Moe-throws-Barney":[(0,0,500,284),(0,284,500,568),(0,568,500,842)],
    "Mother-Ignoring-Kid-Drowning-In-A-Pool":[(0,245,209,400),(182,105,316,268),(360,120,500,300),(0,400,500,659)],
    "Mr-incredible-mad":[(0,0,566,295),(0,213,566,422)],
    "Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask":[(0,0,500,525)],
    "Running-Away-Balloon":[(0,0,250,326),(250,0,500,326),(0,326,166,665)],
    "Skinner-Out-Of-Touch":[(0,0,500,356),(0,356,500,706)],
    "Soyboy-Vs-Yes-Chad":[(0,0,366,280),(366,0,733,280)],
    "Spider-Man-Double":[[(0,0,352,345),(352,0,704,354)],[(0,0,280,275),(280,0,560,282)]],
    "Spongebob-Burning-Paper":[(0,0,250,302),(250,0,500,300),(250,300,500,605)],
    "Squidward":[(0,0,500,375),(0,375,500,740)],
    "Teachers-Copy":[(0,0,346,250),(346,0,692,250)],
    "They-are-the-same-picture":[(0,0,255,240),(255,0,500,240),(0,244,500,533)],
    "This-is-Worthless":[(0,0,500,240),(0,280,500,530)],
    "Tuxedo-Winnie-the-Pooh-grossed-reverse":[(320,0,640,243),(320,243,640,487)],
    "Tuxedo-Winnie-The-Pooh":[(296,0,687,250),(296,250,687,500),(0,0,298,252),(0,252,298,500)],
    "Two-Paths":[(0,0,250,290),(250,0,500,290),(0,290,500,490)],
    "Weak-vs-Strong-Spongebob":[(374,0,748,250),(374,250,748,499),(0,0,375,499)],
    "The-Scroll-Of-Truth":[(29,251,255,480),(259,344,494,480)],
    "Left-Exit-12-Off-Ramp":[[(0,88,363,340),(363,88,640,340),(0,447,804,750)],[(0,0,225,225),(225,0,524,225),(0,226,524,480)]],
    "Types-of-Headaches-meme":[(204,230,500,514)],
    "This-Is-Brilliant-But-I-Like-This":[(236,96,500,223),(0,340,224,481),(0,0,500,81)]
}


standard_imagesize_dict = {
    "Anime-Girl-Hiding-from-Terminator":[(500,544)],
    "Arthur-Fist":[(666,375)],
    "Blank-Nut-Button":[(600,446)],
    "Both-Buttons-Pressed":[(500,756)],
    "Buff-Doge-vs-Cheems":[(650,500)],
    "Clown-Applying-Makeup":[(500,532)],
    "Cuphead-Flower":[(500,500)],
    "Disappointed-Black-Guy":[(775,500)],
    "Distracted-Boyfriend":[(750,500)],
    "Drake-Hotline-Bling":[(500,500)],
    "Epic-Handshake":[(697,500)],
    "Ew-i-stepped-in-shit":[(500,707)],
    "Fancy-pooh":[(500,537)],
    "Feels-Good-Man":[(541,500)],
    "Hide-the-Pain-Harold":[(590,500)],
    "If-those-kids-could-read-theyd-be-very-upset":[(500,561)],
    "Is-This-A-Pigeon":[(556,500)],
    "kermit-window":[(700,469)],
    "Moe-throws-Barney":[(500,852)],
    "Mother-Ignoring-Kid-Drowning-In-A-Pool":[(500,659)],
    "Mr-incredible-mad":[(566,441)],
    "Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask":[(500,535)],
    "Running-Away-Balloon":[(500,672)],
    "Skinner-Out-Of-Touch":[(500,713)],
    "Soyboy-Vs-Yes-Chad":[(733,499)],
    "Spider-Man-Double":[(704,354),(560,282)],
    "Spongebob-Burning-Paper":[(500,605)],
    "Squidward":[(500,750)],
    "Teachers-Copy":[(692,500)],
    "They-are-the-same-picture":[(500,560)],
    "This-is-Worthless":[(500,560)],
    "Tuxedo-Winnie-the-Pooh-grossed-reverse":[(640,487)],
    "Tuxedo-Winnie-The-Pooh":[(687,500)],
    "Two-Paths":[(500,500)],
    "Weak-vs-Strong-Spongebob":[(748,499)],
    "The-Scroll-Of-Truth":[(517,499)],
    "Left-Exit-12-Off-Ramp":[(804,767),(524,499)],
    "Types-of-Headaches-meme":[(500,514)],
    "This-Is-Brilliant-But-I-Like-This":[(500,516)],
    }

entity_workdirnames = {
    0: "entityACROP.png",
    1: "entityBCROP.png",
    2: "entityCCROP.png",
    3: "entityDCROP.png",
    4: "entityECROP.png"
}


def obtain_json(target_dir):
    # utility function. Obtains all relevant JSONs containing data we originally pulled.
    all_meme_types = {}
    targetlist = os.listdir(target_dir)
    
    for target in targetlist:
        if not "meme_" in target:
            continue
    
        with open(os.path.join(target_dir,target),"r",encoding="utf-8") as pulled_json_file:
            loaded_img_info = json.load(pulled_json_file)
    
        for i in loaded_img_info:
            all_meme_types[i["filename"]] = target.replace("_labelout.json","").replace("meme_","")
    return all_meme_types




class mass_tokenizer:
    # Designed to hold all the tokenizers together. Used in generating final_dataset json

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


def all_sequence_extraction(a):
    outy = set()
    for n in range(len(a)):
        outy|=set(list(zip(*[a[i:] for i in range(n)])))
    return outy




def straightener():
    items = os.listdir()
    for item in items:
        if ".json" == item[-5:]:
            print("intiating:",item)
            with open(item,"r",encoding="utf-8") as opened_dict_file:
                loaded_dict = json.load(opened_dict_file)
            with open(item,"w",encoding="utf-8") as opened_dict_file:
                json.dump(loaded_dict,opened_dict_file,indent=4)
            print("Completed redump (indent=4).")





def workcrop(path_to_image,targetentityboxes,namedict,meme_name):
    abclist = []
    with Image.open(path_to_image) as im:
        if len(standard_imagesize_dict[meme_name])>1:
            for idx in range(len(standard_imagesize_dict[meme_name])):
                # check template conforming crops.
                print(idx)
                if im.size[0]!=standard_imagesize_dict[meme_name][idx][0] or im.size[1]!=standard_imagesize_dict[meme_name][idx][1] :
                    continue # wrong size.
                print(targetentityboxes[idx])
                for croptarget in range(len(targetentityboxes[idx])):
                    cropped_part = im.crop(targetentityboxes[idx][croptarget])
                    cropped_part.save(namedict[croptarget])
                    abclist.append(namedict[croptarget])
        else:
            for croptarget in range(len(targetentityboxes)):
                    cropped_part = im.crop(targetentityboxes[croptarget])
                    cropped_part.save(namedict[croptarget])
                    abclist.append(namedict[croptarget])
    return abclist
    
    

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



def initial_OCR_labelstudio_generator(data_dir = "raw_data_jsons", data_dir_dumpfile = "image_dir",limiter=150,workingdir = "working_dir",dump_target="label_studio_reference_input_OCR_INITIAL.json"):

    # The First Step.
    # generates a label studio file for OCR.
    # Also saves the bounding boxes for all the texts. 
    # The goal for this annotation is to first combine the texts detected into proper textboxes and do spelling correction first.
    # note that the label studio file given does not allow for bounding box correction. OCR corrections and bounding box corrections are separated.
    # Limiter designates how many to "take" from each file, since we technically pull way more than required.discord
    import easyocr # only import easyocr if we are attempting this. Reduces import requirements.
    
    
    do_boundingboxsaver = True
    

    if not os.path.isdir(workingdir):
        os.mkdir(workingdir)
        
    for idx in entity_workdirnames:
        entity_workdirnames[idx] = os.path.join(workingdir,entity_workdirnames[idx])
    

    all_usedfiles = set()
        
    target_candidates = []
    for i in os.listdir(data_dir):
        if "_labelout.json" in i:
            target_candidates.append([os.path.join(data_dir,data_dir_dumpfile), os.path.join(data_dir,i), i.replace("_labelout.json","").replace("meme_","")])
                        # dump dir for images, label file
                        
                        
    OCRreader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    easyocr_minimum_score = 0.1
    
    valid_sample_counter = {}
    seen_images = set()
    for item in target_candidates:
        proposed_json = []
        valid_sample_counter[item[2]] = 0
        # if not master_dict[item[2]]: # if we didn't indicate relations, don't do yet.
            # continue
        
        with open(item[1],"r",encoding="utf-8") as pulled_file:
            itemstuff = json.load(pulled_file)
        
        totalcounter = 0
        
        for sample in itemstuff:
            totalcounter+=1
            if totalcounter>600:
                break
            filepath = item[0]+"/"+sample["filename"]
            print(filepath)
            
            if not "filename" in sample: # not valid entry.
                print("has no saved file...")
                continue
            if sample["filename"] in all_usedfiles:
                continue
                
            if not "tags" in sample: # continue. Missing tags..
                print("no tags...")
                continue
            if not os.path.exists(filepath):
                # print(filepath)
                # input()
                print("filepath didn't exist...")
                continue # SKIP ENTRY. REMOVED FOR VARIOUS REASONS
            if filepath in seen_images: # duped label.
                print("duped entry.")
                continue
            else:
                seen_images.add(filepath)
                
            
            truename = sample["filename"]
            
            tags = sample["tags"]
            

            print(item[2])
            with Image.open(filepath) as im:
                pulled_image_width, pulled_image_height = im.size
                haspassed=False
                for sizechecker in standard_imagesize_dict[item[2]]:
                    if pulled_image_width==sizechecker[0] and pulled_image_height==sizechecker[1]:
                        haspassed=True
                if not haspassed:
                    continue # don't process this image. it does not conform to normal unedited template dimensions for any within the "approved list.


            complete_image_text_items = []
            all_image_text_proposals = OCRreader.readtext(filepath)
            for detection in all_image_text_proposals:
                textdetected = detection[1]
                confidence = detection[2]
                print(textdetected,confidence)
                if confidence> easyocr_minimum_score:
                    boundingboxcleaner = []
                    for bb_coord in detection[0]:
                        boundingboxcleaner.append([int(bb_coord[0]),int(bb_coord[1])])
                    complete_image_text_items.append([textdetected,boundingboxcleaner])
            
            internalrecorder = {}
            optionlist = []
            counter = 0
            mapperdict_cleanedtextver = {} 
            for completecounter in range(len(complete_image_text_items)):
                internalrecorder[completecounter] = complete_image_text_items[completecounter]
                mapperdict_cleanedtextver[complete_image_text_items[completecounter][0]] = str(counter)+": "+complete_image_text_items[completecounter][0]
                optionlist.append({"value":str(counter)+": "+complete_image_text_items[completecounter][0]}) 
                # adding the counter prevents same strings from having similar dictionary entries
                # it's an issue when parsing results later.
                counter+=1
            

            proposed_dictionary = {"data":{"archetype":item[2],
            # "image":sample["filename"],
            "image":"/data/local-files/?d="+urllib.parse.quote(os.path.join(os.path.abspath(os.getcwd()),filepath)),
            # Sample: Users%5CUser%5CDesktop%5Cmeow%20meow%5CCompiled_images%5C3t986v.jpg
            # "image":os.path.join(item[0],sample["filename"]),
            "true_filename":sample["filename"],
            "tags":tags,
            "alloptions":optionlist,
            "internal_record":internalrecorder,
            # "all_detected_textboxes":alldetected_textlist
            "original_OCRtext": mapperdict_cleanedtextver # label studio won't mark properly..
            }}
            print(proposed_dictionary)
            proposed_json.append(proposed_dictionary)
    
        # with open(item[2]+"_label_studio_reference_output.json","w",encoding="utf-8") as dumpfile:
            # json.dump(proposed_json,dumpfile,indent=4)
    print("items:",len(proposed_json))
    with open(dump_target,"w",encoding="utf-8") as dumpfile:
        json.dump(proposed_json,dumpfile,indent=4)






def OCR_load_results(targetjson,report_types=False):
    # Utilised internally by the generator for the bounding box correction Json.
    
    typecounter = {}
    allresults = {}
    allfiles = set()
    revertlookupdict = {"A_true":["Atext","text"], "B_true":["Btext","text"], "C_true":["Ctext","text"], "D_true":["Dtext","text"], "E_true":["Etext","text"],
            "A":["Aboxes","choices","Atext"],"B":["Bboxes","choices","Btext"],"C":["Cboxes","choices","Ctext"],"D":["Dboxes","choices","Dtext"],"E":["Eboxes","choices","Etext"]}
    
    with open(targetjson,"r",encoding="utf-8") as opened_file:
        loaded_json = json.load(opened_file)
    
    
    for result in loaded_json:
        newresult_dict = {
            "alltext" :"",
            "Atext":"",
            "Aboxes":[],
            "Btext":"",
            "Bboxes":[],
            "Ctext":"",
            "Cboxes":[],
            "Dtext":"",
            "Dboxes":[],
            "Etext":"",
            "Eboxes":[],
        }
        
        
        selected = False
        for item in result["annotations"][0]["result"]:
            if item["from_name"]=="Accept":
                if item["value"]["choices"]==["Accept"]: # only one option but it's saved as a list.
                    selected = True
        
        allfiles.add(result["data"]["true_filename"])

        
        if not selected: # skip entry.
            continue
            
        true_filename = result["data"]["true_filename"]
        internalrecord = result["data"]["internal_record"]
        alloptions = result["data"]["alloptions"]
        # original_OCRmap = {v:k for k,v in result["data"]["original_OCRtext"].items()}
        
        bboxdata = {}
        for item in internalrecord:
            bboxdata[item+": "+internalrecord[item][0]] = internalrecord[item][1:]
        # print(bboxdata)
        # print(internalrecord)
        for item in result["annotations"][0]["result"]:
            # print(item)
            if item["from_name"]=="Accept":
                continue # ignore.
            elif revertlookupdict[item["from_name"]][1]=="choices":
                totaltext = ""
                for selectedbox in item["value"][revertlookupdict[item["from_name"]][1]]:
                    # print(selectedbox)
                    try:
                        bboxinvolved = bboxdata[selectedbox] # get the relevant bounding box.
                    except KeyError:
                        bboxinvolved = internalrecord[selectedbox.split(":")[0]][1]
                        # print(bboxinvolved)
                        # input()
                    newresult_dict[revertlookupdict[item["from_name"]][0]].append(bboxinvolved[0])
                    totaltext+=" "+ selectedbox[3:]
                newresult_dict[revertlookupdict[item["from_name"]][2]] = totaltext.strip()
                #parse as list
            else:
                # is textinput
                # print(item["value"][revertlookupdict[item["from_name"]][1]])
                newresult_dict[revertlookupdict[item["from_name"]][0]] = item["value"][revertlookupdict[item["from_name"]][1]][0]
        # print(newresult_dict)
        newresult_dict["alltext"] = " ".join([newresult_dict["Atext"],newresult_dict["Btext"],newresult_dict["Ctext"],newresult_dict["Dtext"],newresult_dict["Etext"]])
        newresult_dict["filename"] = true_filename
        newresult_dict["archetype"] = result["data"]["archetype"]
        newresult_dict["original_OCRtext"] = result["data"]["original_OCRtext"] 
        allresults[true_filename] = newresult_dict
        if not  result["data"]["archetype"] in typecounter:
            typecounter[result["data"]["archetype"]] = 0
        typecounter[result["data"]["archetype"]] = typecounter[result["data"]["archetype"]] + 1
        

        
    allfiles = list(allfiles)
    
    with open("allfilesusedthusfar.json","w",encoding="utf-8") as used_filesopenfile:
        json.dump(allfiles,used_filesopenfile,indent=4)
    
    
    if report_types:
        pprint.pprint(typecounter)
    return allresults


def clean_strings_processed(targetfile):
    with open(targetfile,"r",encoding="utf-8") as sacfile:
        loaded_json = json.load(sacfile)


    for imagekey in loaded_json:
        for entityspan in loaded_json[imagekey]["entity_spans_labels"]:
            if type(loaded_json[imagekey]["entity_spans_labels"][entityspan])==str:
                continue
            loaded_json[imagekey]["entity_spans_labels"][entityspan]["ACTUAL_TEXT"] = re.sub(' +', ' ',loaded_json[imagekey]["entity_spans_labels"][entityspan]["ACTUAL_TEXT"]).strip()
        for textbox in loaded_json[imagekey]["tokenised_strings"]:
            loaded_json[imagekey]["tokenised_strings"][textbox]["input_text"] = re.sub(' +', ' ',loaded_json[imagekey]["tokenised_strings"][textbox]["input_text"]).strip()
        
        for textbox in loaded_json[imagekey]["textbox_locations"]:
            loaded_json[imagekey]["textbox_locations"][textbox][1] = re.sub(' +', ' ',loaded_json[imagekey]["textbox_locations"][textbox][1]).strip()

    with open(targetfile,"w",encoding="utf-8") as sacfile:
        json.dump(loaded_json,sacfile,indent=4)



def generate_boundingbox_correction_json(data_dir="raw_data_jsons",data_dir_dumpfile="image_dir",OCR_result_jsons = ["ZZZZ_OCR_OUTPUTJSON.json"],dump_target=""):
    workingdir = "working_dir"
    if not os.path.isdir(workingdir):
        os.mkdir(workingdir)
        
    for idx in entity_workdirnames:
        entity_workdirnames[idx] = os.path.join(workingdir,entity_workdirnames[idx])
    
    
    target_candidates = {}
    for i in os.listdir(data_dir):
        if "meme_" in i and "_labelout.json" in i:
            with open(os.path.join(data_dir,i),"r",encoding="utf-8") as opened_pullfile:
                loaded_pulldict = json.load(opened_pullfile)
                for pulled_item in loaded_pulldict:
                    if not "tags" in pulled_item:
                        continue
                    target_candidates[pulled_item["filename"]] = pulled_item["tags"]
            # dump dir for images, label file
        
    # LOAD ALL IMAGES POSSIBLE AND GET TAGS??

        
           

    overall_OCRdict = {}
    for jsontarget in OCR_result_jsons:
        overall_OCRdict.update(OCR_load_results(jsontarget))
    
           
    proposed_json = []
    valid_sample_counter = {}
    seen_images = set()
    iduniquecounter=0

    for item in overall_OCRdict:
        # print(item)
        # print(overall_OCRdict[item])
        sample = overall_OCRdict[item]
        truename = sample["filename"]
        
        tags = target_candidates[truename]
        filepath = os.path.join(data_dir,data_dir_dumpfile,truename)

        
        entity_to_text = {
            0:"Meme",
            1:sample["Atext"],
            2:sample["Btext"],
            3:sample["Ctext"],
            4:sample["Dtext"],
            5:sample["Etext"],
        }
        
        
        
        
        proposedresults = []
                
        entity_reflector = {
        0:"Meme",
        1:"A_entity",
        2:"B_entity",
        3:"C_entity",
        4:"D_entity",
        5:"E_entity"}
        
        
                
        
        if not os.path.exists(filepath): # check for image existence. Removal means invalid from final publish.
            continue 
        with Image.open(filepath) as im:
            pulled_image_width, pulled_image_height = im.size

        # print("Height:",pulled_image_height)
        # print("Width:",pulled_image_width)
        falsepredictionslist = []
        
        
        
        letters = ["A","B","C","D","E"]
        for letter in letters:
            if sample[letter+"boxes"]: # there are boxes selected, it isn't none.
                # print(sample[letter+"text"])
                # print(sample[letter+"boxes"])
                leftmost = 99999
                rightmost = -1
                leastheight = 99999
                highestheight = -1
                erroneousbox = False
                # print(sample)
                for setofboxes in sample[letter+"boxes"]:
                    for single_anchorbox in setofboxes:
                        try:
                            if single_anchorbox[0]<leftmost:
                                leftmost = single_anchorbox[0]
                            if single_anchorbox[0]>rightmost:
                                rightmost = single_anchorbox[0]
                            if single_anchorbox[1]<leastheight:
                                leastheight = single_anchorbox[1]
                            if single_anchorbox[1]>highestheight:
                                highestheight = single_anchorbox[1]
                        except TypeError as e:
                            erroneousbox=True
                            break
                # print(leftmost, rightmost, leastheight, highestheight)
                if erroneousbox:
                    continue
                boxwidth = (rightmost-leftmost)/pulled_image_width*100
                boxheight = (highestheight-leastheight)/pulled_image_height*100

                        
                # pulled_image_width
                # pulled_image_height
                
                proposed_dictionary = {
                    "id": str(datetime.datetime.now().isoformat()).replace("-","").replace(":","").replace(".","")+str(iduniquecounter), # unique id, tagged to sys time.
                    "type": "rectanglelabels",        
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": pulled_image_width, "original_height": pulled_image_height,
                    "image_rotation": 0,
                    "value": {
                      "rotation": 0,          
                      "x": leftmost/pulled_image_width*100,
                      "y": leastheight/pulled_image_height*100,
                      "width": boxwidth,
                      "height": boxheight,
                      "rectanglelabels": [letter]
                    }
                }
                iduniquecounter+=1
                falsepredictionslist.append(proposed_dictionary)
        
        proposed_dictionary = {
            "data":{
                "archetype":sample["archetype"],
                # "image":sample["filename"],
                "image":"/data/local-files/?d="+urllib.parse.quote(os.path.join(os.path.abspath(os.getcwd()),filepath)),
                # Sample: Users%5CUser%5CDesktop%5Cmeow%20meow%5CCompiled_images%5C3t986v.jpg
                # "image":os.path.join(item[0],sample["filename"]),
                "A_entity":sample["Atext"],
                "B_entity":sample["Btext"],
                "C_entity":sample["Ctext"],
                "D_entity":sample["Dtext"],
                "E_entity":sample["Etext"],
                "originalAboxes":sample["Aboxes"],
                "originalBboxes":sample["Bboxes"],
                "originalCboxes":sample["Cboxes"],
                "originalDboxes":sample["Dboxes"],
                "originalEboxes":sample["Eboxes"],
                "plain_cropbox":entity_crop_areas[sample["archetype"]],
                "tags":tags,
                "true_filename":sample["filename"],
                # "all_detected_textboxes":alldetected_textlist
                "original_OCRtext": sample["original_OCRtext"],
            },
            "predictions":[{
                "model_version":"one",
                "score": 0.5,
                "result":falsepredictionslist
                
            }]
        }
        
        
        for targetkey in ["A_entity","B_entity","C_entity","D_entity","E_entity","originalAboxes","originalBboxes","originalCboxes","originalDboxes","originalEboxes"]:
            if not proposed_dictionary["data"][targetkey]:
                proposed_dictionary["data"][targetkey]="None"
        

        # pprint.pprint(proposed_dictionary)
        # input()
        proposed_dictionary.update(overall_OCRdict[sample["filename"]])
        proposed_json.append(proposed_dictionary)
        

    with open(dump_target,"w",encoding="utf-8") as dumpfile:
        json.dump(proposed_json,dumpfile,indent=4)


    # <View>
    # <Image name="image" value="$image"/>
      
      # <RectangleLabels name="label" toName="image">
        # <Label value="A" background="green"/>
        # <Label value="B" background="blue"/>
        # <Label value="C" background="red"/>
        # <Label value="D" background="orange"/>
        # <Label value="E" background="yellow"/>
      # </RectangleLabels>
      # <View style="display: grid;  grid-template-columns: 1fr 1fr; max-height: 500px; width: 80%;">
        # <Header value="A"/>
        # <Header value="$A_entity"/>
        # <Header value="B"/>
        # <Header value="$B_entity"/>
        # <Header value="C"/>
        # <Header value="$C_entity"/>
        # <Header value="D"/>
        # <Header value="$D_entity"/>
        # <Header value="E"/>
        # <Header value="$E_entity"/>
      # </View>
      
      # <Header value="A"/>
          # <Header value="$A_entity"/>

            # <TextArea name="A_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="B"/>
          # <Header value="$B_entity"/>

            # <TextArea name="B_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="C"/>
          # <Header value="$C_entity"/>

            # <TextArea name="C_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="D"/>
          # <Header value="$D_entity"/>

            # <TextArea name="D_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="E"/>
          # <Header value="$E_entity"/>

            # <TextArea name="E_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
      
    # </View>



    ########################################################################################################################################


    # <View>
        # <View style="display: grid;  grid-template-columns: 1fr 1fr;max-height: 800px; width: 100%;">
        # <Image name="image" value="$image"/>
          # <RectangleLabels name="label" toName="image">
          # <View style="display: grid;  grid-template-columns: 1fr 1fr; 1fr; max-height: 500px; width: 100%;">
          # <Label value="A" background="green"/>
          # <Header value="$A_entity"/>
          # <Label value="B" background="blue"/>
          # <Header value="$B_entity"/>
          # <Label value="C" background="red"/>
          # <Header value="$C_entity"/>
          # <Label value="D" background="orange"/>
          # <Header value="$D_entity"/>
          # <Label value="E" background="yellow"/>
          # <Header value="$E_entity"/>
          # </View>
          # </RectangleLabels>
        
      # </View>
      # <Choices name="acceptance" toName="image">
        # <Choice value="REJECT"/>
      # </Choices>
      
        
        
      
      # <Header value="A"/>
          # <Header value="$A_entity"/>

            # <TextArea name="A_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="B"/>
          # <Header value="$B_entity"/>

            # <TextArea name="B_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="C"/>
          # <Header value="$C_entity"/>

            # <TextArea name="C_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="D"/>
          # <Header value="$D_entity"/>

            # <TextArea name="D_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="E"/>
          # <Header value="$E_entity"/>

            # <TextArea name="E_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
      
    # </View>
    pass




def parse_boundingboxes_fixed(targetfile):
    # parse the results of the bounding box corrections annotations. Does not generate the Relations + Entity annotations file yet.
    # Reminder: Step 1 -> Spelling Correction + Grouping of text  
    # Step 2-> Bounding Box Corrections 
    # Step 3-> Relations + Entity Annotations
    with open(targetfile,"r",encoding="utf-8") as targetfile_opened:
        loaded_json = json.load(targetfile_opened)
    
    accepted_instances = []
    
    for labelinstance in loaded_json:
        rejection_indicator = False
        
        for annotation_result in labelinstance["annotations"][0]["result"]:  # check rejection
            if "choices" in annotation_result["value"]:
                if "REJECT" == annotation_result["value"]["choices"][0]:
                    rejection_indicator=True
                    break
                    
        if rejection_indicator:
            # some flagged action here.
            continue
        
        pairwisekeys = [["A","Abox"],["B","Bbox"],["C","Cbox"],["D","Dbox"],["E","Ebox"]]
        dualitydict = {
            "A":False,"Abox":False,
            "B":False,"Bbox":False,
            "C":False,"Cbox":False,
            "D":False,"Dbox":False,
            "E":False,"Ebox":False,
            }
        
        os.path.join( labelinstance["data"]["true_filename"])
        
           
            
        for annotation_result in labelinstance["annotations"][0]["result"]:
            
            if annotation_result["type"]=="rectanglelabels":
                targetletter = annotation_result["value"]["rectanglelabels"][0]
                picheight = annotation_result["original_height"]
                picwidth = annotation_result["original_width"]
                x = annotation_result["value"]["x"]#/100.0*picwidth
                y = annotation_result["value"]["y"]#/100.0*picheight
                boxwidth = annotation_result["value"]["width"]#/100.0*picwidth
                boxheight = annotation_result["value"]["height"]#/100.0*picheight
                dualitydict[targetletter+"box"] = {"height":boxheight,"width":boxwidth,"x":x,"y":y}
                
                
            if annotation_result["type"]=="textarea":
                corrected_text = annotation_result["value"]["text"][0]
                targetletter = annotation_result["from_name"][0]
                dualitydict[targetletter] = corrected_text
        
        for pairwise in pairwisekeys:
            if dualitydict[pairwise[1]]:
                # A bounding box was indicated
                if not dualitydict[pairwise[0]]:
                    # it's not been "corrected". We hence load the old text we corrected after OCR.
                    dualitydict[pairwise[0]] =  labelinstance["data"][pairwise[0]+"_entity"]
        
        textboxlist = []
        
        for pairwise in pairwisekeys:
            if dualitydict[pairwise[1]]:
                textboxlist.append([dualitydict[pairwise[0]],dualitydict[pairwise[1]]])
        
            
        proposed_dictionary = {
            "archetype":labelinstance["data"]["archetype"],
            "tags":labelinstance["data"]["tags"],
            "blind_defaultcrop":labelinstance["data"]["plain_cropbox"],
            "blind_defaultcrop":labelinstance["data"]["plain_cropbox"],
            "true_filename": labelinstance["data"]["true_filename"],
            "textbox_lists":textboxlist,
            "pic_height":picheight,
            "pic_width":picwidth,
        }
        accepted_instances.append(proposed_dictionary)
            
    return accepted_instances





def relation_annotation_json_generator(data_dir = "raw_data_jsons", data_dir_dumpfile = "image_dir", workingdir = "working_dir",fixed_boundingbox_results="ZZZZ_FIXED_BOUNDINGBOXRESULTS.json",dumptarget = "label_studio_reference_input_Relations_Entitiesasdsadasd.json"):
    all_fixed_boundingboxes = parse_boundingboxes_fixed(fixed_boundingbox_results)
    if not os.path.isdir(workingdir):
        os.mkdir(workingdir)
    
    # print(len(all_fixed_boundingboxes))
    
    

           
    proposed_json = []
    valid_sample_counter = {}
    seen_images = set()
    for item in all_fixed_boundingboxes:
        # pprint.pprint(item)
        truename = item["true_filename"]
        
        filepath = os.path.join(data_dir,data_dir_dumpfile,truename)
        
        
        
        
        proposed_dictionary = {
            "data":{
                "memecreator":"MEME_CREATOR",
                "image":"/data/local-files/?d="+urllib.parse.quote(os.path.join(os.path.abspath(os.getcwd()),filepath)),
                "text1":"",
                "text1box":"",
                "text2":"",
                "text2box":"",
                "text3":"",
                "text3box":"",
                "text4":"",
                "text4box":"",
                "text5":"",
                "text5box":"",
            },
        }
        
        
        
        textboxes = item["textbox_lists"]
        
        counter=1
        for textboxpair in textboxes:
            proposed_dictionary["data"]["text"+str(counter)] = textboxpair[0]
            proposed_dictionary["data"]["text"+str(counter)+"box"] = textboxpair[1]
            counter+=1
        
        
        proposed_dictionary["data"].update(item)
        
        

        # pprint.pprint(proposed_dictionary)
        # input()
        proposed_json.append(proposed_dictionary)
        
    with open(dumptarget,"w",encoding="utf-8") as dumpfile:
        json.dump(proposed_json,dumpfile,indent=4)
            

        # <View>
        # <Image name="image" value="$image"/>
        # <Relations>
            # <Relation value="Affirm/Favor"/>
            # <Relation value="Doubt/Disfavor"/>
            # <Relation value="Superior"/>
            # <Relation value="Equal"/>
            # <Relation value="Upgrade"/>
            # <Relation value="Degrade"/>
            # <Relation value="Indifferent"/>
        # </Relations>
          # <Labels name="creatorlabel" toName="creatortext">
              # <Label value="MEME CREATOR"/>
          # </Labels>
          # <Text name="creatortext" value="$memecreator" granularity="word"/>
          
          
          # <Labels name="label1" toName="text1">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text1" value="$text1" granularity="word"/>
          
          
          # <Labels name="label2" toName="text2">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text2" value="$text2" granularity="word"/>
          
          
            # <Labels name="label3" toName="text3">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text3" value="$text3" granularity="word"/>
          
          
            # <Labels name="label4" toName="text4">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text4" value="$text4" granularity="word"/>
          
          
            # <Labels name="label" toName="text5">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text5" value="$text5" granularity="word"/>
            
          # <Choices name="REJECTION" toName="image">
            # <Choice value="REJECT"/>
          # </Choices>
        # </View>

    










        # <View>
        # <Image name="image" value="$image"/>
        # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
        # <Relations>
            # <Relation value="Affirm/Favor"/>
            # <Relation value="Doubt/Disfavor"/>
            # <Relation value="Superior"/>
            # <Relation value="Equal"/>
            # <Relation value="Upgrade"/>
            # <Relation value="Degrade"/>
            # <Relation value="Indifferent"/>
        # </Relations>
          # <Labels name="creatorlabel" toName="creatortext">
              # <Label value="MEME CREATOR"/>
          # </Labels>
          # <Text name="creatortext" value="$memecreator" granularity="word"/>
          # </View>  
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
          # <Labels name="label1" toName="text1">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text1" value="$text1" granularity="word"/>
          # </View>
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
          
          # <Labels name="label2" toName="text2">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text2" value="$text2" granularity="word"/>
          
            # </View>
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
            # <Labels name="label3" toName="text3">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text3" value="$text3" granularity="word"/>
          
            # </View>
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
            # <Labels name="label4" toName="text4">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text4" value="$text4" granularity="word"/>
          
            # </View>
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">
            # <Labels name="label" toName="text5">
            # <Label value="Entity 1"/>
            # <Label value="Entity 2"/>
            # <Label value="Entity 3"/>
            # <Label value="Entity 4"/>
            # <Label value="Entity 5"/>
            # <Label value="Entity 6"/>
          # </Labels>
          # <Text name="text5" value="$text5" granularity="word"/>
          # </View>
          
          # <View style="display: grid;  grid-template-columns: 1fr; max-height: 500px; width: 80%; border-style: solid; align:center;">    
          # <Choices name="REJECTION" toName="image">
            # <Choice value="REJECT"/>
          # </Choices>
          # </View>
        # </View>

    pass
    



def relationshipfinal_output_parser(targetfile,target_savefile = "ZZZZ_completed_processed_fullabels_shaun.json",errorlog="errortargetfile.json"):
    with open(targetfile,"r",encoding="utf-8") as targetfile_opened:
        loaded_json = json.load(targetfile_opened)
    
    errorlist= []
    
    
    
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
    correction_dumper = []
    accepted_counter = 0
    rejected_counter = 0
    spell_counter = 0
    
    if os.path.exists(target_savefile):
        with open(target_savefile,"r",encoding="utf-8") as savepointfile:
            accepted_instances = json.load(savepointfile)
        
    with open("Final_accepted_images_ref.json","r",encoding="utf-8") as reffile:
        ref_imagelist = json.load(reffile)
        
    for labelinstance in loaded_json:
    
        if not labelinstance["data"]["true_filename"] in ref_imagelist:
            continue
    
        if labelinstance["data"]["true_filename"] in accepted_instances:
            continue
        
    
        
        rejection_indicator = False
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
                                        errorlist.append(labelinstance["data"]["true_filename"])
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
                                                errorlist.append(labelinstance["data"]["true_filename"])
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
                                                errorlist.append(labelinstance["data"]["true_filename"])
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
        textbox_locations = {}
        
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

        
            
        
        
        for textbox_accepted in tokenised_strings:
            textbox_locations[textbox_accepted] = [labelinstance["data"][textbox_accepted+"box"],labelinstance["data"][textbox_accepted]]
            


        final_dataset_output = {
            "equivalent_ids":final_idmatchup, # save IDs that have no relation, as they are equivalent.
            "relationmapper":relationmapper,  # save Relations between IDs.
            "entity_spans_labels":entity_matchup_dict, #get the exact entity spans and expected labels for each model output.      
            "tokenised_strings":tokenised_strings, # all tokenised versions of the strings.
            "filename":labelinstance["data"]["true_filename"],
            "archetype": labelinstance["data"]["archetype"],
            "blind defaults": labelinstance["data"]["blind_defaultcrop"],
            "textbox_locations":textbox_locations
        }
        
        accepted_instances[labelinstance["data"]["true_filename"]] = final_dataset_output # something...?
        
        accepted_counter+=1
        if random.randint(0,10)>9: # rng based saving.
            with open(target_savefile,"w",encoding="utf-8") as completefile:
                    json.dump(accepted_instances,completefile,indent=4)
        
    with open(target_savefile,"w",encoding="utf-8") as completefile:
        json.dump(accepted_instances,completefile,indent=4)
    print("spell_counter",spell_counter)
    print("accepted_counter", accepted_counter)
    print("rejected_counter",rejected_counter)

        # we can dump the final file.


    with open(errorlog,"w",encoding="utf-8") as errorfile:
        json.dump(errorlist,errorfile,indent=4)
    
    for i in ref_imagelist:
        if not i in accepted_instances:
            print("Failed to accept:",i)
    
    return accepted_instances


    

class dataset_information_generation_class():
    # Class that generates final dataset json. (After all annotations).
    # The json is then loaded by the dataset class.

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

    map_templates_names = {
        'Anime-Girl-Hiding-from-Terminator' : 'Anime-Girl-Hiding-from-Terminator.jpg',
        'Arthur-Fist' : 'Arthur-Fist.jpg',
        'Blank-Nut-Button' : 'Blank-Nut-Button.jpg',
        'Both-Buttons-Pressed' : 'Both-Buttons-Pressed.jpg',
        'Buff-Doge-vs-Cheems' : 'Buff-Doge-vs-Cheems.png',
        'Clown-Applying-Makeup' : 'Clown-Applying-Makeup.jpg',
        'Cuphead-Flower' : 'Cuphead-Flower.png',
        'Disappointed-Black-Guy' : 'Disappointed-Black-Guy.jpg',
        'Distracted-Boyfriend' : 'Distracted-Boyfriend.jpg',
        'Drake-Hotline-Bling' : 'Drake-Hotline-Bling.jpg',
        'Epic-Handshake' : 'Epic-Handshake.jpg',
        'Ew-i-stepped-in-shit' : 'Ew-i-stepped-in-shit.jpg',
        'Fancy-pooh' : 'Fancy-pooh.png',
        'Feels-Good-Man' : 'Feels-Good-Man.jpg',
        'Hide-the-Pain-Harold' : 'Hide-the-Pain-Harold.jpg',
        'If-those-kids-could-read-theyd-be-very-upset' : 'If-those-kids-could-read-theyd-be-very-upset.png',
        'Is-This-A-Pigeon' : 'Is-This-A-Pigeon.jpg',
        'kermit-window' : 'kermit-window.jpg',
        'Left-Exit-12-Off-Ramp' : 'Left-Exit-12-Off-Ramp.jpg',
        'Moe-throws-Barney' : 'Moe-throws-Barney.jpg',
        'Mother-Ignoring-Kid-Drowning-In-A-Pool' : 'Mother-Ignoring-Kid-Drowning-In-A-Pool.jpg',
        'Mr-incredible-mad' : 'Mr-incredible-mad.png',
        'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask' : 'Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask.jpg',
        'Running-Away-Balloon' : 'Running-Away-Balloon.jpg',
        'Skinner-Out-Of-Touch' : 'Skinner-Out-Of-Touch.jpg',
        'Soyboy-Vs-Yes-Chad' : 'Soyboy-Vs-Yes-Chad.jpg',
        'Spider-Man-Double' : 'Spider-Man-Double.jpg',
        'Spongebob-Burning-Paper' : 'Spongebob-Burning-Paper.jpg',
        'Squidward' : 'Squidward.jpg',
        'Teachers-Copy' : 'Teachers-Copy.png',
        'The-Scroll-Of-Truth' : 'The-Scroll-Of-Truth.jpg',
        'They-are-the-same-picture' : 'They-are-the-same-picture.jpg',
        'This-Is-Brilliant-But-I-Like-This' : 'This-Is-Brilliant-But-I-Like-This.jpg',
        'This-is-Worthless' : 'This-is-Worthless.jpg',
        'Tuxedo-Winnie-the-Pooh-grossed-reverse' : 'Tuxedo-Winnie-the-Pooh-grossed-reverse.jpg',
        'Tuxedo-Winnie-The-Pooh' : 'Tuxedo-Winnie-The-Pooh.png',
        'Two-Paths' : 'Two-Paths.png',
        'Types-of-Headaches-meme' : 'Types-of-Headaches-meme.jpg',
        'Weak-vs-Strong-Spongebob' : 'Weak-vs-Strong-Spongebob.png'
        }
    
    def __init__(self, importfile=None, dumpdir = None,target_tokeniser=False,approved_images=[],verbose=False,use_templates=False,template_dir="templates"):
        if dumpdir==None:
            dumpdir = os.path.join("parse","raw_data_jsons","image_dir")
        
        with open(importfile,"r",encoding="utf-8") as opened_file:
            all_items = json.load(opened_file)

        relationships_counter = {}
        textcounter = {}
        entitycounter = {}
        archcounter = {}
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
                
            if not all_items[imagekey]["archetype"] in archcounter:
                archcounter[all_items[imagekey]["archetype"]] = 0
            archcounter[all_items[imagekey]["archetype"]] +=1
            
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
                    "archetype": all_items[imagekey]["archetype"]
                    }
            
            if use_templates:
                singleinstance["source_image"] = os.path.join(template_dir, self.map_templates_names[all_items[imagekey]["archetype"]])
                    
            # pprint.pprint(singleinstance)
            # print(numerical_relationships)
            # print(inverted_numerical_relationships)
            # input()
            self.savelist.append(singleinstance)
    
        print("Archetype counter:",archcounter)
        
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
    STRAIGHTEN = False
    if STRAIGHTEN:
        straightener() # un minimises a json, making its structure readable to humans. 

    # parse the results of the bounding box corrections annotations. 
    # Reminder: Step 1 -> Spelling Correction + Grouping of text  
    # Step 2-> Bounding Box Corrections 
    # Step 3-> Relations + Entity Annotations
    # Step 4 -> Compose into dataset jsons.


    #Step 1
    annotator_reference = {}
    initial_OCR_labelstudio_generator(data_dir = "raw_data_jsons", data_dir_dumpfile = "image_dir",limiter=150,workingdir = "working_dir",dump_target="label_studio_reference_input_OCR_INITIAL.json")
    # produces "label_studio_reference_input_OCR_INITIAL.json", the json to load to perform bounding box corrections with.
    # Annotate, and then export from label studio. Place the output file as the argument below.
    
    
    
    #Step 2
    OCR_load_results("Annotated_OCR_OUTPUT.json") # Loads the ocr annotation output. Utility function that can be used to test if the export has no issue.
    generate_boundingbox_correction_json(OCR_result_jsons=["Annotated_OCR_OUTPUT.json"], dump_target ="label_studio_reference_input_BoundingBox_Corrections.json")
    # generates the file label_studio_reference_input_BoundingBox_Corrections.json, which is imported into label studio.
    # Annotate the bounding boxes and tehn export from label studio. Place the output file as the argument below.
    
    
    #Step 3
    parse_boundingboxes_fixed("Annotated_BoundingBox_Corrections.json") # utility function used within the relation annotation json generator. tests for export issues.
    relation_annotation_json_generator(fixed_boundingbox_results="Annotated_BoundingBox_Corrections.json",dumptarget = "label_studio_reference_input_Relations_Entities.json")
    
    
    
    #Step 4
    relationshipfinal_output_parser("Annotated_Final.json",target_savefile="Processed_annotations.json")
    final_dataset_generator = dataset_information_generation_class(importfile="Processed_annotations.json", dumpdir = os.path.join("raw_data_jsons","image_dir"), target_tokeniser=False,approved_images=[],verbose=False,use_templates=False,template_dir="templates")
    final_dataset_generator.dump_dataset(targetfile = "final_dataset_cleared_NON_CLEARED.json")
    quit()
        



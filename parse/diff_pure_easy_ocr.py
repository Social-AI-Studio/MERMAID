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
import easyocr
import numpy as np
from PIL import Image, ImageChops
from transformers import AutoTokenizer, ViltProcessor, AutoProcessor, Blip2Processor




    
# Provide all OCR items for a targetted bunch of pictures.
def create_all_ocrs_differential_type(data_dir = "raw_data_jsons", data_dir_dumpfile = "image_dir",limiter=150,workingdir = "working_dir",combined_json_existence=False,combined_json_name="all_memes_combined_json.json",template_dir="templates"):
    # generates a label studio file for OCR, along with bounding boxes.
    # note that the label studio file given does not allow for bounding box correction. OCR corrections and bounding box corrections  are separated.
    # Limiter designates how many to "take" from each file, since we technically pull way more than required.
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
    
    
    
    do_boundingboxsaver = True
    
    full_template_dir = os.path.join(data_dir,template_dir)

    if not os.path.isdir(workingdir):
        os.mkdir(workingdir)

    target_candidates = []
    for i in os.listdir(data_dir):
        if "_labelout.json" in i:
            target_candidates.append([os.path.join(data_dir,data_dir_dumpfile), os.path.join(data_dir,i), i.replace("_labelout.json","").replace("meme_","")])
                        # dump dir for images, label file
                        
    
    OCRreader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    easyocr_minimum_score = 0.1
    
    valid_sample_counter = {}
    
    if os.path.exists("diff_ocr_savepoint"):
        with open("diff_ocr_savepoint","r",encoding="utf-8") as difffile:
            seen_images = set(json.load(difffile))
    else:
        seen_images = set()    

    
    if os.path.exists("failed_template_search.json"):
        with open("failed_template_search.json","r",encoding="utf-8") as difffile:
            failed_templates = set(json.load(difffile))
    else:
        failed_templates = set()       
            
    proposed_json = []
    if os.path.exists("OCR_bitwised"+"_label_studio_reference_output.json"):
        with open("OCR_bitwised"+"_label_studio_reference_output.json","r",encoding="utf-8") as dumpfile:
            proposed_json = json.load(dumpfile)
        
        
    for item in target_candidates:
        
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

            
            
                
            if not "tags" in sample: # continue. Missing tags..
                print("no tags...")
                continue
            if not os.path.exists(filepath):
                # print(filepath)
                # input()
                print("filepath didn't exist...")
                continue # SKIP ENTRY. REMOVED FOR VARIOUS REASONS
            
            if filepath in failed_templates:
                print("failed search last time.")
                continue
                
            if filepath in seen_images: # duped label.
                print("duped entry.")
                continue
            else:
                seen_images.add(filepath)
                
            
            truename = sample["filename"]
            
            tags = sample["tags"]
            

            # print(item[2])
            with Image.open(filepath) as im:
                pulled_image_width, pulled_image_height = im.size
                haspassed=False
                for sizechecker in standard_imagesize_dict[item[2]]:
                    if pulled_image_width==sizechecker[0] and pulled_image_height==sizechecker[1]:
                        haspassed=True
                if not haspassed:
                    continue # don't process this image. it does not conform to normal unedited template dimensions for any within the "approved list.
                
                
                with Image.open(os.path.join(full_template_dir,item[2]+".png")) as im2:
                    template_image_width,template_image_height = im2.size
                    # im2= im2.convert("RGB") # conversion like this causes a lot of noise. the formula doesn't really seem to map to pixel values directly somehow. Possible conversion issue.
                    im2 = Image.fromarray(np.array(im2)[:,:,:3],mode="RGB")
                    if template_image_width!=pulled_image_width or template_image_height!=pulled_image_height:
                        print("Image:",pulled_image_width,pulled_image_height)
                        print(item[2])
                        print("Template:",template_image_width,template_image_height)
                        failed_templates.add(filepath)
                        # raise ValueError()
                    
                    diff = ImageChops.difference(im2,im)
                    data = np.array(diff)
                    data[(data > (20,20,20)).all(axis = -1)] = (255,255,255)
                    data[(data <= (20,20,20)).all(axis = -1)] = (0,0,0)
                    diff2 = Image.fromarray(data, mode='RGB')

                    
                    imarray = np.array(im)
                    # im2array = np.array(im2.convert("RGB"))
                    im2array = np.array(im2)
                    for outerdim in range(len(imarray)):
                        for innerdim in range(len(imarray[outerdim])):
                            if im2array[outerdim][innerdim][0] == imarray[outerdim][innerdim][0] and im2array[outerdim][innerdim][1] == imarray[outerdim][innerdim][1] and im2array[outerdim][innerdim][2] == imarray[outerdim][innerdim][2]:
                                imarray[outerdim][innerdim] = (0,0,0)
                            else:
                                difference_val1 = abs(int(im2array[outerdim][innerdim][0])-int(imarray[outerdim][innerdim][0]))
                                difference_val2 = abs(int(im2array[outerdim][innerdim][1])-int(imarray[outerdim][innerdim][1]))
                                difference_val3 = abs(int(im2array[outerdim][innerdim][2])-int(imarray[outerdim][innerdim][2]))
                                if sum([difference_val1,difference_val2,difference_val3])<=45:
                                    imarray[outerdim][innerdim] = (0,0,0)
                                # else:
                                    # print(sum([difference_val1,difference_val2,difference_val3]))
                    directcomparator = Image.fromarray(imarray, mode='RGB')
                    
                    
                    
                diff2.save(os.path.join(workingdir,"differential.jpg"))
                diff.save(os.path.join(workingdir,"alt_differential.jpg"))
                directcomparator.save(os.path.join(workingdir,"rep_differential.jpg"))

            complete_image_text_items = []
            all_image_text_proposals = OCRreader.readtext(os.path.join(workingdir,"alt_differential.jpg")) ########## WHICH DO WE USE????
            # all_image_text_proposals = OCRreader.readtext(filepath) # original.
            
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
            

            proposed_dictionary = {
                "archetype":item[2],
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
            }
            print(proposed_dictionary)
            proposed_json.append(proposed_dictionary)
            if random.randint(0,100)>90:
                with open("failed_template_search.json","w",encoding="utf-8") as difffile:
                    json.dump(list(failed_templates),difffile,indent=4)
                with open("diff_ocr_savepoint","w",encoding="utf-8") as difffile:
                    json.dump(list(seen_images),difffile,indent=4)
                with open("OCR_bitwised"+"_label_studio_reference_output.json","w",encoding="utf-8") as dumpfile:
                    json.dump(proposed_json,dumpfile,indent=4)
        # with open(item[2]+"_label_studio_reference_output.json","w",encoding="utf-8") as dumpfile:
            # json.dump(proposed_json,dumpfile,indent=4)
    print("items:",len(proposed_json))
    with open("OCR_bitwised"+"_label_studio_reference_output.json","w",encoding="utf-8") as dumpfile:
        json.dump(proposed_json,dumpfile,indent=4)
    os.remove("diff_ocr_savepoint")

    # <View>
      # <Image name="image" value="$image"/>
      # <Header value="Select if the meme usage is literal/no actual entities/events are mentioned. Else select NIL. (Leave all other fields blank if meme usage was literal.)"/>
      # <Choices name="Accept" toName="image" choice="single" required="true">
            # <Choice value="Accept"/>
            # <Choice value="Reject"/>
      # </Choices>
        # <Header value="OCR/Statement Corrections"/>
        # <Header value="A"/>
            # <Choices name="A" toName="image" choice="multiple" value="$alloptions"/>
            # <TextArea name="A_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="B"/>
            # <Choices name="B" toName="image" choice="multiple" value="$alloptions"/>
            # <TextArea name="B_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="C"/>
            # <Choices name="C" toName="image" choice="multiple" value="$alloptions"/>
            # <TextArea name="C_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="D"/>
            # <Choices name="D" toName="image" choice="multiple" value="$alloptions"/>
            # <TextArea name="D_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <Header value="E"/>
            # <Choices name="E" toName="image" choice="multiple" value="$alloptions"/>
            # <TextArea name="E_true" toName="image" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
    # </View>

if __name__=="__main__":
    create_all_ocrs_differential_type()
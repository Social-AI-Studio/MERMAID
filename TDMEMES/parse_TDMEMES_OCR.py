import json
import os
import pprint
import urllib.parse
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true


def straighten(targetfile):
    with open(targetfile,"r",encoding="utf-8") as ocrfile:
        loaded_out = json.load(ocrfile)

    with open(targetfile,"w",encoding="utf-8") as ocrfile:
        json.dump(loaded_out,ocrfile,indent=4)





if __name__=="__main__":
    targetfile = "SGMEMES_Corrected_OCR_labelout.json"
    straighten(targetfile)
    duplicant_collection = []
    full_collection = []
    
    with open(targetfile,"r",encoding="utf-8") as ocrfile:
        loaded_out = json.load(ocrfile)
    data_dir = "TDMEMES"
    data_dir_dumpfile = "TD_Memes"
    
    
    total_analyzed_count = 0
    unsuitable_nondupe_count = 0
    for sample in loaded_out:
        if sample["annotations"][0]["was_cancelled"]:
            unsuitable_nondupe_count+=1
            continue
        total_analyzed_count+=1
        sample_dict = {"annotations":{}}
        sample_dict["source_image"] = sample["data"]["source_image"]
        sample_dict["original_textboxes"] = sample["data"]["all_text_in_image"].split("\n")
        
        skipsample_flag = False
        
        
        
        for annotation_item in sample["annotations"][0]["result"]:
            
            if annotation_item["type"]=="textarea":
                corrected_textbox = annotation_item["value"]["text"][0]
                entity_num = int(annotation_item["from_name"].split("_")[0][-1])
                if not entity_num in sample_dict["annotations"]:
                    sample_dict["annotations"][entity_num] = {}
                    
                if "text" in sample_dict["annotations"][entity_num]: # already been filled in 
                    if "text" in annotation_item["from_name"]:
                        sample_dict["annotations"][entity_num]["text"] = corrected_textbox # replace with corrected text
                    else:
                        continue # it's the original text, uncorrected
                else:
                    sample_dict["annotations"][entity_num]["text"] = corrected_textbox
                
            elif annotation_item["type"]=="rectanglelabels":
                entity_num = int(annotation_item["value"]["rectanglelabels"][0].split("_")[0][-1])
                if not entity_num in sample_dict["annotations"]:
                    sample_dict["annotations"][entity_num] = {}
                sample_dict["annotations"][entity_num]["rect"] = annotation_item["value"]
                del sample_dict["annotations"][entity_num]["rect"]["rotation"]
                del sample_dict["annotations"][entity_num]["rect"]["rectanglelabels"]
            
                sample_dict["original_height"] = annotation_item["original_height"]
                sample_dict["original_width"] = annotation_item["original_width"]
                    
            elif annotation_item["type"]=="choices":
                if annotation_item["value"]["choices"][0]=="REJECT": 
                    skipsample_flag=True
                    
        
        
        if skipsample_flag and not sample_dict["annotations"]:
            duplicant_collection.append(sample_dict["source_image"])
            continue
            
        full_collection.append(sample_dict)
    
    print("Total Number of Memes looked through:",unsuitable_nondupe_count+total_analyzed_count)
    print("Total unsuitable:",unsuitable_nondupe_count)
    print("Total Suitable and Analyzed:",total_analyzed_count)
    print("Number of duplicate participants:",len(duplicant_collection))
    print("Total Accepted Memes:",len(full_collection))
    
    
           
    proposed_json = []
    valid_sample_counter = {}
    seen_images = set()
    for item in full_collection:
        # pprint.pprint(item)
        
        
        
        
        filepath = os.path.join(data_dir,data_dir_dumpfile,item["source_image"])

        proposed_dictionary = {
            "data":{
                "memecreator":"MEME_CREATOR",
                "image":"/data/local-files/?d="+urllib.parse.quote(os.path.join(os.path.abspath(os.getcwd()),filepath)),
                "text1":"",
                "text1box":"",
                "rect1height":"",
                "rect1width":"",
                "rect1x":"",
                "rect1y":"",
                "text2":"",
                "text2box":"",
                "rect2height":"",
                "rect2width":"",
                "rect2x":"",
                "rect2y":"",
                "text3":"",
                "text3box":"",
                "rect3height":"",
                "rect3width":"",
                "rect3x":"",
                "rect3y":"",
                "text4":"",
                "text4box":"",
                "rect4height":"",
                "rect4width":"",
                "rect4x":"",
                "rect4y":"",
                "text5":"",
                "text5box":"",
                "rect5height":"",
                "rect5width":"",
                "rect5x":"",
                "rect5y":"",
            },
        }
        
        
        
        # pprint.pprint(item)
        for entity in item["annotations"]:
        
            proposed_dictionary["data"]["rect"+str(entity)+"height"] = item["annotations"][entity]["rect"]["height"]
            proposed_dictionary["data"]["rect"+str(entity)+"width"] = item["annotations"][entity]["rect"]["width"]
            proposed_dictionary["data"]["rect"+str(entity)+"y"] = item["annotations"][entity]["rect"]["y"]
            proposed_dictionary["data"]["rect"+str(entity)+"x"] = item["annotations"][entity]["rect"]["x"]
            proposed_dictionary["data"]["text"+str(entity)] = item["annotations"][entity]["text"]
            proposed_dictionary["data"]["text"+str(entity)+"box"] = []
        proposed_dictionary["data"]["fullheight"] = item["original_height"]
        proposed_dictionary["data"]["fullwidth"] = item["original_width"]
            
        
        proposed_dictionary["data"]["backup"] = (item)
        

        # pprint.pprint(proposed_dictionary)
        # input()
        proposed_json.append(proposed_dictionary)
        
    with open("SGMEMES_RelationAnnotationImport_dict"+".json","w",encoding="utf-8") as dumpfile:
        json.dump(proposed_json,dumpfile,indent=4)
            

     


    pass
    
    
import os
import json

if __name__=="__main__":

    # This file is designed to count the number of invalid entities if you use RAW OCR. i.e do not do spelling corrections and grouping of text.
    # Note that EasyOCR is somewhat pseudorandom and results will differ very slightly between runs.

    ocroutputfile = os.path.join("parse","Annotated_OCR_OUTPUT.json")
    main_dataset_file = "final_dataset_cleared.json"
    with open(ocroutputfile,"r",encoding="utf-8") as ocrfile:
        targetjson = json.load(ocrfile)
    
    
    with open(main_dataset_file,"r",encoding="utf-8") as full_dataset_file:
        full_dataset = json.load(full_dataset_file)
    
    full_dataset_dict = {}
    for item in full_dataset:
        full_dataset_dict[item["source_image"]] = len(list(item["text_locs"].keys()))
    
    true_mapperdict = {
        "A_true":"A",
        "B_true":"B",
        "C_true":"C",
        "D_true":"D",
        "E_true":"E",
    }
    
    letter_mapperdict = {v:k for k,v in true_mapperdict.items()}
    total_combines = 0
    total_corrections = 0
    combinationdict = {"NOTE":"count, corrections"}
    detected_textboxes = {}
    number_memes = 0
    total_textboxes = 0
    recorrected_images = 0
    combined_textbox_dict = {}
    for sample in targetjson:
        # for option in sample["data"]["alloptions"]:
            # all_options_mapped.append(option["value"][3:])
        if not sample["data"]["true_filename"] in full_dataset_dict:
            continue
        
        count_combines = 0
        corrections = 0
        ignore_sample = False
        data_saver = []
        corrected_textbox_count = 0
        
        
        # check for rejection of the meme sample in OCR stage. And then count the number of final "proposed textboxes" and respectively the corrections.
        for annotation_result in sample["annotations"][0]["result"]:
            if annotation_result["from_name"]=="Accept":
                if annotation_result["value"]["choices"][0]=="Reject":
                    ignore_sample = True
                    break
                else:
                    continue
            elif annotation_result["from_name"] in letter_mapperdict:
                corrected_textbox_count+=1
                if len(annotation_result["value"]["choices"])>1:
                    if not len(annotation_result["value"]["choices"]) in combined_textbox_dict:
                        combined_textbox_dict[len(annotation_result["value"]["choices"])] = 0
                    combined_textbox_dict[len(annotation_result["value"]["choices"])] +=1
                    count_combines+=1
                # is the selection of several choices.
            elif annotation_result["from_name"] in true_mapperdict:
                corrections+=1
                
        if ignore_sample:
            continue
        
        number_memes+=1
        
        
        # register the number of detected textboxes.
        if not len(sample["data"]["alloptions"]) in detected_textboxes: 
            detected_textboxes[len(sample["data"]["alloptions"])] = 0
        detected_textboxes[len(sample["data"]["alloptions"])] +=1        
        num_textboxes = len(sample["data"]["alloptions"])
        
        
        # check that the numbers tally up. for final corrected count vs the number we have in the dataset.
        if corrected_textbox_count != full_dataset_dict[sample["data"]["true_filename"]]:
            # print(sample["data"]["true_filename"])
            recorrected_images+=1
        
        
        total_textboxes+= num_textboxes
        if not (count_combines,corrections) in combinationdict:
            combinationdict[(count_combines,corrections)] = 0
        combinationdict[(count_combines,corrections)] +=1
        
        # if not (corrected_textbox_count,count_combines,corrections) in combinationdict:
            # combinationdict[(corrected_textbox_count,count_combines,corrections)] = 0
        # combinationdict[(corrected_textbox_count,count_combines,corrections)] +=1
        
        
        total_combines+= count_combines
        total_corrections+= corrections
    print("Combinations, Corrections, Combinations viewed")
    # sorted_combinationdict = dict(sorted(combinationdict.items()))
    # print(sorted_combinationdict)
    print(combinationdict)
    print("Text box spread")
    sorted_detected_textboxes = dict(sorted(detected_textboxes.items()))
    print(sorted_detected_textboxes)
    print("Totan Number of textboxes:",total_textboxes)
    print("Total Combined stuff:",total_combines)
    print("Total number of textboxes that still need spelling corrections AFTER merge:",total_corrections)
    print("Total Number of Memes:",number_memes)
    print("Total number of Recorrected (After this OCR) Images:",recorrected_images)
    print("Number of OCR Textboxes combined into one corrected textbox frequencies",combined_textbox_dict)
    totalcombinations = 0
    totaltextbox = 0
    for k in combined_textbox_dict:
        totalcombinations+= combined_textbox_dict[k]
        totaltextbox+= combined_textbox_dict[k]*k
    print("Average number of textboxes we merge per correction:",totaltextbox/totalcombinations)
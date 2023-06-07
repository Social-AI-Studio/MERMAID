import os
import json
import easyocr
import zipfile
import random
import urllib.parse
from PIL import Image
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

if __name__=="__main__":
    "all_text_in_image"
    targetdir = os.path.join("TDMEMES","TD_Memes")
    OCRreader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
    easyocr_minimum_score = 0.1
    with open(os.path.join("TDMEMES","annotation.json"),"r",encoding="utf-8") as infofile:
        infodict = json.load(infofile)
    approved_images = infodict["SG_Memes"]
    # approved_images = infodict["Non_SG_Memes"]
    
    # print(len(approved_images))
    # print(approved_images)
    dumptarget = "TD_savepoint.json"
    fulldumptarget = "TD_labelstudio_import_SGONLY.json"
    
    
    if os.path.exists(dumptarget):
        with open(dumptarget,"r",encoding="utf-8") as dumpfile:
            outputdict = json.load(dumpfile)
        outputlist = []
        for i in outputdict:
            outputlist.append(outputdict[i])
    else:
        outputdict = {}
        outputlist = []
        
        
    for item in os.listdir(targetdir):
        if not item  in approved_images:
            continue
        filepath = os.path.join(targetdir,item)
        if item in outputdict:
            continue
        
        with Image.open(filepath) as im:
            pulled_image_width, pulled_image_height = im.size
            complete_image_text_items = []
            all_image_text_proposals = OCRreader.readtext(filepath)
            
            for detection in all_image_text_proposals:
                textdetected = detection[1]
                confidence = detection[2]
                # print(textdetected,confidence)
                if confidence> easyocr_minimum_score:
                    complete_image_text_items.append(textdetected)
        
    
        proposed_dictionary = {"data":{
            "image":"/data/local-files/?d="+urllib.parse.quote(os.path.join(os.path.abspath(os.getcwd()),filepath)),
            "source_image":item,
            "all_text_in_image":"\n".join(complete_image_text_items),
            }
        }
        print(proposed_dictionary)
        outputdict[item] = proposed_dictionary
        outputlist.append(proposed_dictionary)
        if random.randint(0,100)>95:
            with open(dumptarget,"w",encoding="utf-8") as targetdumpfile:
                json.dump(outputdict, targetdumpfile,indent=4)
                
    with open(fulldumptarget,"w",encoding="utf-8") as targetdumpfile:
        json.dump(outputlist,targetdumpfile,indent=4)
            
            
# <View>
  # <Image name="image" value="$image"/>
  # <RectangleLabels name="label1" toName="image">
          # <Label value="Entity1_Box" background="green"/>
              # <Label value="Entity2_Box" background="blue"/>
          # <Label value="Entity3_Box" background="red"/>
          # <Label value="Entity4_Box" background="purple"/>
          # <Label value="Entity5_Box" background="orange"/>
  # </RectangleLabels>
  # <Header value="All Detected Text (Correct if required):"/>
  # <Text name="text" value="$all_text_in_image"/>
  # <Header value="OCR/Statement Corrections  (If correction is needed)"/>
  	# <Header value="A"/>
  		# <TextArea name="Entity1_original" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <TextArea name="Entity1_text" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
  	# <Header value="B"/>

 	 	# <TextArea name="Entity2_original" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <TextArea name="Entity2_text" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
  	# <Header value="C"/>
  		# <TextArea name="Entity3_original" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <TextArea name="Entity3_text" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
  	# <Header value="D"/>
  		# <TextArea name="Entity4_original" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <TextArea name="Entity4_text" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
  	# <Header value="E"/>
  		# <TextArea name="Entity5_original" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
        # <TextArea name="Entity5_text" toName="text" placeholder="Correction Here" editable="true" maxSubmissions="1"/>
  	# <Choices name="REJECT" toName="image">
      # <Choice value="REJECT"/>
      # </Choices>
# </View>
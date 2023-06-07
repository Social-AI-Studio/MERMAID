import os
import json

with open("final_dataset_cleared.json","r",encoding="utf-8") as finalfile:
    a = json.load(finalfile)
    
    
totaledict = {}
for i in a:
    for k in i["relationships_read"]:
        if not k in totaledict:
            totaledict[k]=0
        totaledict[k] = totaledict[k]+len(i["relationships_read"][k])
        
        
    if "Superior" in i["relationships_read"]:
        if "Inferior" in i["relationships_read"]:
            if len(i["relationships_read"]["Superior"])==len(i["relationships_read"]["Inferior"]):
                continue
            else:
                print(i["source_image"])
        else:
            print(i["source_image"])
print(totaledict)
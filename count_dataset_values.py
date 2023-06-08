import os
import json


targetfile = "final_dataset_cleared.json"

# targetfile = "final_dataset.json"

with open(targetfile,"r",encoding="utf-8") as finalfile:
    a = json.load(finalfile)


print(targetfile)
entitycount = 0
totaledict = {}
for i in a:
    for k in i["relationships_read"]:
        if not k in totaledict:
            totaledict[k]=0
        totaledict[k] = totaledict[k]+len(i["relationships_read"][k])
        
    entitycount+=len(i["actual_entities"])
    if "Superior" in i["relationships_read"]:
        if "Inferior" in i["relationships_read"]:
            if len(i["relationships_read"]["Superior"])==len(i["relationships_read"]["Inferior"]):
                continue
            else:
                print(i["source_image"])
        else:
            print(i["source_image"])
print(totaledict)
print("Total entities:",entitycount)
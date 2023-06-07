import json
import os

# shrink or unshrink all jsons in the directory by changing the dump indentation.
shrink = True
if shrink:
    for i in os.listdir():
        if ".json" in i:
            with open(i,"r",encoding="utf-8") as openedfile:
                a = json.load(openedfile)
            with open(i,"w",encoding="utf-8") as openedfile:
                json.dump(a, openedfile)
else:
    for i in os.listdir():
        if ".json" in i:
            with open(i,"r",encoding="utf-8") as openedfile:
                a = json.load(openedfile)
            with open(i,"w",encoding="utf-8") as openedfile:
                json.dump(a, openedfile,indent=4)
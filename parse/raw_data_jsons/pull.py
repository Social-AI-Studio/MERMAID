from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from PIL import Image
from io import BytesIO
import selenium
import time
import urllib.request
import os
import json
# set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true

all_seen_tags = {}
seen_links = set()

def download_image(entitylist,hardcode_dumpdir="dumpdir"):
    for item in entitylist:
        if not os.path.isdir(hardcode_dumpdir):
            os.mkdir(hardcode_dumpdir)
        
        
        src = item["imgsrc"]    
        req = urllib.request.Request(src, headers={'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"}) 
        newdict = {"filename":src.split("/")[-1]}
        try:
            with urllib.request.urlopen(req) as response:
                image_page = response.read()
                Image.open(BytesIO(image_page)).save(os.path.join(hardcode_dumpdir,item["imgsrc"].split("/")[-1]))
        except urllib.error.HTTPError as e:
                print("----------------------------")
                print(e)
                print(type(e))
                print("----------------------------")
                continue # skip error image. won't be added to the dict as a result.
        item.update(newdict)
    return entitylist


def page_extraction(driver,original_window_id,all_seen_tags,seen_links,filters=[],isclean=False):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    element_linklist = driver.find_elements(By.CSS_SELECTOR,"a.base-img-link")
    giflist = []
    outlist = []
    for linkidx in range(len(element_linklist)):
        driver.execute_script("arguments[0].scrollIntoView();",element_linklist[linkidx])
        
        # print("*"*50)
        # print(element_linklist[linkidx].get_attribute("innerHTML"))
        
        try:
            outlist.append({"href":element_linklist[linkidx].get_attribute("href"),"imgsrc":element_linklist[linkidx].find_element(By.CSS_SELECTOR,"img.base-img").get_attribute("src")})
            # print(element_linklist[linkidx].find_element(By.CSS_SELECTOR,"img.base-img"))
        except selenium.common.exceptions.NoSuchElementException as e:
            # print("-"*50)
            # print(e)
            # print("-"*50)
            # non standard memes appear to be wrapped in noscript, and then a div. 
            try:
                outlist.append({"href":element_linklist[linkidx].get_attribute("href"),"imgsrc":element_linklist[linkidx].find_element(By.CSS_SELECTOR,"div.base-img").get_attribute("data-src")})
            
            except selenium.common.exceptions.NoSuchElementException as e:
                giflist.append({"href":element_linklist[linkidx].get_attribute("href"),"imgsrc":element_linklist[linkidx].find_element(By.CSS_SELECTOR,"source").get_attribute("src"),"vidtype":element_linklist[linkidx].find_element(By.CSS_SELECTOR,"source").get_attribute("type")})
                # continue, to not perform a check on the latest added item. since nothing was added to the outlist.
                continue 
                
            except selenium.common.exceptions.StaleElementReferenceException as e: 
                print("-Redoing page.-") # the javascript did not load in time and a DIV was replaced as an image.
                return False,False,False 
                
        outlist[-1]["noisy"] = isclear # imprint if the item is noisy.
        print(outlist)
        
        if outlist[-1]["imgsrc"][0]=="/":
                outlist[-1]["imgsrc"] = "https:"+ outlist[-1]["imgsrc"]
        if outlist[-1]["imgsrc"] in seen_links:
            outlist.pop(-1) # already been added.
    time.sleep(1)
    filtered_outlist = []
    for idx in range(len(outlist)):
        if outlist[idx]["href"] in seen_links:
            continue
        # print(outlist[idx])
        driver.switch_to.new_window('tab')
        driver.get(outlist[idx]["href"])
        seen_links.add(outlist[idx]["href"])
        taglist = driver.find_elements(By.CLASS_NAME,"img-tag")
        outlist[idx]["tags"] = []
        
        try:
            title = driver.find_element(By.ID,"img-title").text
        except selenium.common.exceptions.NoSuchElementException as e:
            # the meme is untitled.
            title = ""
            
        outlist[idx]["Meme Title"] = title
        outlist[idx]["filename"] = outlist[idx]["imgsrc"].split("/")[-1]
        for tag in taglist:
            outlist[idx]["tags"].append(tag.text)
            # do you want to save all possible tags?
            if not tag.text in all_seen_tags:
                all_seen_tags[tag.text] = 0
            all_seen_tags[tag.text] = all_seen_tags[tag.text] + 1
            
        time.sleep(0.5)
        driver.close()
        driver.switch_to.window(original_window_id)
        if filters:
            for item in filters:
                if item[0] in outlist[idx]["tags"] and item[1] in outlist[idx]["tags"]:
                    filtered_outlist.append(outlist[idx])
                    break
        else:
            filtered_outlist = outlist
    filtered_outlist = download_image(filtered_outlist)
    return filtered_outlist+giflist,seen_links,all_seen_tags
    
    


if __name__=="__main__":
    list_of_targets = [# ("https://imgflip.com/meme/","Buff-Doge-vs-Cheems"),
                       # ("https://imgflip.com/meme/","Drake-Hotline-Bling"),
                       # ("https://imgflip.com/meme/","Tuxedo-Winnie-The-Pooh"),
                       # ("https://imgflip.com/meme/","176944602/Fancy-pooh"),
                       # ("https://imgflip.com/meme/","Blank-Nut-Button"),
                       # ("https://imgflip.com/meme/","50421420/Disappointed-Black-Guy"),
                       # ("https://imgflip.com/meme/","Is-This-A-Pigeon"),
                       # ("https://imgflip.com/meme/","Distracted-Boyfriend"),
                       # ("https://imgflip.com/meme/","Epic-Handshake"),
                       # ("https://imgflip.com/meme/","318009221/Teachers-Copy"),
                       # ("https://imgflip.com/meme/","Running-Away-Balloon"),
                       # ("https://imgflip.com/meme/","Clown-Applying-Makeup"),
                       # ("https://imgflip.com/meme/","252758727/Mother-Ignoring-Kid-Drowning-In-A-Pool"),
                       # ("https://imgflip.com/meme/","224514655/Anime-Girl-Hiding-from-Terminator"),
                       # ("https://imgflip.com/meme/","232844223/Soyboy-Vs-Yes-Chad"),
                       # ("https://imgflip.com/meme/","286646496/Both-Buttons-Pressed"),
                       # ("https://imgflip.com/meme/","139971723/Spongebob-Burning-Paper"),
                       # ("https://imgflip.com/meme/","188789496/Moe-throws-Barney"),
                       # ("https://imgflip.com/meme/","93160967/Skinner-Out-Of-Touch"),
                       # ("https://imgflip.com/meme/","103123450/Spider-Man-Double"),
                       # ("https://imgflip.com/meme/","36698509/kermit-window"),
                       # ("https://imgflip.com/meme/","197846899/Mr-incredible-mad"),
                       # ("https://imgflip.com/meme/","309668311/Two-Paths"),
                       # ("https://imgflip.com/meme/","Squidward"),
                       # ("https://imgflip.com/meme/","153452716/This-is-Worthless"),
                       # ("https://imgflip.com/meme/","243100573/If-those-kids-could-read-theyd-be-very-upset"),
                       # ("https://imgflip.com/meme/","72598094/Ew-i-stepped-in-shit"),
                       # ("https://imgflip.com/meme/","81332206/Hide-the-Pain-Harold"),
                       # ("https://imgflip.com/meme/","174791714/Weak-vs-Strong-Spongebob"),
                       # ("https://imgflip.com/meme/","99924861/Pretending-To-Be-Happy-Hiding-Crying-Behind-A-Mask"),
                       # ("https://imgflip.com/meme/","213473065/Cuphead-Flower"),
                       # ("https://imgflip.com/meme/","Arthur-Fist"),
                       # ("https://imgflip.com/meme/","19750160/Feels-Good-Man"),
                       # ("https://imgflip.com/meme/","214162718/Tuxedo-Winnie-the-Pooh-grossed-reverse"),
                       # ("https://imgflip.com/meme/","174489685/They-are-the-same-picture"),
                       # ("https://imgflip.com/meme/","The-Scroll-Of-Truth"),
                       # ("https://imgflip.com/meme/","Left-Exit-12-Off-Ramp"),
                       # this is fine isn't taken due to the large number of bold strokes in the image, which isn't OCR friendly.
                       # ("https://imgflip.com/meme/","119215120/Types-of-Headaches-meme"),
                       # ("https://imgflip.com/meme/","136553931/This-Is-Brilliant-But-I-Like-This"),
                       ]
    meme_cap = 100
    
    # list_of_targets = [("https://imgflip.com/tag/","covid-19")]
    # list_of_targets = [("https://imgflip.com/tag/","ukraine")]
    # list_of_targets = [("https://imgflip.com/tag/","money")]
    # list_of_targets = [("https://imgflip.com/tag/","school")]
    
    
    informationdump = "_labelout.json"

    driver = webdriver.Firefox()
    
        
    for tagtarget in list_of_targets:
        if "/" in tagtarget[1]:
            desired_dumptarget = tagtarget[0].split("/")[-2] + "_" + tagtarget[1].split("/")[-1] + informationdump
        else:
            desired_dumptarget = tagtarget[0].split("/")[-2] + "_" + tagtarget[1] + informationdump
        
        if os.path.exists(desired_dumptarget):
            with open(desired_dumptarget,"r",encoding="utf-8") as labelfile:
                overall_pulled = json.load(labelfile)
        else:
            overall_pulled = []
        
        for i in overall_pulled:
            seen_links.add(i["imgsrc"])
    
    
        driver.get(tagtarget[0]+tagtarget[1])
        
        if tagtarget[0] == "https://imgflip.com/tag/":
            isclear = True
        elif tagtarget[0] =="https://imgflip.com/meme/":
            isclear = False
        else:
            raise ValueError("Unknown if pulled data is 'CLEAR' of noncontaminant memes via tagging.")
        
        original_window_id = driver.current_window_handle
        tryoutcount = 0
        while True:
            try:
                print(len(overall_pulled))
                if len(overall_pulled)>meme_cap: # break after grabbing X amount of a meme.
                    print("Pulled more than",meme_cap,"of ",tagtarget[1], "memes")
                    break
                tryoutcount+=1
                if tryoutcount>3:
                    print("Tried a page too many times... changing targets")
                    break
                time.sleep(1)
                outlist,seen_links_temp,all_seen_tags_temp = page_extraction(driver,original_window_id,all_seen_tags,seen_links,[],isclear)
                
                # outlist,seen_links_temp,all_seen_tags_temp = page_extraction(driver,original_window_id,all_seen_tags,seen_links,ukraine_filter,isclear)
                
                # outlist,seen_links_temp,all_seen_tags_temp = page_extraction(driver,original_window_id,all_seen_tags,seen_links,money_filter,isclear)
                
                # outlist,seen_links_temp,all_seen_tags_temp = page_extraction(driver,original_window_id,all_seen_tags,seen_links,school_filter,isclear)

                
                
                # print(outlist)
                if not outlist and not seen_links_temp and not all_seen_tags_temp: # reloop. the page didn't load properly
                # this is due to javascript only loading some divs into proper images as you scroll downwards.
                    continue
                overall_pulled.extend(outlist)
                seen_links = seen_links_temp # page check did not fail, can update seen links
                all_seen_tags = all_seen_tags_temp # page check did not fail, update all seen tags.
                
                next_page = driver.find_element(By.CSS_SELECTOR , "a.pager-next.l.but") # MIGHT BE WRONG
                driver.execute_script("arguments[0].scrollIntoView();",next_page)
                ActionChains(driver).click(next_page).perform()
                tryoutcount=0
                
            except selenium.common.exceptions.NoSuchElementException as e:
                print("----------------------------")
                print(e)
                print(type(e))
                print("----------------------------")
                continue
            
            
                
                
                
            # driver.switch_to.new_window('tab')
            # driver.switch_to.window(original_window)
            with open(desired_dumptarget,"w",encoding="utf-8") as dumpfile_target:
                json.dump(overall_pulled,dumpfile_target,indent=4)

    driver.close()
        
    driver.quit()

        


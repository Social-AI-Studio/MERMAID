import os
import torch
import pprint
import time
import json
import utils
import cProfile
import random
import itertools
import sklearn.metrics
import numpy as np
from io import BytesIO    
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from tokeniser_prepper import BERT_preparation, ROBERTA_preparation, VILT_preparation, BLIP_preparation, BLIP_2_preparation, FLAVA_preparation, CLIP_preparation, VisualBERT_preparation, data2vec_preparation
from model_dataset_class import text_vector_dataset_extractor, entity_error_analysis,dual_dataset,dataset_list_split
from attached_heads import trilinear_head,deeper_trilinear_head_relations
torch.set_num_threads(2)
torch.set_num_interop_threads(2)


# we need to compare the ocr outputs to the  final dataset texts. If the Entity cannot be directly found in the OCR, we will mark it as wrong right away.


def access_target_original_parse_textboxes(target_file):
    # Returns a dictionary [imagefile], textboxlist
    # given the target original OCR (uncorrected file)
    with open(target_file,"r",encoding="utf-8") as original_parsefile:
        original_parses = json.load(original_parsefile)
    allparses_sorted = {}
    for item in original_parses:
        proposed_textboxes = []
        key = item["data"]["image"].split("/")[-1]
        for textbox in item["data"]["internal_record"]:
            proposed_textboxes.append(item["data"]["internal_record"][textbox][0].lower())
        allparses_sorted[key] = proposed_textboxes
    return allparses_sorted


if __name__=="__main__":
    labels_file = "final_dataset_cleared.json"
    dataset = dual_dataset(labels_file,target_tokeniser=["bert","bert"])
    # dual_dataset(labels_file,target_tokeniser=["roberta","roberta"])
    all_parses_dict = access_target_original_parse_textboxes(os.path.join("parse","label_studio_reference_input_OCR_INITIAL.json"))
        
    
    possible_textbox_combinatorial = [0,0]
    
    OCR_valid_images = set()
    for item in dataset.savelist:
        entity_dict = {}
        itemkey = item["source_image"]
        for true_textbox in item["correct_answers"]:
            if "ACTUAL_TEXT" in item["correct_answers"][true_textbox]:
                entity_dict[item["correct_answers"][true_textbox]["ACTUAL_TEXT"][0].lower()] = False
        proposed_boxes = all_parses_dict[itemkey]
        for item in entity_dict:
            if item in proposed_boxes:
                entity_dict[item] = True
        for item in entity_dict:
            possible_textbox_combinatorial[1] +=1
            if entity_dict[item]:
                possible_textbox_combinatorial[0] +=1
                OCR_valid_images.add(itemkey)
    print(len(OCR_valid_images))
    print(possible_textbox_combinatorial)
    with open("OCR_valid_images_list.json","w",encoding="utf-8") as dump_OCR_valid_file:
        json.dump(list(OCR_valid_images),dump_OCR_valid_file,indent=4)
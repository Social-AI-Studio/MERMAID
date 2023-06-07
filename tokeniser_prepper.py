import os
import torch
import pprint
import time
import json
import utils
import cProfile
import itertools
import PIL.Image
import sklearn.metrics
import numpy as np
from io import BytesIO    
from PIL import Image
#BERT
from transformers import BertTokenizer, BertModel

#BLIP
from transformers import AutoProcessor, BlipModel

#BLIP-2
from transformers import Blip2Processor, Blip2Model

#CLIP
from transformers import CLIPProcessor, CLIPTextModel, CLIPVisionModel

# Data2vec
from transformers import RobertaTokenizer, Data2VecTextModel
from transformers import AutoImageProcessor, Data2VecVisionModel

# ROBERTA
from transformers import RobertaTokenizer, RobertaModel # for reference repeat tokenizer import.

# VisualBERT
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from utils import Config
from transformers import VisualBertModel, BertTokenizerFast

#VILT
from transformers import ViltProcessor, ViltModel

#FLAVA
from transformers import AutoTokenizer, FlavaTextModel, FlavaImageModel, AutoImageProcessor

#ALIGN
from transformers import AutoTokenizer, AlignTextModel, AutoProcessor, AlignVisionModel

# GPT-NEO
from transformers import GPTNeoModel, GPT2Tokenizer





# Contains the "Preparation classes for running. Acts as the Text/Image tokeniser classes for the models.
# BERT
class BERT_preparation:
    def __init__(self,prefix,device):
        self.embedsize = 768
        self.word_only = True
        self.model_type = "bert"
        self.model_name = "BERT_" + prefix
        self.texttokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device
    
    def __call__(self,targetstringslist):
        outputtextlist = []
        for textstring in targetstringslist:
            textinputs = self.texttokenizer(textstring, return_tensors="pt",padding="longest",truncation=True).to(self.device)
            outputs = self.model(**textinputs)  # shape = 6,combined_shape
            # print("huggingface model runtime:",time.time()-huggingfacemodelstart)
            outputtextlist.append(outputs)
        return outputtextlist,None

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
        
    def return_statedicts(self):
        return [[self.model.state_dict(),"BERT"]]
        
    def return_model(self):
        return [[self.model,"BERT"]]
        # return self model for loading.
        
        

# ROBERTA
class ROBERTA_preparation:
    def __init__(self,prefix,device):
        self.embedsize = 768
        self.word_only = True
        self.model_type = "roberta"
        self.model_name = "RoBERTa_"+ prefix
        self.texttokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model = RobertaModel.from_pretrained("roberta-base").to(device)
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device
    
    def __call__(self,targetstringslist):
        outputtextlist = []
        for textstring in targetstringslist:
            textinputs = self.texttokenizer(textstring, return_tensors="pt",padding="longest",truncation=True).to(self.device)
            # print(textinputs["input_ids"].tolist(),"LENGTH:",sum(textinputs["attention_mask"].tolist()[0]))
            outputs = self.model(**textinputs)
            outputtextlist.append(outputs)
        return outputtextlist, None
    
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()

    def return_statedicts(self):
        return [[self.model.state_dict(),"roberta"]]
        
    def return_model(self):
        return [[self.model,"roberta"]]
# VILT
class VILT_preparation:
    def __init__(self,prefix,device):
        self.word_only = False
        self.model_type = "vilt"
        self.model_name = "VILT_"+ prefix
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device)
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device

        
    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        combination_outputs = []
        for targetstring in targetstringslist:
            inputs = self.processor(opened_image, targetstring, return_tensors="pt",padding="longest",truncation=True).to(self.device)
            # multiply. because they need to have the same shaped inputs.
            # print(inputs["pixel_values"].shape)
            output = self.model(**inputs)
            combination_outputs.append(output)
        return combination_outputs

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
        
    def return_statedicts(self):
        return [[self.model.state_dict(),"vilt"]]

#BLIP
class BLIP_preparation:
    def __init__(self,prefix,device):
        self.imagembed_size1 = 512 # image
        self.imagembed_size2 = 1 # image
        self.textembed_size = 512
        self.word_only = False
        self.model_type = "blip"
        self.model_name = "BLIP_"+prefix
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device
        
        

    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        textlist = []
        # print(targetstringslist)
        processed_inputs = self.processor(images=opened_image,return_tensors="pt",padding="longest",truncation=True).to(self.device)
        imageoutput = self.model.get_image_features(**processed_inputs)
        for targetstring in targetstringslist:
            processed_inputs = self.processor(text=targetstring,return_tensors="pt",padding="longest",truncation=True).to(self.device)
            outputs = self.model.get_text_features(**processed_inputs)
            # print(outputs.shape)
            textlist.append({"pooler_output":outputs})
        # print(len(targetstringslist))
        # print(len(textlist))
        # input()
        # print(outputs)
        # print(textlist)
        return textlist,{"last_hidden_state":imageoutput.unsqueeze(0)}

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
        
    def return_statedicts(self):
        return [[self.model.state_dict(),"blip"]]
    
    def return_model(self):
        return [[self.model,"blip"]]

#BLIP-2
class BLIP_2_preparation:
    def __init__(self,prefix,device):
        self.imagembed_size1 = 768 # image
        self.imagembed_size2 = 50 # image
        self.textembed_size = 512
        self.word_only = False
        self.model_type = "blip2"
        self.model_name = "BLIP2_"+prefix
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device
        
    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        textlist = []
        processed_inputs = self.processor(images=opened_image,return_tensors="pt",padding=True,truncation=True)
        imageoutput = self.model.get_image_features(**processed_inputs)
        for targetstring in targetstringslist:
            processed_inputs = self.processor(text=targetstring,images=opened_image,return_tensors="pt",padding=True,truncation=True)
            outputs = self.model(**processed_inputs)
            textlist.append(outputs)
        return textlist,imageoutput

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()


    def return_statedicts(self):
        return [[self.model.state_dict(),"blip2"]]
        
        
    def return_model(self):
        return [[self.model,"blip2"]]


# FLAVA
class FLAVA_preparation:
    def __init__(self,prefix,device):
        self.imagembed_size1 = 768 # image
        self.imagembed_size2 = 197 # image
        self.textembed_size = 768
        self.word_only = False
        self.model_type = "flava"
        self.model_name = "FLAVA_"+ prefix
        self.processor = AutoTokenizer.from_pretrained("facebook/flava-full")
        self.imageprocessor = AutoImageProcessor.from_pretrained("facebook/flava-full")
        self.imagemodel = FlavaImageModel.from_pretrained("facebook/flava-full").to(device)
        self.textmodel = FlavaTextModel.from_pretrained("facebook/flava-full").to(device)
        self.optimizer_parameters = list(self.textmodel.parameters())+ list(self.imagemodel.parameters())
        self.device = device
        

    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        textlist = []
        for targetstring in targetstringslist:
            text_embed_input = self.processor(targetstring, padding="longest",truncation=True, return_tensors="pt").to(self.device)
            textembed_outputs = self.textmodel(**text_embed_input)
            textlist.append(textembed_outputs)
        image_embed_input = self.imageprocessor(images=opened_image, return_tensors="pt").to(self.device)
        imageembed_outputs = self.imagemodel(**image_embed_input)
        # print("huggingface model runtime:",time.time()-huggingfacemodelstart)
        return textlist, imageembed_outputs

    def train(self):
        self.textmodel.train()
        self.imagemodel.train()
    def eval(self):
        self.textmodel.eval()
        self.imagemodel.eval()

    def return_statedicts(self):
        return [[self.textmodel.state_dict(),"flava_text"],[self.imagemodel.state_dict(),"flava_image"]]
    
    def return_model(self):
        return [[self.textmodel,"flava_text"],[self.imagemodel,"flava_image"]]



# CLIP
class CLIP_preparation:
    def __init__(self,prefix,device):
        self.imagembed_size1 = 768 # image
        self.imagembed_size2 = 50 # image
        self.textembed_size = 512
        self.word_only = False
        self.model_type = "clip"
        self.model_name = "CLIP_"+ prefix
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.imagemodel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.textmodel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.optimizer_parameters = list(self.textmodel.parameters())+ list(self.imagemodel.parameters())
        self.device = device
        

    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        textlist = []
        for targetstring in targetstringslist:
            text_embed_input = self.processor(targetstring, padding="longest",truncation=True, return_tensors="pt").to(self.device)
            textembed_outputs = self.textmodel(**text_embed_input)
            textlist.append(textembed_outputs)
        image_embed_input = self.processor(images=opened_image, return_tensors="pt").to(self.device)
        imageembed_outputs = self.imagemodel(**image_embed_input)
        # print("huggingface model runtime:",time.time()-huggingfacemodelstart)
        return textlist, imageembed_outputs

    def train(self):
        self.textmodel.train()
        self.imagemodel.train()
    def eval(self):
        self.textmodel.eval()
        self.imagemodel.eval()

    def return_statedicts(self):
        return [[self.textmodel.state_dict(),"clip_text"],[self.imagemodel.state_dict(),"clip_image"]]
    
    def return_model(self):
        return [[self.textmodel,"clip_text"],[self.imagemodel,"clip_image"]]


#VisualBERT
class VisualBERT_preparation:
    def __init__(self,prefix,device):
        # Visual BERT (Do not use the one in documentation if detectron2 can't install since it's finnicky)     https://github.com/huggingface/transformers/tree/main/examples/research_projects/visual_bert
        self.device = device
        self.word_only = False
        self.model_type = "vilbert"
        self.model_name = "VisualBERT_"+ prefix
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg._pointer["model"]._pointer["device"] = self.device
        
        self.region_imagemodel = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(self.device)
        # for some reason, shifting the model via to device doesn't work
        
        self.image_preprocess = Preprocess(self.frcnn_cfg,device) 
        # Line 85 in processing_image.py has been edited to save device according to device specified from this class.
        # other necessary adjustments were made.
        # but it doesn't actually adjust the vector to the selected device. so we're just going to hardcode it in __call__
        
        self.internal_config = self.frcnn_cfg
        self.internal_preprocessor = self.image_preprocess
        self.texttokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)
        self.optimizer_parameters = list(self.region_imagemodel.parameters())+list(self.model.parameters())

        

    def __call__(self,targetstringslist,single_image_addr):
        images, sizes, scales_yx = self.internal_preprocessor(single_image_addr)
        output_dict = self.region_imagemodel(
            images.to(self.device),
            sizes.to(self.device),
            scales_yx=scales_yx.to(self.device),
            padding="max_detections",
            max_detections=self.internal_config.max_detections,
            return_tensors="pt",
        )
        features = output_dict.get("roi_features").to(self.device)
        outputlist = []
        for targetstring in targetstringslist:
            inputs = self.texttokenizer(targetstring, return_tensors="pt",padding="longest",truncation=True).to(self.device)
            # print(len(inputs["input_ids"][0]))
            outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    visual_embeds=features,
                    visual_attention_mask=torch.ones(features.shape[:-1]).to(self.device),
                    token_type_ids=inputs.token_type_ids,
                    output_attentions=False)
            # print("huggingface model runtime:",time.time()-huggingfacemodelstart)
            outputlist.append(outputs)
        return outputlist
    
    def train(self):
        self.region_imagemodel.train()
        self.model.train()
    def eval(self):
        self.region_imagemodel.eval()
        self.model.eval()
    
    def return_statedicts(self):
        return [[self.region_imagemodel.state_dict(),"region_proposalFRCNN"],[self.model.state_dict(),"visualbert"]]
    

#data2vec
class data2vec_preparation:
    def __init__(self,prefix,device):
        self.imagembed_size1 = 768
        self.imagembed_size2 = 197
        self.textembed_size = 768

        self.device = device
        self.word_only = False
        self.model_type = "data2vec"
        self.model_name = "data2vec_"+prefix
        self.imagetokenizer = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        self.imagemodel = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base").to(self.device)
        self.texttokenizer = RobertaTokenizer.from_pretrained("facebook/data2vec-text-base")
        self.textmodel = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base").to(self.device)
        self.optimizer_parameters = list(self.imagemodel.parameters()) + list(self.textmodel.parameters())
        

    def __call__(self,targetstringslist,single_image_addr):
        opened_image = Image.open(single_image_addr)
        image_inputs = self.imagetokenizer(opened_image, return_tensors="pt").to(self.device)
        imageembed_outputs = self.imagemodel(**image_inputs)
        textoutputlist = []

        for targetstring in targetstringslist:
            processed_target_string = self.texttokenizer(targetstring, return_tensors="pt",padding="longest",truncation=True).to(self.device)
            text_outputs = self.textmodel(**processed_target_string)
            textoutputlist.append(text_outputs)
        # print(text_outputs["pooler_output"].shape)
        return textoutputlist, imageembed_outputs

    def train(self):
        self.textmodel.train()
        self.imagemodel.train()
    def eval(self):
        self.textmodel.eval()
        self.imagemodel.eval()
        
    def return_statedicts(self):
        return [[self.imagemodel.state_dict(),"data2vec_image"],[self.textmodel.state_dict(),"data2vec_text"]]

    def return_model(self):
        return [[self.imagemodel,"data2vec_image"],[self.textmodel,"data2vec_text"]]





class GPT_NEO_preparation:
    # 2.7B 
    def __init__(self,prefix,device):
        self.embedsize = 2560
        self.word_only = True
        self.model_type = "gpt_neo_SINGLISH"
        self.model_name = "gpt_neo_SINGLISH" + prefix
        self.model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
        self.texttokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        self.optimizer_parameters = list(self.model.parameters())
        self.device = device
        
        
    
    def __call__(self,targetstringslist):
        outputtextlist = []
        for textstring in targetstringslist:
            textinputs = self.texttokenizer(textstring, return_tensors="pt",truncation=True).to(self.device)
            outputs = self.model(**textinputs)  # shape = 6,combined_shape
            # print("huggingface model runtime:",time.time()-huggingfacemodelstart)
            outputtextlist.append(outputs)
        return outputtextlist,None

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
        
    def return_statedicts(self):
        return [[self.model.state_dict(),"BERT"]]
        
    def return_model(self):
        return [[self.model,"BERT"]]
        # return self model for loading.
        
        
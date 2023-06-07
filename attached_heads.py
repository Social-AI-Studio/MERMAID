import os
import torch
import pprint
import time
import json
import utils
import gc
import PIL.Image
import numpy as np
from torchvision.transforms import ToTensor


class trilinear_head(torch.nn.Module):
    def __init__(self,device,embed_dimension=768,targetsize = 2):
        super(trilinear_head,self).__init__()

        self.device = device
        self.linear1 = torch.nn.Linear(embed_dimension,256)
        self.linear2 = torch.nn.Linear(256,64)
        self.linear3 = torch.nn.Linear(64,targetsize) 
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, input_processed_string):
        linear1out = self.linear1(input_processed_string)
        linear2out = self.linear2(linear1out)
        linear3out = self.linear3(linear2out)
        return linear3out
  
        
        
        
class deeper_trilinear_head_relations(torch.nn.Module):
    def __init__(self,device,image_embed1=768,image_embed2=50,text_embed=512,targetsize = 9,parameter=True,wordonly=False,noposition=False):
        super(deeper_trilinear_head_relations,self).__init__()
        self.model_name = "deeper"
        self.device = device
        self.noposition = noposition
        self.wordonly = wordonly
        self.image_embed1 = image_embed1
        self.image_embed2 = image_embed2
        self.text_embed = text_embed
        self.targetsize = targetsize
        
        if parameter:
            completion = []
            self.text_parameter = torch.nn.parameter.Parameter(torch.empty(1,text_embed).normal_(mean=1,std=0.5)).to(self.device)
            completion.append(self.text_parameter.squeeze())
            
            if not wordonly:
                self.image_parameter = torch.nn.parameter.Parameter(torch.empty(image_embed2,image_embed1).normal_(mean=1,std=0.5)).to(self.device)
                completion.append(self.image_parameter.flatten())
            
            
            if not noposition:
                self.position_parameter = torch.nn.parameter.Parameter(torch.empty(4).normal_(mean=1,std=0.5)).to(self.device)
                completion.append(self.position_parameter)
            
            self.parameter = torch.cat(completion,dim=0).to(self.device)
        else:
            self.parameter = None
        
        if not noposition:
            self.linear_positional1 = torch.nn.Linear(4,image_embed1)
            self.linear_positional2 = torch.nn.Linear(image_embed1,image_embed1*image_embed2)
        
        if not wordonly:
            self.linear1_receiver = torch.nn.Linear(image_embed1*image_embed2+text_embed,1024).to(self.device)
            self.linear1_sender = torch.nn.Linear(image_embed1*image_embed2+text_embed,1024).to(self.device)
            self.linear2 = torch.nn.Linear(1024*2,512).to(self.device)
            self.linear3 = torch.nn.Linear(512,targetsize).to(self.device)
        else:
            self.linear1_receiver = torch.nn.Linear(text_embed,1024).to(self.device)
            self.linear1_sender = torch.nn.Linear(text_embed,1024).to(self.device)
            self.linear2 = torch.nn.Linear(1024*2,512).to(self.device)
            self.linear3 = torch.nn.Linear(512,targetsize).to(self.device)
            
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, image1,position1,text1, image2,position2,text2):
        if not self.noposition:
            position1 = torch.stack(position1)
            position2 = torch.stack(position2)
        # print(len(image1))
        if self.image_embed2==1: # the squeeze step from externally causes shape issues.
            holder = []
            for item in image1:
                if len(item.shape)==1:
                    holder.append(item.unsqueeze(0))
                else:
                    holder.append(item)
            image1 = torch.stack(holder)
            # print("image1stack:",image1.shape)
            
            holder = []
            for item in image2:
                if len(item.shape)==1:
                    holder.append(item.unsqueeze(0))
                else:
                    holder.append(item)
            image2 = torch.stack(holder)
            # print("image2stack:",image2.shape)
            
        else:
            image1 = torch.stack(image1)
            image2 = torch.stack(image2)
        text1 = torch.stack(text1)
        text2 = torch.stack(text2)
        
        if self.wordonly:
            senderout = self.linear1_sender(text1)
            receiverout = self.linear1_receiver(text2)
        
        elif not self.noposition:
            positional_out1_1 = self.linear_positional1(position1)
            positional_out1_2 = self.linear_positional2(positional_out1_1)
        
            positional_out2_1 = self.linear_positional1(position2)
            positional_out2_2 = self.linear_positional2(positional_out2_1)
            # print(image1.shape,positional_out1_2.shape)
            positional_out1_2 = positional_out1_2.reshape(-1,self.image_embed2,self.image_embed1)
            positional_out2_2 = positional_out2_2.reshape(-1,self.image_embed2,self.image_embed1)
            dotted_1 = image1*positional_out1_2
            dotted_2 = image2*positional_out2_2
            # print(dotted_1.reshape(-1,self.image_embed2*self.image_embed1).shape,text1.squeeze().shape)
            senderout = self.linear1_sender(torch.cat([dotted_1.reshape(-1,self.image_embed2*self.image_embed1),text1.squeeze()],dim=1))
            receiverout = self.linear1_receiver(torch.cat([dotted_2.reshape(-1,self.image_embed2*self.image_embed1),text2.squeeze()],dim=1))
            
        else:
            # print(image1.shape)
            # print(text1.squeeze().shape)
            # print(self.linear1_sender)
            senderout = self.linear1_sender(torch.cat([image1.reshape(image1.shape[0],-1),text1.squeeze()],dim=1))
            receiverout = self.linear1_receiver(torch.cat([image2.reshape(image1.shape[0],-1),text2.squeeze()],dim=1))
                
        
        linear2out = self.linear2(torch.cat([senderout,receiverout],dim=1))
        
        linear3out = self.linear3(linear2out)
        return linear3out
        
        
        
        
        
        
        
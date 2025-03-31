# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:34:41 2023

@author: ICML4119 AUTHORS
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

from models.inceptiontime import InceptionTimeFeatureExtractor


from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            

    
class ItimeNet(nn.Module):
    def __init__(self, in_features, n_classes=2, mDim=64, max_seq_len=400,dropout=0.):
        super().__init__()
     

        # define backbone Can be replace here
        self.feature_extractor = InceptionTimeFeatureExtractor(n_in_channels=in_features )
        self.in_features = in_features
        self._fc1 = nn.Sequential(
            nn.Linear(mDim,mDim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
        ) 
        
        self._fc2 = nn.Linear(mDim, n_classes)
        
        
        initialize_weights(self)
        
        
    def forward(self, x,warmup=False,view_feat=False):
        ############### test InsNorm
#         x_enc = x
#         means = x_enc.mean(1, keepdim=True)#+self.mean_gain
            
            
#         x_enc_original = x_enc


#         x_enc = x_enc - means
#         stdev = torch.sqrt(
#             torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5) #+ self.var_gain
#         x_enc = x_enc/ stdev

#         x_enc = x_enc
        
#         x = x_enc
      ##########
        
        x1 = self.feature_extractor(x.transpose(1, 2))
        x1 = x1.transpose(1, 2)
        x= x1

        B, seq_len, D = x.shape

        
        view_x = x.clone()
        
      
        global_token = x.mean(dim=1)#[0]
      
        
        x = self._fc1(global_token)
        feat = x
        logits = self._fc2(x)
            
        if view_feat == True:
        
            return logits,feat
        else:
            return logits


if __name__ == "__main__":
    x = torch.randn(3, 400, 4).cuda()
    model = ItimeNet(in_features=4,mDim=128).cuda()
    ylogits =model(x)
    print(ylogits.shape)


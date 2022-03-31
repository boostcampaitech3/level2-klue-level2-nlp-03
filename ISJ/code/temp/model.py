import pickle as pickle
import os
import pandas as pd
import torch
import torch.nn as nn
import sklearn
import numpy as np
from transformers import AutoTokenizer, AutoConfig, EarlyStoppingCallback, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, SequenceClassifierOutput

import warnings
warnings.filterwarnings('ignore')



class ThreeWayLSTMModels(nn.Module):
    
    def __init__(self, path):
        super(self, ThreeWayLSTMModels).__init__()

        c1 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=2)
        c2 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=29)
        c3 = AutoConfig.from_pretrained('klue/roberta-large', num_labels=30)

        self.hidden_dim= c1.hidden_size
        
        self.roberta1 = AutoModelForSequenceClassification.from_pretrained("path", config=c1)
        self.roberta2 = AutoModelForSequenceClassification.from_pretrained("path", config=c2)
        self.roberta3 = AutoModelForSequenceClassification.from_pretrained("path", config=c3)
        
                
        for p in self.roberta1.parameters():
            p.requires_grad = False
        for p in self.roberta2.parameters():
            p.requires_grad = False
        for p in self.roberta3.parameters():
            p.requires_grad = False

        self.fc1 = nn.Linear(2, self.hidden_dim)
        self.fc2 = nn.Linear(29, self.hidden_dim)
        self.fc3 = nn.Linear(30, self.hidden_dim)

        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim * 15, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.2)
        )

        self.lstm= nn.LSTM(input_size= self.hidden_dim, hidden_size= self.hidden_dim, num_layers= 2, dropout= 0.2,
                            batch_first= True, bidirectional= True)
        
        self.fc= nn.Linear(self.hidden_dim*2, self.model_config.num_labels)

    def forward(self, input_ids, attention_mask):
        logits_1 = self.roberta1(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_2 = self.roberta2(
            input_ids.clone(), attention_mask=attention_mask).get('logits')
        logits_3 = self.roberta3(
            input_ids.clone(), attention_mask=attention_mask).get('logits')

        logits_1 = self.fc1(logits_1)
        logits_2 = self.fc2(logits_2)
        logits_3 = self.fc1(logits_3)

        threeway = torch.cat((logits_1, logits_2, logits_3), dim=-1)
        # 가로로 결과가 나열 될 것

        output = self.linear(threeway)
        # 그것을 가지고 linear를 한번 진행해준다.
        # 진행하지 않고 LSTM에 바로 넣어도 괜찮을 것 같기는 하다.

        # 그리고 그것을 LSTM에 태우는 구조
        hidden, (last_hidden, last_cell)= self.lstm(output)
        
        # layer를 두개 쌓았으므로 걔네에 대해서 fc를 진행하는 것
        # 하나로도 가능하지 않을까 싶다.
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim= 1)
        
        logits= self.fc(cat_hidden)

        outputs = SequenceClassifierOutput(logits=logits)

        return outputs
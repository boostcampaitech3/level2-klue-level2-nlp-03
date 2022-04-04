import pickle as pickle
import os
import pandas as pd
import torch
from preprocss import *
# added by sujeong;
from entity_marker import *

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset, augmentation):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  
  if augmentation == "ONLY_AUG":
    subject_entity = []
    object_entity = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      #i = eval(i)['word']
      #j = eval(j)['word']
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['en_trans_sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    return out_dataset
  
  elif augmentation == "AUG":
    subject_entity = []
    object_entity = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      #i = eval(i)['word']
      #j = eval(j)['word']
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    out_dataset_source = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    
    subject_entity = []
    object_entity = []
    
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      #i = eval(i)['word']
      #j = eval(j)['word']
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    
    out_dataset_aug = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['en_trans_sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})

    out_dataset = pd.concat([out_dataset_source, out_dataset_aug])
    
    return out_dataset
  
  else:  #default = "NO_AUG"
    subject_entity = []
    object_entity = []
    for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
      #i = eval(i)['word']
      #j = eval(j)['word']
      i = i[1:-1].split(',')[0].split(':')[1]
      j = j[1:-1].split(',')[0].split(':')[1]

      subject_entity.append(i)
      object_entity.append(j)
    out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
    return out_dataset


def load_data(dataset_dir, augmentation, add_entity_marker, entity_marker_type, data_preprocessing):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  
  # 중복 데이터 제거를 위한 전처리
  pd_dataset = preprocess(pd_dataset)

  # entity marker 추가
  # added by sujeong;
  print("add_entity_marker : " , add_entity_marker)
  print("entity_marker_type: ", entity_marker_type)
  if add_entity_marker:
    pd_dataset = get_entity_marked_data(df=pd_dataset, marker_type=entity_marker_type)

  # 데이터 전처리 (중복 괄호 제거, 이상한 문장 부호 수정, 연속된 공백 수정)
  # added by sujeong;
  print("data_preprocessing : ", data_preprocessing)
  if data_preprocessing:
    run_preprocess(pd_dataset)

  # 데이터셋으로 제작
  dataset = preprocessing_dataset(pd_dataset, augmentation)

  return dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  print(concat_entity)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def get_cls_list(pd_dataset):
    labels = label_to_num(pd_dataset['label'])
    _, distr = np.unique(labels, return_counts=True)
    return distr
from distutils.command.config import config
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModel
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from arguments import get_args

from custom.trainer import customTrainer
from models.custom_roberta import customRobertaForSequenceClassification

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer, add_entity_marker, entity_marker_type, data_preprocessing):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir, "NO_AUG", add_entity_marker, entity_marker_type, data_preprocessing)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  if args.add_entity_marker: # added by sujeong; if entity marker==True, add special token.
    added_token_num, tokenizer = add_special_token(tokenizer, args.entity_marker_type) 
  
  
  MODEL_NAME = args.model_dir

  model_config = AutoConfig.from_pretrained(Tokenizer_NAME)
  model_config.update({"head_type": args.head_type})
  model_config.num_labels = 30
  if args.add_entity_marker: # added by sujeong; if entity marker==True, add special token.
    model_config.vocab_size+=added_token_num
  ## load my model
  
  # model dir
  
  if args.head_type == 'base':
    # 아예 hugging face 지원 구조
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  else:
    model = customRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  
  best_state_dict= torch.load(args.model_dir)
  model.load_state_dict(best_state_dict)

  # model_config= AutoConfig.from_pretrained(Tokenizer_NAME)
  # model_config.num_labels= 30
  # model= AutoModelForSequenceClassification.from_pretrained(Tokenizer_NAME, config= model_config)
  
  # best_state_dict= torch.load(args.model_dir)
  # model.load_state_dict(best_state_dict)
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})
  fold_num = args.fold_num
  if not os.path.exists(f'./prediction/{args.model_name}'):
    os.makedirs(f'./prediction/{args.model_name}')
  output.to_csv(f'./prediction/{args.model_name}/submission_{fold_num}_end.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')


def str2bool(v):
    # arguments에 유사한 값들끼리 다 boolean 처리 되도록

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':    
  # model dir
  for i in range (10):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="roberta-large")
    # entity_marker
    # added by sujeong;
    parser.add_argument("--add_entity_marker", type=str2bool, default=False, help="If you want to add entity marker, set this argument True.")
    parser.add_argument("--entity_marker_type", type=str, default="entity_marker_punc",
                        help="type of entity marker"
                            "`entity_marker`:  [E1] Bill [/E1] was born in [E2] Seattle [/E2]"
                            "`entity_marker_punc`:  @ Bill @ was born in # Seattle #"
                            "`typed_entity_marker`: <S:PERSON> Bill </S:PERSON> was born in <O:CITY> Seattle </O:CITY>."
                            "`typed_entity_marker_punc`:  @ +person+ Bill @ was born in #^city^ Seattle #.")
    # preprocessing # 중복 괄호 제거, 이상한 문장 부호 수정, 연속 공백 수정
    # added by sujeong;
    parser.add_argument("--data_preprocessing", type=str2bool, default=False, help="If you want to make data preprocessed, set this argument True.")
    
    args = parser.parse_args()
    #parser.add_argument('--model_dir', type=str, default=f"./best_model_{i}/{args.model_name}/pytorch_model.bin")
    parser.add_argument('--model_dir', type=str, default=f"./results_with_embedding/fold_{i}/checkpoint-1800/pytorch_model.bin")
    parser.add_argument('--fold_num', type=int, default=i)
    parser.add_argument('--head_type', type=str,
                      default="more_dense")
    args = parser.parse_args()
    print(args)
    main(args)
  

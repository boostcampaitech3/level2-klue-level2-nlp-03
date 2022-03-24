from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from dataset import *

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

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
    print('main inference start')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer= AutoTokenizer.from_pretrained(args.model)

    df_list= []
    for i in range(args.kfold):
        print(f'KFOLD : {i} inference start !')
        
        MODEL_NAME = args.model_dir + f'/model_{i}.bin'
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = args.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        model.model.resize_token_embeddings(tokenizer.vocab_size + args.add_token)

        best_state_dict= torch.load(os.path.join(f'{args.model_path}_{i}', 'pytorch_model.bin'))
        model.load_state_dict(best_state_dict)
        model.to(device)
        
        test_id, test_dataset, test_label= load_test_dataset(args.test_path, tokenizer, args)
        testset= Dataset(test_dataset, test_label)

        pred_answer, output_prob= inference(model, testset, device, args)
        pred_answer= num_to_label(pred_answer)
        print(len(test_id), len(pred_answer), len(output_prob))

        output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        output.to_csv(os.path.join(args.save_dir, f'submission{i}.csv'), index= False)
        
        print(f'KFOLD : {i} inference fin !')
    print('FIN')
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./best_model")
  args = parser.parse_args()
  print(args)
  main(args)
  

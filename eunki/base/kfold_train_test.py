import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoConfig, EarlyStoppingCallback, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer

from load_data import *
from loss import *
# added by sujeong;
from entity_marker import *
from utils import *

from inference import *

from custom.callback import customWandbCallback
    #customTrainerState,customTrainerControl,customTrainerCallback

from custom.trainer import customTrainer,customTrainer2
from models.custom_roberta import customRobertaForSequenceClassification

import warnings
warnings.filterwarnings('ignore')


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    ElectraModel,
    ElectraTokenizer,
    #RobertaForPreTraining
)
from pytorch_lightning.lite import LightningLite

# get arguments
from arguments import get_args

import random
import wandb

def seed_fix(seed):
    """seed setting 함수"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
       'org:product', 'per:title', 'org:alternate_names',
       'per:employee_of', 'org:place_of_headquarters', 'per:product',
       'org:number_of_employees/members', 'per:children',
       'per:place_of_residence', 'per:alternate_names',
       'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
       'per:spouse', 'org:founded', 'org:political/religious_affiliation',
       'org:member_of', 'per:parents', 'org:dissolved',
       'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
       'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
       'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.
  
  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('./dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label


class Lite(LightningLite):

  def run(self, args,exp_full_name,reports='wandb'):
      # load model and tokenizer
      # MODEL_NAME = "bert-base-uncased"
      MODEL_NAME = args.model_name
      tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
      
      # add special tokens
      # added by sujeong; if entity marker==True, add special token.
      if args.add_entity_marker: 
        added_token_num, tokenizer = add_special_token(tokenizer, args.entity_marker_type) 

      # load dataset
      # augmentation 인자를 전달
      # edited by sujeong;(args.add_entity_marker, args.entity_marker_type, args.data_preprocessing 추가)
      print('Loading Data...')
      """
      # edited vy soyeon;(entity type2 + run preprocess 돌린 csv 파일 생성했기 때문에 해당 부분 불러오도록 함) 만약 다른 설정으로 할 경우 수행해야함(약 20분 소요)
      
      """
      #total_train_dataset = load_data(args.train_data_dir, args.augmentation, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)
      # tapt_dataset = load_data("../dataset/test/test_data.csv", args.augmentation, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)
      # total_train_dataset = pd.read_csv('/opt/ml/dataset/train/final_preprocess_entity_marker2.csv')
      total_train_dataset = pd.read_csv('./final_preprocess_entity_marker2.csv')
      print('Done!')

      # 먼저 중복여부 판별을 위한 코드
      total_train_dataset['is_duplicated'] = total_train_dataset['sentence'].duplicated(keep=False)
      
      result = label_to_num(total_train_dataset['label'].values)
      total_train_label = pd.DataFrame(data = result, columns = ['label'])

      # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
      # dev_label = label_to_num(dev_dataset['label'].values)
      
      kfold= StratifiedKFold(n_splits=5, shuffle= True, random_state= 42)
      
      print('Start Training...')
      for fold, (train_idx, val_idx) in enumerate(kfold.split(total_train_dataset, total_train_label)):
        
        print("fold : ", fold)

        #run= wandb.init(project= 'klue', entity= 'boostcamp-nlp3', name= f'KFOLD_{fold}_{args.wandb_path}')
        
        train_dataset= total_train_dataset.iloc[train_idx]
        val_dataset= total_train_dataset.iloc[val_idx]
        train_label = total_train_label.iloc[train_idx]
        val_label = total_train_label.iloc[val_idx]

        train_dataset.reset_index(drop= True, inplace= True)
        val_dataset.reset_index(drop= True, inplace= True)
        train_label.reset_index(drop= True, inplace= True)
        val_label.reset_index(drop= True, inplace= True)
        
        temp = []    
        
        for val_idx in val_dataset.index:
            if val_dataset['is_duplicated'].iloc[val_idx] == True:
                if val_dataset['sentence'].iloc[val_idx] in train_dataset['sentence'].values:
                    train_dataset.append(val_dataset.iloc[val_idx])
                    train_label.append(val_label.iloc[val_idx])
                    temp.append(val_idx)
                
        val_dataset.drop(temp, inplace= True, axis= 0)
        val_label.drop(temp, inplace= True, axis= 0)

        train_label_list = train_label['label'].values.tolist()
        val_label_list = val_label['label'].values.tolist()

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(val_dataset, tokenizer)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label_list)
        RE_dev_dataset = RE_Dataset(tokenized_dev, val_label_list)

        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = args.num_labels
        
        # if args.tapt:
        #   tapt_model = RobertaForPreTraining(model_config)
        #   tapt_model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)

        #   _, tapt_dataset, tapt_label = load_test_dataset(test_dataset_dir, tokenizer, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)
        #   Re_tapt_dataset = RE_Dataset(test_dataset ,test_label)

          # training_args = TrainingArguments(
          #   output_dir=output_dir,  # output directory
          #   num_train_epochs=3,  # total number of training epochs
          #   learning_rate=args.lr,  # learning_rate
          #   per_device_train_batch_size=args.train_bs,  # batch size per device during training
          #   per_device_eval_batch_size=args.eval_bs,  # batch size for evaluation
          #   warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
          #   weight_decay=args.weight_decay,  # strength of weight decay
          #   evaluation_strategy=args.eval_strategy, # evaluation strategy to adopt during training
          #                                           # `no`: No evaluation during training.
          #                                           # `steps`: Evaluate every `eval_steps`.
          #                                           # `epoch`: Evaluate every end of epoch.
          #   eval_steps=args.eval_steps,  # evaluation step.
          #   load_best_model_at_end=args.load_best_model_at_end,
          #   group_by_length= True,
          #   fp16 = True,
          #   #fp16_full_eval=True,
          #   adafactor = True,
          # )
          # trainer = Trainer(
          # model=model,  # the instantiated 🤗 Transformers model to be trained
          # train_dataset=RE_tapt_dataset,  # training dataset
          # compute_metrics=compute_metrics , # define metrics function
          # )

        


        

        
        
        model_config.update({"head_type": args.head_type})
        model = customRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config  = model_config )
        if args.add_entity_marker:
          model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
        model.to(device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        
        
        output_dir = args.output_dir + f'/fold_{fold}'
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            save_total_limit=args.save_total_limit,  # number of total save model.
            save_steps=args.save_steps,  # model saving step.
            num_train_epochs=args.epochs,  # total number of training epochs
            learning_rate=args.lr,  # learning_rate
            per_device_train_batch_size=args.train_bs,  # batch size per device during training
            per_device_eval_batch_size=args.eval_bs,  # batch size for evaluation
            warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,  # strength of weight decay
            logging_dir=args.logging_dir,  # directory for storing logs
            logging_steps=args.logging_steps,  # log saving step.
            evaluation_strategy=args.eval_strategy, # evaluation strategy to adopt during training
                                                    # `no`: No evaluation during training.
                                                    # `steps`: Evaluate every `eval_steps`.
                                                    # `epoch`: Evaluate every end of epoch.
            eval_steps=args.eval_steps,  # evaluation step.
            load_best_model_at_end=args.load_best_model_at_end,
            # https://docs.wandb.ai/guides/integrations/huggingface
            # Hugging face Trainer 내부 integration 된 wandb로 logging
            group_by_length= True,
            fp16 = True,
            #fp16_full_eval=True,
            adafactor = True,
            report_to=reports,
            run_name = exp_full_name,
        )

        cls_list = get_cls_list(train_dataset)

        trainer = customTrainer2(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics , # define metrics function
        callbacks = [customWandbCallback(), EarlyStoppingCallback(early_stopping_patience= 6)],
        cls_list = cls_list,
        add_args = args
        )

        # trainer = Trainer(
        #     model=model,  # the instantiated 🤗 Transformers model to be trained
        #     args=training_args,  # training arguments, defined above
        #     train_dataset=RE_train_dataset,  # training dataset
        #     eval_dataset=RE_dev_dataset,  # evaluation dataset
        #     compute_metrics=compute_metrics,  # define metrics function
        #     callbacks= [EarlyStoppingCallback(early_stopping_patience= 3)]
        # )

        # train model
        trainer.train()
        #model.save_pretrained(args.model_save_dir)
        folder_name = MODEL_NAME.split('/')[-1]
        if not os.path.exists(f'{args.model_save_dir}_{fold}/{folder_name}'):
            os.makedirs(f'{args.model_save_dir}_{fold}/{folder_name}', exist_ok= True)
        torch.save(model.state_dict(), os.path.join(f'{args.model_save_dir}_{fold}/{folder_name}', 'pytorch_model.bin'))
        print(f'{MODEL_NAME} version, fold{fold} fin!')
        
        
        #run.finish()

def make_dirs(args):
    # args에 지정된 폴더가 존재하나 해당 폴더가 없을 경우 대비
    # model save
    model_save_dir = args.model_save_dir
    for i in range(5):
      os.makedirs(model_save_dir + f'/fold_{i}', exist_ok=True)
    # output
    os.makedirs(args.output_dir, exist_ok=True)


def main():

  args = get_args()
  seed_fix(args.seed)
  # make directories
  make_dirs(args)
  # https://docs.wandb.ai/guides/integrations/huggingface

    # 디버깅 때는 wandb 로깅 안하기 위해서
  if args.use_wandb:
    # TODO; 실험 이름 convention은 천천히 정해볼까요?
    exp_full_name = f'{args.user_name}_{args.exp_name}_{args.head_type}_{args.add_entity_marker}'
    wandb.login()
    # sangryul
    # project : 우리 그룹의 프로젝트 이름
    # name : 저장되는 실험 이름
    # entity : 우리 그룹/팀 이름

    wandb.init(project='Final', #args.user_name,
                name=exp_full_name,
                entity='boostcamp-nlp3')
                #entity='boostcamp-nlp3')  # nlp-03
    wandb.config.update(args)

    print('#######################')
    print(f'Experiments name: {exp_full_name}')
    print('#######################')
  else:
    exp_full_name = ''
    print('@@@@@@@@Notice@@@@@@@@@@')
    print('YOU ARE NOT LOGGING RESULTS NOW')
    print('@@@@@@@@$$$$$$@@@@@@@@@@')

  Lite(devices=1, accelerator="gpu", precision="bf16").run(args, exp_full_name)
  # only when using notebook
  # wandb.finish()


if __name__ == '__main__':
  main()

# export WANDB_PROJECT=KLUE
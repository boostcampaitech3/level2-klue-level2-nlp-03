import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import \
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, \
    Trainer, TrainingArguments, \
    RobertaConfig, RobertaTokenizer, \
    RobertaForSequenceClassification, BertTokenizer, \
    EarlyStoppingCallback
from load_data import *
import wandb
import random

from sklearn.metrics import confusion_matrix
from utils import plot_cm_by_num_samples, plot_cm_by_ratio
from custom.callback import customWandbCallback
    #customTrainerState,customTrainerControl,customTrainerCallback
from custom.trainer import customTrainer, customTrainer2,customTrainer3
from train import get_args,seed_fix

from models.custom_roberta_test import customRobertaForSequenceClassification
# from models.custom_roberta import customRobertaForSequenceClassification
def compute_metrics(pred):
    """ validation을 위한 metrics function """
    global label_list

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    # conf = confusion_matrix(labels, preds)
    # fig1 = plot_cm_by_num_samples(conf, label_list)
    # fig2 = plot_cm_by_ratio(conf, label_list)

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
        # 'cm_samples': fig1,
        # 'cm_ratio': fig2
    }

args = get_args()
args.train_data_dir = "/opt/ml/dataset/train/train.csv"
seed_fix(args.seed)

MODEL_NAME = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# added_token_num, tokenizer = add_special_token(tokenizer, args.entity_marker_type)
# total_train_dataset = load_data(args.train_data_dir, args.augmentation, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)

# breakpoint()

train_dataset, eval_dataset = load_split_data(args.train_data_dir,
                                                          args.seed,
                                                          args.eval_ratio,
                                                          augmentation='NO_AUG+IDX')
train_label = label_to_num(train_dataset['label'].values)
eval_label = label_to_num(eval_dataset['label'].values)

# tokenized_train = tokenized_dataset(train_dataset, tokenizer)
# tokenized_eval = tokenized_dataset(eval_dataset, tokenizer)
tokenized_train = tokenized_dataset_IDX(train_dataset, tokenizer)
tokenized_eval = tokenized_dataset_IDX(eval_dataset, tokenizer)
# make dataset for pytorch.
# RE_train_dataset = RE_Dataset(tokenized_train, train_label)
# RE_eval_dataset = RE_Dataset(tokenized_eval, eval_label)
RE_train_dataset = RE_Dataset_IDX(tokenized_train, train_label)
breakpoint()
RE_eval_dataset = RE_Dataset_IDX(tokenized_eval, eval_label)
# RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

print(device)
# setting model hyperparameter
model_config = AutoConfig.from_pretrained(MODEL_NAME)
model_config.num_labels = args.num_labels
# model_config.vocab_size+=added_token_num
# model_config.update({"head_type": args.head_type})
model_config.update({"head_type": 'more_dense'})
model = customRobertaForSequenceClassification.from_pretrained(MODEL_NAME,
                                                               config  = model_config )
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
print(model.config)
model.parameters
model.to(device)

training_args = TrainingArguments(
    output_dir=args.output_dir,  # output directory
    save_total_limit=args.save_total_limit,  # number of total save model.
    save_steps=args.save_steps,  # model saving step.
    num_train_epochs=args.epochs,  # total number of training epochs
    learning_rate=args.lr,  # learning_rate
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=args.eval_bs,  # batch size for evaluation
    warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,  # strength of weight decay
    logging_dir=args.logging_dir,  # directory for storing logs
    logging_steps=args.logging_steps,  # log saving step.
    evaluation_strategy=args.eval_strategy,  # evaluation strategy to adopt during training
    # `no`: No evaluation during training.
    # `steps`: Evaluate every `eval_steps`.
    # `epoch`: Evaluate every end of epoch.
    eval_steps=args.eval_steps,  # evaluation step.
    load_best_model_at_end=args.load_best_model_at_end,
    # https://docs.wandb.ai/guides/integrations/huggingface
    # Hugging face Trainer 내부 integration 된 wandb로 logging
    report_to='none',
    fp16=True,
    adafactor=True
    # run_name=exp_full_name,
)

cls_list = get_cls_list(train_dataset)
trainer = customTrainer3(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=RE_train_dataset,  # training dataset
    eval_dataset=RE_eval_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # define metrics function
    # callbacks=[customWandbCallback()],
    cls_list=cls_list,
    add_args=args
)

# train model
trainer.train()
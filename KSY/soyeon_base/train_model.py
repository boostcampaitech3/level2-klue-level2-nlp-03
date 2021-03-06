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

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix
from utils import plot_cm_by_num_samples, plot_cm_by_ratio
from custom.callback import customWandbCallback
    #customTrainerState,customTrainerControl,customTrainerCallback
from custom.trainer import customTrainer, customTrainer2,customTrainer3
from train import get_args,seed_fix

from models.custom_roberta_test import customRBERTForSequenceClassification
# from models.custom_roberta import customRobertaForSequenceClassification

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
    global label_list

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
    global label_list

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.
    conf = confusion_matrix(labels, preds)
    fig1 = plot_cm_by_num_samples(conf, label_list)
    fig2 = plot_cm_by_ratio(conf, label_list)

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
        'cm_samples': fig1,
        'cm_ratio': fig2
    }

args = get_args()
args.train_data_dir = "/opt/ml/dataset/train/train.csv"
seed_fix(args.seed)

MODEL_NAME = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# added_token_num, tokenizer = add_special_token(tokenizer, args.entity_marker_type)
# total_train_dataset = load_data(args.train_data_dir, args.augmentation, args.add_entity_marker, args.entity_marker_type, args.data_preprocessing)
if args.add_entity_marker:
    added_token_num, tokenizer = add_special_token(tokenizer, args.entity_marker_type)

total_train_dataset = pd.read_csv('final_train_entity_marker2.csv')
# 먼저 중복여부 판별을 위한 코드
total_train_dataset['is_duplicated'] = total_train_dataset['sentence'].duplicated(keep=False)

result = label_to_num(total_train_dataset['label'].values)
total_train_label = pd.DataFrame(data=result, columns=['label'])

# dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
# dev_label = label_to_num(dev_dataset['label'].values)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Start Training...')

for fold, (train_idx, val_idx) in enumerate(kfold.split(total_train_dataset, total_train_label)):
    if fold !=0:
        break
    train_dataset = total_train_dataset.iloc[train_idx]
    val_dataset = total_train_dataset.iloc[val_idx]
    train_label = total_train_label.iloc[train_idx]
    val_label = total_train_label.iloc[val_idx]

    train_dataset.reset_index(drop=True, inplace=True)
    val_dataset.reset_index(drop=True, inplace=True)
    train_label.reset_index(drop=True, inplace=True)
    val_label.reset_index(drop=True, inplace=True)

    temp = []

    for val_idx in val_dataset.index:
        if val_dataset['is_duplicated'].iloc[val_idx] == True:
            if val_dataset['sentence'].iloc[val_idx] in train_dataset['sentence'].values:
                train_dataset.append(val_dataset.iloc[val_idx])
                train_label.append(val_label.iloc[val_idx])
                temp.append(val_idx)

    val_dataset.drop(temp, inplace=True, axis=0)
    val_label.drop(temp, inplace=True, axis=0)

    train_label_list = train_label['label'].values.tolist()
    val_label_list = val_label['label'].values.tolist()

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(val_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label_list)
    RE_dev_dataset = RE_Dataset(tokenized_dev, val_label_list)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = args.num_labels
    model_config.update({"use_entity_embedding": True})
    model_config.update({"entity_emb_size": 2})
    # model_config.type_vocab_size = 2
    # model_config.vocab_size+=added_token_num
    model_config.update({"head_type": args.head_type})
    # model_config.update({"head_type": 'more_dense'})

    model = customRBERTForSequenceClassification.from_pretrained(MODEL_NAME,
                                                                   config  = model_config )
    if args.add_entity_marker:
        model.resize_token_embeddings(tokenizer.vocab_size + added_token_num)
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # wandb.init(project='KLUE',
    #            name='rbert_test',
    #            entity='kimcando')  # nlp-03

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
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
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        # callbacks=[customWandbCallback()],
        cls_list=cls_list,
        add_args=args
    )

    # train model
    trainer.train()

#### old version
# breakpoint()

# train_dataset, eval_dataset = load_split_data(args.train_data_dir,
#                                                           args.seed,
#                                                           args.eval_ratio,
#                                                           augmentation='NO_AUG+IDX')
# train_label = label_to_num(train_dataset['label'].values)
# eval_label = label_to_num(eval_dataset['label'].values)

# tokenized_train = tokenized_dataset(train_dataset, tokenizer)
# tokenized_eval = tokenized_dataset(eval_dataset, tokenizer)

# tokenized_train = tokenized_dataset_IDX(train_dataset, tokenizer)
# tokenized_eval = tokenized_dataset_IDX(eval_dataset, tokenizer)

# make dataset for pytorch.
# RE_train_dataset = RE_Dataset(tokenized_train, train_label)
# RE_eval_dataset = RE_Dataset(tokenized_eval, eval_label)

# RE_train_dataset = RE_Dataset_IDX(tokenized_train, train_label)
# breakpoint()
# RE_eval_dataset = RE_Dataset_IDX(tokenized_eval, eval_label)

# RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)
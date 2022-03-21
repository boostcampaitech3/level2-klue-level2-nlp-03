# base libraries
import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *

# get arguments
from arguments import get_args

# additionals
import wandb
import random

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
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def label_to_num(label):
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(args,reports, exp_full_name):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data(args.train_data_dir)
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset['label'].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = args.num_labels

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
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
        report_to=reports,
        run_name = exp_full_name,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained(args.model_save_dir)

def make_dirs(args):
    # model save
    os.makedirs(args.model_save_dir, exist_ok=True)
    # output
    os.makedirs(args.output_dir, exist_ok=True)


def main():
    args = get_args()
    seed_fix(args.seed)

    # make directories
    make_dirs(args)
    # https://docs.wandb.ai/guides/integrations/huggingface
    if args.use_wandb:
        # entity : 우리 그룹/팀 이름
        # project : 우리 그룹의 프로젝트 이름
        # name : 저장되는 실행 이름 (#TODO: 아직 convention 정하지 않음)
        reports = 'wandb'
        exp_full_name = f'{args.user_name}_{args.model_name}_{args.lr}_{args.optimizer}_{args.loss_fn}'
        # wandb.init(project='klue-re',
        #            name=exp_total_name)  # nlp-03
        # wandb.init(project='klue-re',
        #            name=exp_total_name,
        #            entity='kimcando')  # nlp-03
        # wandb.config.update(args)
        print('#######################')
        print(f'Experiments name: {exp_full_name}')
        print('#######################')
    else:
        reports = 'none'
        exp_full_name = ''
        print('@@@@@@@@Notice@@@@@@@@@@')
        print('YOU ARE NOT LOGGING RESULTS NOW')
        print('@@@@@@@@$$$$$$@@@@@@@@@@')

    train(args, reports, exp_full_name)
    # only when using notebook
    # wandb.finish()

if __name__ == '__main__':
    main()

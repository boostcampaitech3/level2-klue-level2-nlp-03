#  base
import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np

#  library
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
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
)
from pytorch_lightning.lite import LightningLite
import wandb
import random

#  Custom - python file
from arguments import get_args  # get arguments
from load_data import *


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


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
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


class Lite(LightningLite):
    def run(self, args, exp_full_name, reports="wandb"):
        # load model and tokenizer
        MODEL_NAME = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        total_train_dataset = load_data(args.train_data_dir)
        # 먼저 중복여부 판별을 위한 코드
        total_train_dataset["is_duplicated"] = total_train_dataset["sentence"].duplicated(keep=False)
        # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
        result = label_to_num(total_train_dataset["label"].values)
        total_train_label = pd.DataFrame(data=result, columns=["label"])
        # dev_label = label_to_num(dev_dataset['label'].values)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(total_train_dataset, total_train_label)):
            print("# of fold : ", fold)

            train_dataset = total_train_dataset.loc[train_idx]
            val_dataset = total_train_dataset.loc[val_idx]
            train_label = total_train_label.loc[train_idx]
            val_label = total_train_label.loc[val_idx]

            train_dataset.reset_index(drop=True, inplace=True)
            val_dataset.reset_index(drop=True, inplace=True)
            train_label.reset_index(drop=True, inplace=True)
            val_label.reset_index(drop=True, inplace=True)

            temp = []

            for val_idx in val_dataset.index:
                if val_dataset["is_duplicated"].iloc[val_idx] == True:
                    if val_dataset["sentence"].iloc[val_idx] in train_dataset["sentence"].values:
                        train_dataset.append(val_dataset.iloc[val_idx])
                        train_label.append(val_label.iloc[val_idx])
                        temp.append(val_idx)
            val_dataset.drop(temp, inplace=True, axis=0)
            val_label.drop(temp, inplace=True, axis=0)

            train_label_list = train_label["label"].values.tolist()
            val_label_list = val_label["label"].values.tolist()

            # tokenizing dataset
            tokenized_train = tokenized_dataset(train_dataset, tokenizer)
            tokenized_dev = tokenized_dataset(val_dataset, tokenizer)

            # make dataset for pytorch.
            RE_train_dataset = RE_Dataset(tokenized_train, train_label_list)
            RE_dev_dataset = RE_Dataset(tokenized_dev, val_label_list)

            # setting model hyperparameter
            model_config = AutoConfig.from_pretrained(MODEL_NAME)
            model_config.num_labels = args.num_labels

            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
            print(model.config)

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
                save_strategy=args.eval_strategy,  # evaluation을 매 epoch마다 하려면 save_strategy option도 바꿔줘야 한다
                eval_steps=args.eval_steps,  # evaluation step.
                load_best_model_at_end=args.load_best_model_at_end,
                # Hugging face Trainer 내부 integration 된 wandb로 logging
                group_by_length=args.group_by_length,
                report_to=reports,
                run_name=exp_full_name,
            )
            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                args=training_args,  # training arguments, defined above
                train_dataset=RE_train_dataset,  # training dataset
                eval_dataset=RE_dev_dataset,  # evaluation dataset
                compute_metrics=compute_metrics,  # define metrics function
            )

            # train model
            trainer.train()
            # model.save_pretrained(args.model_save_dir)
            if not os.path.exists(f"{args.model_save_dir}_{fold}"):
                os.makedirs(f"{args.model_save_dir}_{fold}")
            torch.save(model.state_dict(), os.path.join(f"{args.model_save_dir}_{fold}", "pytorch_model.bin"))
            print(f"fold{fold} fin!")


def custom_kfold_train(args, exp_full_name):
    Lite(devices=1, accelerator=args.accelerator, precision=args.precision).run(args, exp_full_name)

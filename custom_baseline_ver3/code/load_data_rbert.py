import pickle as pickle
import os
import pandas as pd
import torch

# additional
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

# added by eunki;
# typo here..
from preprocss import *
# added by sujeong;
from entity_marker import *


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

class RE_Dataset_IDX(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {}
    for key, val in self.pair_dataset.items():
        if key =='subj_idxs':
            item.update({'subj_start':val[idx][0]})
            item.update({'subj_end': val[idx][1]})
        elif key =='obj_idxs':
            item.update({'obj_start':val[idx][0]})
            item.update({'obj_end': val[idx][1]})
        else:
            item.update({key: val[idx].clone().detach()})
    # item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

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
        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['en_trans_sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'], })
        return out_dataset

    elif augmentation == "AUG":
        subject_entity = []
        object_entity = []
        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset_source = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'], })

        subject_entity = []
        object_entity = []

        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)

        out_dataset_aug = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['en_trans_sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'], })

        out_dataset = pd.concat([out_dataset_source, out_dataset_aug])

        return out_dataset

    else:  # default = "NO_AUG"
        subject_entity = []
        object_entity = []
        for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
            i = i[1:-1].split(',')[0].split(':')[1]
            j = j[1:-1].split(',')[0].split(':')[1]

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'], })
        return out_dataset


def load_data(dataset_dir, augmentation, add_entity_marker, entity_marker_type, data_preprocessing):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)

    # 중복 데이터 제거를 위한 전처리
    pd_dataset = preprocess(pd_dataset)

    # entity marker 추가
    # added by sujeong;
    print("add_entity_marker : ", add_entity_marker)
    print("entity_marker_type: ", entity_marker_type)
    if add_entity_marker:
        pd_dataset = get_entity_marked_data(df=pd_dataset, marker_type=entity_marker_type)
    breakpoint()

    # 데이터 전처리 (중복 괄호 제거, 이상한 문장 부호 수정, 연속된 공백 수정)
    # added by sujeong;
    print("data_preprocessing : ", data_preprocessing)
    if data_preprocessing:
        run_preprocess(pd_dataset)

    # 데이터셋으로 제작
    dataset = preprocessing_dataset(pd_dataset, augmentation)
    breakpoint()
    return dataset


def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    # print(concat_entity)
    tokenized_sentences = tokenizer(
        # concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )

    tokenized_sentences = add_entity_embeddings(tokenized_sentences)
    return tokenized_sentences


def label_to_num(label):
    num_label = []
    with open('./dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def get_cls_list(pd_dataset):
    labels = label_to_num(pd_dataset['label'])
    _, distr = np.unique(labels, return_counts=True)
    return distr


def add_entity_embeddings(
        tokenized_sentences,
        # max_seq_len,
        # tokenizer,
        # cls_token="[CLS]",
        # cls_token_segment_id=0,
        # sep_token="[SEP]",
        # pad_token=0,
        # pad_token_segment_id=0,
        # sequence_a_segment_id=0,
        # add_sep_token=False,
        # mask_padding_with_zero=True,
):
    # features = []
    e1_mask_total = []
    e2_mask_total = []
    length = tokenized_sentences['attention_mask'].size()[0]
    dim = tokenized_sentences['attention_mask'].size()[1]

    for index in range(length):
        # for (ex_index, example) in enumerate(examples):
        if index % 5000 == 0:
            print("Writing example %d of %d" % (index, length))

        tokens_a = tokenized_sentences.tokens(index)
        # tokens_a = tokenizer.tokenize(example.text_a)
        # https://velog.io/@spasis/%EB%B9%A0%EB%A5%B8-%ED%86%A0%ED%81%AC%EB%82%98%EC%9D%B4%EC%A0%80Fast-tokenizer%EC%9D%98-%ED%8A%B9%EB%B3%84%ED%95%9C-%EB%8A%A5%EB%A0%A5

        # 일단은 entity marker 2번으로 진행한다고 생각하고 2번만 구현
        subj_t = []
        obj_t = []
        for i, t in enumerate(tokens_a):
            if t == '@':
                subj_t.append(i)
            elif t == '#':
                obj_t.append(i)

        e11_p = subj_t[0]  # the start position of entity1
        e12_p = subj_t[1]  # the end position of entity1
        e21_p = obj_t[0]  # the start position of entity2
        e22_p = obj_t[1]  # the end position of entity2

        # Replace the token
        # tokens_a[e11_p] = "$"
        # tokens_a[e12_p] = "$"
        # tokens_a[e21_p] = "#"
        # tokens_a[e22_p] = "#"

        # Add 1 because of the [CLS] token
        # e11_p += 1
        # e12_p += 1
        # e21_p += 1
        # e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        # if add_sep_token:
        #    special_tokens_count = 2
        # else:
        #    special_tokens_count = 1
        # if len(tokens_a) > max_seq_len - special_tokens_count:
        #    tokens_a = tokens_a[: (max_seq_len - special_tokens_count)]

        # tokens = tokens_a
        # if add_sep_token:
        #    tokens += [sep_token]

        # token_type_ids = [sequence_a_segment_id] * len(tokens)

        # tokens = [cls_token] + tokens
        # token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenized_sentences['input_ids'][index]
        attention_mask = tokenized_sentences['attention_mask'][index]
        token_type_ids = tokenized_sentences['token_type_ids'][index]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        # padding_length = max_seq_len - dim
        # input_ids = input_ids + ([pad_token] * padding_length)
        # attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        # token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * dim
        e2_mask = [0] * dim

        for i in range(e11_p + 1, e12_p):
            e1_mask[i] = 1
        for j in range(e21_p + 1, e22_p):
            e2_mask[j] = 1

        # assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        # assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
        #    len(attention_mask), max_seq_len
        # )
        # assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(
        #    len(token_type_ids), max_seq_len
        # )

        # label_id = int(example.label)

        if index < 5:
            print("*** Example ***")
            # print("guid: %s" % example.guid)
            print("tokens: %s" % " ".join([str(x) for x in tokens_a]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            # print("label: %s (id = %d)" % (example.label, label_id))
            print("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            print("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        e1_mask_total.append(torch.tensor(e1_mask))
        e2_mask_total.append(torch.tensor(e2_mask))
        # features.append(
        #    InputFeatures(
        #        input_ids=input_ids,
        #        attention_mask=attention_mask,
        #        token_type_ids=token_type_ids,
        #        #label_id=label_id,
        #        e1_mask=e1_mask,
        #        e2_mask=e2_mask,
        #    )
        # )
    e1_mask_total = torch.stack(e1_mask_total, dim=0)
    e2_mask_total = torch.stack(e2_mask_total, dim=0)
    tokenized_sentences['e1_mask'] = e1_mask_total
    tokenized_sentences['e2_mask'] = e2_mask_total

    return tokenized_sentences


if __name__ == '__main__':
    from transformers import AutoTokenizer
    data_dir = "/opt/ml/dataset/train/train.csv"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # train_dataset, val_dataset = load_split_data(data_dir)
    train_dataset, val_dataset = load_split_data(data_dir,augmentation='NO_AUG+IDX')

    labels = label_to_num(train_dataset['label'])
    # eunki
    # train_dataset, val_dataset, train_label_list, val_label_list = load_split_data(data_dir)

    # tokenizing dataset
    tokenized_train= tokenized_dataset_IDX(train_dataset, tokenizer)
    tokenized_eval = tokenized_dataset(val_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    # RE_train_dataset = RE_Dataset(tokenized_train, train_label_list)
    RE_train_dataset = RE_Dataset_IDX(tokenized_train, labels)
    breakpoint()
    RE_eval_dataset = RE_Dataset(tokenized_eval, val_label_list)

import pickle as pickle
import os
import pandas as pd
import torch

# additional
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np

# added by eunki;
from preprocess import *
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
            if key == 'subj_idxs':
                item.update({'subj_start': val[idx][0]})
                item.update({'subj_end': val[idx][1]})
            elif key == 'obj_idxs':
                item.update({'obj_start': val[idx][0]})
                item.update({'obj_end': val[idx][1]})
            else:
                item.update({key: val[idx].clone().detach()})
        # item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# eunki + sujeong;
def preprocessing_dataset(dataset, augmentation='NO_AUG'):
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
    elif augmentation == 'NO_AUG+IDX':
        subject_entity = []
        object_entity = []
        subj_idxs = []
        obj_idxs = []
        for subj, obj in zip(dataset['subject_entity'], dataset['object_entity']):
            breakpoint()
            i = subj[1:-1].split(',')[0].split(':')[1]
            j = obj[1:-1].split(',')[0].split(':')[1]

            subj_idxs.append([eval(subj)['start_idx'], eval(subj)['end_idx']])
            obj_idxs.append([eval(obj)['start_idx'], eval(obj)['end_idx']])
            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
             'object_entity': object_entity, 'label': dataset['label'],
             'subj_idxs': subj_idxs, 'obj_idxs': obj_idxs})
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


def train_eval_dup_split(pd_dataset, seed, val_ratio):
    """
    sentence 중복이 있기 때문에 전체 dataframe에서 train_test_split 을 하면 data leakage 발생 가능합니다.
    sentence 별로 사용된 카운트는 {1: 25380, 2: 3179, 3: 244} 입니다. 따라서,
    1) 1번만 쓰인 dataframe 기준으로 train_ratio: val_ratio 로 나눕니다. 이 경우 class에 따라 stratify 됩니다.
    2) 2번,3번 쓰인 경우 sentence로 묶고(group), sentence 에 대해 train_ratio:val_ratio 로 나눕니다.
    3) 이후, 각 train_dataframe, eval_dataframe으로 합칩니다.

    # TODO: uniq sentence에 대해서는 cls stratified 적용되지만, duplicated는 고려되지 않음ㅠ
    """

    # 1번만 쓰인 애들만 가져옴
    uniq_dataset = pd_dataset.drop_duplicates(subset='sentence', keep=False)
    uniq_label = uniq_dataset['label']
    uniq_train, uniq_eval = train_test_split(uniq_dataset, test_size=val_ratio, shuffle=True,
                                             stratify=uniq_label,
                                             random_state=seed)
    # 각 문장별로 사용된 횟수 기록
    # cnt : {sentence: 해당 dataframe id}
    cnt = defaultdict(list)
    dup_dataset = pd_dataset[pd_dataset.duplicated(['sentence'], keep=False)]
    for itr, vals in dup_dataset.iterrows():
        sents = vals['sentence']
        cnt[sents].append(itr)

    double = defaultdict(list)
    triple = defaultdict(list)
    for s, idxs in cnt.items():
        if len(idxs) == 2:
            double[s] = idxs
        elif len(idxs) == 3:
            triple[s] = idxs
        else:
            raise NotImplementedError

    # 2번, 3번 쓰인 경우들로 각각 분리해서 ratio 에 맞게 split
    double_list = list(double.keys())
    train_split = int((1 - val_ratio) * len(double_list))
    train_double_list = double_list[:train_split]
    eval_double_list = double_list[train_split:]

    triple_list = list(triple.keys())
    train_split = int((1 - val_ratio) * len(triple_list))
    train_triple_list = triple_list[:train_split]
    eval_triple_list = triple_list[train_split:]

    # 현재 저장된 값들은 sentence
    train_list = train_double_list + train_triple_list
    eval_list = eval_double_list + eval_triple_list

    # sentence를 key로 갖는 cnt 딕셔너리를 통해 원래 dataframe의 id 갖는 리스트 생성
    final_train_list = []
    final_eval_list = []
    for s in train_list:
        final_train_list.extend(cnt[s])

    for s in eval_list:
        final_eval_list.extend(cnt[s])

    # duplicated 에 대한 dataframe 완성
    dup_train = dup_dataset.loc[final_train_list]
    dup_eval = dup_dataset.loc[final_eval_list]

    # 위에 uniq_ 데이터 프레임들과 합쳐줌
    train_df = pd.concat([uniq_train, dup_train])
    eval_df = pd.concat([uniq_eval, dup_eval])
    return train_df, eval_df


def load_split_dup_data(dataset_dir, seed=42, eval_ratio=0.2):
    """ csv 파일을 경로에 맡게 불러오고,
    duplicated cls를 고려해 train, eval에 맞게 분리해줍니다
    """
    print('### Split considering dups')
    pd_dataset = pd.read_csv(dataset_dir)
    pd_train, pd_eval = train_eval_dup_split(pd_dataset, seed, eval_ratio)

    train_dataset = preprocessing_dataset(pd_train)
    eval_dataset = preprocessing_dataset(pd_eval)

    return train_dataset, eval_dataset


def label_to_num(label):
    num_label = []
    with open('/opt/ml/level2-klue-level2-nlp-03/KSY/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def load_split_eunki_data(dataset_dir, sedd=42, eval_ratio=0.2):
    """
    은기님 구현 로직대로 우선 StratifiedKFold 중 1개만 선택
    validation 에 있는 문장이 train에 나온다면 맞바꿔서 개별 중복 문장은 training 또는 validation에만 포함되게 함
    """
    from sklearn.model_selection import StratifiedKFold

    pd_dataset = pd.read_csv(dataset_dir)
    total_train_dataset = preprocessing_dataset(pd_dataset)

    total_train_dataset['is_duplicated'] = total_train_dataset['sentence'].duplicated(keep=False)
    result = label_to_num(total_train_dataset['label'].values)

    total_train_label = pd.DataFrame(data=result, columns=['label'])
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = list(kfold.split(total_train_dataset, total_train_label))[0]

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
        if val_dataset['is_duplicated'].iloc[val_idx] == True:
            if val_dataset['sentence'].iloc[val_idx] in train_dataset['sentence'].values:
                train_dataset.append(val_dataset.iloc[val_idx])
                train_label.append(val_label.iloc[val_idx])
                temp.append(val_idx)

    val_dataset.drop(temp, inplace=True, axis=0)
    val_label.drop(temp, inplace=True, axis=0)

    train_label_list = train_label['label'].values.tolist()
    val_label_list = val_label['label'].values.tolist()
    return train_dataset, val_dataset, train_label_list, val_label_list


def load_split_data(dataset_dir, seed=42, eval_ratio=0.2, augmentation='NO_AUG'):
    """ csv 파일을 경로에 맡게 불러오고,
    duplicated cls를 고려하지 않고  train, eval에 맞게 분리해줍니다
    """
    print('### Split basic')
    pd_dataset = pd.read_csv(dataset_dir)
    label = pd_dataset['label']
    pd_train, pd_eval = train_test_split(pd_dataset, test_size=eval_ratio, shuffle=True,
                                         stratify=label,
                                         random_state=seed)

    train_dataset = preprocessing_dataset(pd_train, augmentation=augmentation)
    eval_dataset = preprocessing_dataset(pd_eval, augmentation=augmentation)

    return train_dataset, eval_dataset


def preprocessing_ent_dataset(dataset):
    """
    TEST 중. special token add 후 masking이 안됨
    처음 불러온 csv 파일에서 sentecen에 등장하고 start_idx, end_idx 를 갖는 subject, object를 marking."""

    subject_entity = []
    object_entity = []
    subject_type = []
    object_type = []
    # subject_entity : {'word':- , 'start_idx':-, 'end_idx':-}
    for idx, (subj, obj, sents) in enumerate(
            zip(dataset['subject_entity'], dataset['object_entity'], dataset['sentence'])):
        subj, obj = eval(subj), eval(obj)
        new_subj = '[E1]' + sents[subj['start_idx']: subj['end_idx'] + 1] + '[/E1]'
        new_obj = '[E2]' + sents[obj['start_idx']: obj['end_idx'] + 1] + '[/E2]'
        if subj['start_idx'] >= obj['start_idx']:
            temp = sents[:obj['start_idx']] + new_obj + sents[obj['end_idx'] + 1: subj['start_idx']] + new_subj + sents[
                                                                                                                  subj[
                                                                                                                      'end_idx'] + 1:]
        elif subj['start_idx'] < obj['start_idx']:
            temp = sents[:subj['start_idx']] + new_subj + sents[
                                                          subj['end_idx'] + 1: obj['start_idx']] + new_obj + sents[obj[
                                                                                                                       'end_idx'] + 1:]

        dataset['sentence'][idx] = temp

        subject_entity.append(subj['word'])
        object_entity.append(obj['word'])

        subject_type.append(subj['type'])
        object_type.append(obj['type'])

    # print(len(dataset['id'], len(dataset['sentence']))
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': dataset['sentence'],
                                'subject_entity': subject_entity, 'object_entity': object_entity,
                                'subject_type': subject_type, 'object_type': object_type,
                                'label': dataset['label'], })
    return out_dataset


def load_split_ent_data(dataset_dir, seed=42, eval_ratio=0.2):
    """ TEST 중
    """
    print('### Split basic')
    pd_dataset = pd.read_csv(dataset_dir)
    label = pd_dataset['label']
    pd_train, pd_eval = train_test_split(pd_dataset, test_size=eval_ratio, shuffle=True,
                                         stratify=label,
                                         random_state=seed)

    train_dataset = preprocessing_ent_dataset(pd_train)
    eval_dataset = preprocessing_ent_dataset(pd_eval)

    return train_dataset, eval_dataset


def load_data_base(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_data_base_test(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset, augmentation='NO_AUG+IDX')

    return dataset


def load_data(dataset_dir, augmentation, add_entity_marker, entity_marker_type, data_preprocessing):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)

    # 중복 데이터 제거를 위한 전처리
    pd_dataset = preprocess(pd_dataset)

    # entity marker 추가
    # added by sujeong;
    print("add_entity_marker : ", add_entity_marker)
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


def tokenized_dataset_IDX(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    subj_idxs = []
    obj_idxs = []
    # 이렇게 안넣으면 indexing 망가짐
    for e01, e02, subj_idx, obj_idx in zip(dataset['subject_entity'], dataset['object_entity'], dataset['subj_idxs'],
                                           dataset['obj_idxs']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
        subj_idxs.append(subj_idx)
        obj_idxs.append(obj_idx)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=True,
    )
    breakpoint()
    tokenized_sentences.update({'subj_idxs': subj_idxs})
    tokenized_sentences.update({'obj_idxs': obj_idxs})

    return tokenized_sentences


def get_cls_list(pd_dataset):
    labels = label_to_num(pd_dataset['label'])
    _, distr = np.unique(labels, return_counts=True)
    return distr


if __name__ == '__main__':
    from transformers import AutoTokenizer

    data_dir = "/opt/ml/dataset/train/train.csv"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # train_dataset, val_dataset = load_split_data(data_dir)
    train_dataset, val_dataset = load_split_data(data_dir, augmentation='NO_AUG+IDX')

    labels = label_to_num(train_dataset['label'])
    # eunki
    # train_dataset, val_dataset, train_label_list, val_label_list = load_split_data(data_dir)

    # tokenizing dataset
    tokenized_train = tokenized_dataset_IDX(train_dataset, tokenizer)
    tokenized_eval = tokenized_dataset(val_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    # RE_train_dataset = RE_Dataset(tokenized_train, train_label_list)
    RE_train_dataset = RE_Dataset_IDX(tokenized_train, labels)
    breakpoint()
    RE_eval_dataset = RE_Dataset(tokenized_eval, val_label_list)

import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np

def preprocess(df):
    result = remove_duplicate(df)
    return result

def remove_duplicate(df):
    df['is_duplicated'] = False
    df['is_duplicated'][df[['sentence', 'subject_entity', 'object_entity']].duplicated(keep=False) == True] = True

    temp = []

    # 먼저, 중복된 데이터를 제거할 때, 'sentence', 'subject_entity', 'object_entity'는 같지만 라벨이 다르고 그리고 그때 label이 no_relation인 것들부터 삭제

    for i in range(len(df)):
        if df['is_duplicated'].iloc[i] == True and df['label'].iloc[i] == "no_relation":
            temp.append(i)
    
    df.drop(temp, inplace=True)

    # 그후에 'sentence', 'subject_entity', 'object_entity'이 동일한 경우가 나온다면 맨 첫번째만 유지

    df.drop_duplicates(subset=['sentence', 'subject_entity', 'object_entity'], keep='first')

    df.drop(['is_duplicated'], axis=1, inplace=True)


    return df
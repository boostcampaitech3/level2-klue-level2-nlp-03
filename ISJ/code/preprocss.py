import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
import re

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


# -----------------------------------------데이터 전처리(중복 괄호 제거, 이상한 문자 변경, 띄어쓰기 두 번 수정, 문장 길이 조절 등)-------------------------#
# all added by sujeong;

# 중복 괄호 제거
def remove_useless_breacket_revised(text):
    
    '''--> 비어 있는 괄호랑, 반복되는 괄호만 제거하는 것으로 변경'''
    
    bracket_pattern = re.compile(r"\([^\(\)]*?\)")
    #print(text)
    modi_text = ""
    text = text.replace("()", "")  # 수학() -> 수학
    brackets = bracket_pattern.finditer(text)
    #print(brackets.next().string)
    if not brackets: # 괄호가 없으면
        return text
    
    # key: 원본 문장에서 고쳐야하는 index, value: 고쳐져야 하는 값
    # e.g. {'2,8': '(數學)','34,37': ''}
    s_idx, e_idx = 0,0
    last = None
    for b in brackets:
        #print(b[0])
        if last and e_idx==b.start():
            if last == b[0]:
                #print('yes')
                text = text[:b.start()] + text[b.end():]
        last = b[0]
        s_idx, e_idx = b.start(), b.end()
    
    return text

# 이상한 문장부호 수정
def clean_punc(text):
    punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', } # "_": "-"만 제거

    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])
    text = text.strip()   

    return text


# 전체 전처리 process
def run_preprocess(df : pd.DataFrame):
    print("Run Preprocess...")
    df_preprocessed = pd.DataFrame(columns = ['id' , 'sentence', 'subject_entity', 'object_entity', 'label', 'source'])
    
    for i, data in enumerate(df.iloc):
            data_dict = dict(data)
            i_d, sentence, subj, obj, label, source = data_dict.values()
            
            # entity masking - sentence에서 entity 찾아서 subj, obj로 교체
            #print(eval(subj)['word'])
            if len(re.findall(r'(S_U_B_J)+|(O_B_J)+', sentence)) <= 0:
                sentence = sentence.replace(eval(subj)['word'], 'S_U_B_J')
                sentence = sentence.replace(eval(obj)['word'], 'O_B_J')

            # ----- 전처리 시작 ---- #
            # 불필요한 괄호제거
            sentence = remove_useless_breacket_revised(sentence)
            
            # 문장 부호 수정
            sentence = clean_punc(sentence)

            # 연속 띄어쓰기 수정
            sentence = re.sub(r"\s+", " ", sentence).strip()

            #불용어 제거
            #stop_word_list = ['']
            #sentence = remove_stopwords(sentence, stop_word_list)
            
            #최소최대길이 제한
            #if min_len > len(sentence) or len(sentence) > max_len:
            #    continue

            # ---- 전처리 종료

            if sentence:
                # entity masking 제거
                sentence = sentence.replace('S_U_B_J', eval(subj)['word'])
                sentence = sentence.replace('O_B_J', eval(obj)['word'])
                df_preprocessed.loc[i] = [i_d, sentence, subj, obj, label, source]

    return  df_preprocessed


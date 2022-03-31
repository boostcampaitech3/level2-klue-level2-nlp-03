import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import os

def main(args):
    def num_to_label(n):
        with open('dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
        origin_label = dict_num_to_label[n]
        return origin_label

    def to_nparray(s) :
        return np.array(list(map(float, s[1:-1].split(','))))

    dir = '/opt/ml/git/level2-klue-level2-nlp-03/eunki/code'

    path1 = f'./prediction/{args.model_name}/submission_0.csv'
    path2 = f'./prediction/{args.model_name}/submission_1.csv'
    path3 = f'./prediction/{args.model_name}/submission_2.csv'
    path4 = f'./prediction/{args.model_name}/submission_3.csv'
    path5 = f'./prediction/{args.model_name}/submission_4.csv'


    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)                                                        
    df4 = pd.read_csv(path4)
    df5 = pd.read_csv(path5)

    df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.2) + df2['probs'].apply(lambda x : to_nparray(x)*0.2) + df3['probs'].apply(lambda x : to_nparray(x)*0.2) + df4['probs'].apply(lambda x : to_nparray(x)*0.2) + df5['probs'].apply(lambda x : to_nparray(x)*0.2) 
    df1['softmax'] = df1['probs'].apply(lambda x : F.softmax(torch.tensor(x), dim=-1).detach().cpu().numpy())
    # 0 제외 나머지 클래스에 0.1씩 더하기?
    df1['pred_label'] = df1['softmax'].apply(lambda x : num_to_label(np.argmax(x)))
    df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))
    if not os.path.exists(f'./prediction/final'):
        os.makedirs(f'./prediction/final')
    df1.to_csv(f'./prediction/final/submission_final_roberta_large.csv', index=False)

if __name__ == '__main__':    
  # model dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-large")
    args = parser.parse_args()
    print(args)
    main(args)
  
import numpy as np
import pandas as pd
import pickle
import torch

import torch.nn.functional as F
import argparse
import os
from tqdm import tqdm

def main(args):

    def num_to_label(n):
        with open('dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
        origin_label = dict_num_to_label[n]
        return origin_label

    def to_nparray(s) :
        return np.array(list(map(float, s[1:-1].split(','))))

    dir = '/opt/ml/git/level2-klue-level2-nlp-03/eunki/code'
    path1 = os.path.join(args.save_dir,'submission_final_roberta_large.csv') # 가져올 csv 파일 주소 입력 필수!
    path2 = os.path.join(args.save_dir,'submission_final_koelectra.csv')


    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.2) + df2['probs'].apply(lambda x : to_nparray(x)*0.2)
    
    df1['softmax'] = df1['probs'].apply(lambda x : F.softmax(torch.tensor(x), dim=-1).detach().cpu().numpy())
    # 0 제외 나머지 클래스에 0.1씩 더하기?
    df1['pred_label'] = df1['softmax'].apply(lambda x : num_to_label(np.argmax(x)))
    df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))

    df1.to_csv(f'./prediction/final/submission_final_model_ensemble.csv', index=False)

if __name__ == '__main__':    
  # model dir
  for i in range (5):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=int, default=i)
    parser.add_argument("--save_dir", type=int, default="../prediction/final")
    args = parser.parse_args()
    print(args)
    main(args)
  
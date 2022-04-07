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
        with open('./dict_num_to_label.pkl', 'rb') as f:
            dict_num_to_label = pickle.load(f)
        origin_label = dict_num_to_label[n]
        return origin_label

    def to_nparray(s) :
        return np.array(list(map(float, s[1:-1].split(','))))

    dir = '/opt/ml/git/level2-klue-level2-nlp-03/eunki/code'

    path1 = f'./prediction/{args.model_name}/submission_0_end.csv'
    path2 = f'./prediction/{args.model_name}/submission_1_end.csv'
    path3 = f'./prediction/{args.model_name}/submission_2_end.csv'
    path4 = f'./prediction/{args.model_name}/submission_3_end.csv'
    path5 = f'./prediction/{args.model_name}/submission_4_end.csv'

    path6 = f'./prediction/{args.model_name}/submission_5_end.csv'
    path7 = f'./prediction/{args.model_name}/submission_6_end.csv'
    path8 = f'./prediction/{args.model_name}/submission_7_end.csv'
    path9 = f'./prediction/{args.model_name}/submission_8_end.csv'
    path10 = f'./prediction/{args.model_name}/submission_9_end.csv'

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)                                                        
    df4 = pd.read_csv(path4)
    df5 = pd.read_csv(path5)

    df6 = pd.read_csv(path6)
    df7 = pd.read_csv(path7)
    df8 = pd.read_csv(path8)
    df9 = pd.read_csv(path9)
    df10 = pd.read_csv(path10)

    df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.1) + df2['probs'].apply(lambda x : to_nparray(x)*0.1) + \
                   df3['probs'].apply(lambda x : to_nparray(x)*0.1) + \
                   df4['probs'].apply(lambda x : to_nparray(x)*0.1) + \
                   df5['probs'].apply(lambda x : to_nparray(x)*0.1) + \
                   df6['probs'].apply(lambda x: to_nparray(x) * 0.1) + \
                   df7['probs'].apply(lambda x: to_nparray(x) * 0.1) + \
                   df8['probs'].apply(lambda x: to_nparray(x) * 0.1) + \
                   df9['probs'].apply(lambda x: to_nparray(x) * 0.1) + \
                   df10['probs'].apply(lambda x: to_nparray(x) * 0.1)
    df1['softmax'] = df1['probs'].apply(lambda x : F.softmax(torch.tensor(x), dim=-1).detach().cpu().numpy())
    # 0 제외 나머지 클래스에 0.1씩 더하기?
    df1['pred_label'] = df1['softmax'].apply(lambda x : num_to_label(np.argmax(x)))
    df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))
    if not os.path.exists(f'./prediction/final'):
        os.makedirs(f'./prediction/final')
    df1.to_csv(f'./prediction/final/submission_final_roberta_large_end.csv', index=False)

if __name__ == '__main__':    
  # model dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-large")
    args = parser.parse_args()
    print(args)
    main(args)
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

    path1 = f'./prediction/{args.model_name}/output0.csv'
    path2 = f'./prediction/{args.model_name}/output1.csv'
    path3 = f'./prediction/{args.model_name}/output2.csv'
    path4 = f'./prediction/{args.model_name}/output3.csv'
    path5 = f'./prediction/{args.model_name}/output4.csv'
    path6 = f'./prediction/{args.model_name}/output5.csv'
    path7 = f'./prediction/{args.model_name}/output6.csv'
    path8 = f'./prediction/{args.model_name}/output7.csv'
    path9 = f'./prediction/{args.model_name}/output8.csv'
    path10 = f'./prediction/{args.model_name}/output9.csv'
    path11 = f'./prediction/{args.model_name}/output10.csv'
    path12 = f'./prediction/{args.model_name}/output11.csv'
    path13 = f'./prediction/{args.model_name}/output12.csv'
    path14 = f'./prediction/{args.model_name}/output13.csv'
    path15 = f'./prediction/{args.model_name}/output14.csv'
    


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
    df11 = pd.read_csv(path11)
    df12 = pd.read_csv(path12)
    df13 = pd.read_csv(path13)
    df14 = pd.read_csv(path14)
    df15 = pd.read_csv(path15)

    

    df1['probs'] = df1['probs'].apply(lambda x : to_nparray(x)*0.066) + df2['probs'].apply(lambda x : to_nparray(x)*0.066) + df3['probs'].apply(lambda x : to_nparray(x)*0.066) + df4['probs'].apply(lambda x : to_nparray(x)*0.066) + df5['probs'].apply(lambda x : to_nparray(x)*0.066) + df6['probs'].apply(lambda x : to_nparray(x)*0.066) + df7['probs'].apply(lambda x : to_nparray(x)*0.066) + df8['probs'].apply(lambda x : to_nparray(x)*0.066) + df9['probs'].apply(lambda x : to_nparray(x)*0.066) + df10['probs'].apply(lambda x : to_nparray(x)*0.066) + df11['probs'].apply(lambda x : to_nparray(x)*0.066) + df12['probs'].apply(lambda x : to_nparray(x)*0.066) + df13['probs'].apply(lambda x : to_nparray(x)*0.066) + df14['probs'].apply(lambda x : to_nparray(x)*0.066) + df15['probs'].apply(lambda x : to_nparray(x)*0.066)
    
    df1['softmax'] = df1['probs'].apply(lambda x : F.softmax(torch.tensor(x), dim=-1).detach().cpu().numpy())
    # 0 제외 나머지 클래스에 0.1씩 더하기?
    df1['pred_label'] = df1['softmax'].apply(lambda x : num_to_label(np.argmax(x)))
    df1['probs'] = df1['probs'].apply(lambda x : str(list(x)))
    if not os.path.exists(f'./prediction/final'):
        os.makedirs(f'./prediction/final')
    df1.to_csv(f'./prediction/final/final_final.csv', index=False)

if __name__ == '__main__':    
  # model dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-large")
    args = parser.parse_args()
    print(args)
    main(args)
  
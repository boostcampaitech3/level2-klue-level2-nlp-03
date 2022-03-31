import argparse

def str2bool(v):
    # arguments에 유사한 값들끼리 다 boolean 처리 되도록

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # basic environment args & output , logging saving
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--device", type=str, default='cuda:0', help="device id")
    parser.add_argument("--output_dir", type=str, default='./results', help="output dir")

    # added by sykim; 디버깅할 때 켜두면 실험 너무 쌓여서 디버깅 때는 false하고 돌릴 때 True로 해뒀어요 저는.
    parser.add_argument("--use_wandb", type=str2bool, default=True, help="if you're ready to log in wandb")
    parser.add_argument("--user_name", type=str, default='Eunki', help="your initial name")
    parser.add_argument('--exp_name', type=str, required=False, default="kfold_prec",
                        help="wandb experiments name if needed")

    parser.add_argument("--logging_dir", type=str, default='./logs', help="log dir")
    parser.add_argument("--logging_steps", type=int, default=100, help="log dir")

    # data
    # added by sykim;
    parser.add_argument("--train_data_dir", type=str, default="/opt/ml/git/level2-klue-level2-nlp-03/eunki/dataset/train/train.csv", help="data directory") 
    parser.add_argument("--test_data_dir", type=str, default="/opt/ml/git/level2-klue-level2-nlp-03/eunki/dataset/test_data/test.csv", help="data directory")
    parser.add_argument("--eval_ratio", type=float, default=0.2, help="training/eval split ratio")

    # model
    # added by sykim;
    parser.add_argument("--model_name", type=str, default='klue/roberta-large', help="name of model")
    parser.add_argument("--num_labels", type=int, default=30, help="num of classes for model")
    parser.add_argument("--tokenizer_name", type=str, default='NA', help="name of tokenizer if necessary")

    parser.add_argument("--head_type", type=str, default='base', help="type for final classification head,",
                        choices = ['more_dense', 'base', 'lstm','modifiedBiLSTM'])

    # loss & optimizer
    # added by sykim; loss랑 OPtimizer는 라이브러리 안에 숨어있는 것 같아요. 혹시 몰라서 추가해뒀습니다.
    parser.add_argument("--loss_fn", type=str, default='base', help="name of loss")
    parser.add_argument("--optimizer", type=str, default='adamw_hf', help="name of optimizer")

    parser.add_argument("--gamma", type=float, default=1., help="name of loss")
    parser.add_argument("--smoothing", type=float, default=0.1, help="smoothing factor for label smoothing loss")

    # training basic hyperparms
    parser.add_argument("--epochs", type=int, default=3, help="total number of training epochs")
    parser.add_argument("--train_bs", type=int, default=32, help="batch size per device during training")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default:5e-5)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="number of warmup steps for learning rate scheduler")
    parser.add_argument("--lr_decay", type=float, default=1e-1, help="learning rate decaying when used")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="strength of weight decay")

    # evaluation
    parser.add_argument("--eval_bs", type=int, default=32, help="batch size per device for evaluation")
    parser.add_argument("--eval_strategy", type=str, default='steps',
                        help="evaluation strategy to adopt during training"
                             "`no`:  No evaluation during training."
                             "`steps`: Evaluate every `eval_steps`"
                             "`epoch`: Evaluate every end of epoch.")
    parser.add_argument("--eval_steps", type=int, default=500, help="number of evaluation steps")
    # --load_best_model_at_end requires the saving steps to be a round multiple of the evaluation steps, but found 500, which is not a round multiple of 3.
    parser.add_argument("--load_best_model_at_end", type=str2bool, default=True, help="load_best_model_at_end")
    parser.add_argument("--save_total_limit", type=int, default=3, help="number of total save model.")
    parser.add_argument("--save_steps", type=int, default=500, help="log saving step.")
    
    parser.add_argument("--model_save_dir", type=str, default='./best_model', help="model ckpt")
    parser.add_argument("--augmentation", type=str, default='NO_AUG', help="augmentation")

    args = parser.parse_args()
    return args
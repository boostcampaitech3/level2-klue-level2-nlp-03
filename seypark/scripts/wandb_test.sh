WANDB_PROJECT=klue-re \
python train_custom.py --model_name klue/bert-base \
                --use_wandb True \
                --user_name KSY \
                --exp_name test_wandb \
                --epochs 1 \
                --train_bs 64 \
                --eval_steps 3
python train.py --model_name klue/bert-base \
                --use_wandb True \
                --user_name KSY \
                --exp_name test_wandb \
                --eval_steps 3 \
                --load_best_model_at_end False \
                --epochs 1 \
                --train_bs 64 \
                --eval_bs 64
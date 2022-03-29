python train.py --model_name monologg/koelectra-base-v3-discriminator \
                --use_wandb True \
                --user_name seypark \
                --exp_name test_wandb \
                --eval_steps 3 \
                --load_best_model_at_end False \
                --epochs 5 \
                --train_bs 32 \
                --eval_bs 32 \
                --train_data_dir ../../dataset/train/train.csv \
                --model_name monologg/koelectra-base-v3-discriminator \
                --use_wandb False \
                --eval_strategy epoch \
                --train_method origin


nohup python train.py --model_name monologg/koelectra-base-v3-discriminator \
                --use_wandb True \
                --user_name Eunki \
                --exp_name model_ensemble \
                --eval_steps 300 \
                --save_steps 300 \
                --load_best_model_at_end True \
                --epochs 5 \
                --train_bs 32 \
                --eval_bs 32 \
                --train_data_dir ../dataset/train/en_all_train.csv &
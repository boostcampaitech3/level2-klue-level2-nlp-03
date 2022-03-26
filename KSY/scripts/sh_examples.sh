#python train.py --model_name monologg/koelectra-base-v3-discriminator \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-basic \
#                --eval_steps 100 \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 6 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ele_results_basic \
#                --model_save_dir ./ele_best_model_basic \
#                --eval_ratio 0.1

#python train.py --model_name klue/roberta-large \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-eunki \
#                --eval_steps 100 \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 6 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ro_results_dup \
#                --model_save_dir ./ro_best_model_dup

#python train.py --model_name klue/roberta-large \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-dup \
#                --eval_steps 100 \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 6 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ro_results_dup \
#                --model_save_dir ./ro_best_model_dup
#
python train.py --model_name klue/roberta-large \
                --use_wandb True \
                --user_name KSY \
                --split_mode split-basic \
                --eval_steps 100 \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --epochs 6 \
                --train_bs 32 \
                --eval_bs 250 \
                --output_dir ./ro_results_basic \
                --model_save_dir ./ro_best_model


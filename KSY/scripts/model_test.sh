#python train.py --model_name klue/roberta-large \
#                --head_type base \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-dup \
#                --eval_steps 100 \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 6 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ro_results_dup_mean \
#                --model_save_dir ./ro_best_model_dup_mean

# modifedBiLSTM
python train.py --model_name klue/roberta-large \
                --head_type modifiedBiLSTM \
                --use_wandb True \
                --user_name KSY \
                --split_mode split-basic \
                --eval_steps 100 \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --epochs 6 \
                --train_bs 32 \
                --eval_bs 250 \
                --output_dir ./ro_results_dup_lstm \
                --model_save_dir ./ro_best_model_dup_lstm

python train.py --model_name klue/roberta-large \
                --head_type lstm \
                --use_wandb True \
                --user_name KSY \
                --split_mode split-basic \
                --eval_steps 100 \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --epochs 6 \
                --train_bs 32 \
                --eval_bs 250 \
                --output_dir ./ro_results_dup_lstm \
                --model_save_dir ./ro_best_model_dup_lstm

#python train.py --model_name klue/roberta-large \
#                --head_type more_dense \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-basic \
#                --eval_steps 100 \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 6 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ro_results_dup_more \
#                --model_save_dir ./ro_best_model_dup_more
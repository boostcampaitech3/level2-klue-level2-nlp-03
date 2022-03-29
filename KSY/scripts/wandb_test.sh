#WANDB_PROJECT=klue-re \
#python train_custom.py --model_name klue/bert-base \
#                --use_wandb False \
#                --user_name KSY \
#                --exp_name test_wandb \
#                --epochs 1 \
#                --train_bs 64 \
#                --eval_steps 3
python train_model.py --model_name klue/roberta-large \
                --use_wandb True \
                --user_name KSY \
                --split_mode split-dup \
                --eval_steps 100 \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --epochs 6 \
                --train_bs 32 \
                --eval_bs 250 \
                --output_dir ./ro_results_dup_early \
                --model_save_dir ./ro_best_model_dup_early \
                --eval_ratio 0.05

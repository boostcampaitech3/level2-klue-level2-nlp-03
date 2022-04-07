python train_custom.py --model_name klue/roberta-large \
                --use_wandb True \
                --user_name KSY \
                --split_mode split-basic \
		            --loss_fn labelsmoothingloss \
                --eval_steps 100  \
                --logging_steps 10 \
                --load_best_model_at_end True \
                --epochs 6 \
                --train_bs 32 \
                --eval_bs 250 \
                --output_dir ./ro_results_basic_ls \
                --model_save_dir ./ro_best_model_dup_ls

#python train_custom.py --model_name klue/roberta-large \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-basic \
#		            --loss_fn focalloss \
#                --eval_steps 100  \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 4 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --output_dir ./ro_results_basic_focal \
#                --model_save_dir ./ro_best_model_dup_focal
#
#python train_custom.py --model_name klue/roberta-large \
#                --use_wandb True \
#                --user_name KSY \
#                --split_mode split-basic \
#		            --loss_fn focalloss \
#                --eval_steps 100  \
#                --logging_steps 10 \
#                --load_best_model_at_end True \
#                --epochs 4 \
#                --train_bs 32 \
#                --eval_bs 250 \
#                --gamma 2. \
#                --output_dir ./ro_results_basic_focal2 \
#                --model_save_dir ./ro_best_model_dup_focal2
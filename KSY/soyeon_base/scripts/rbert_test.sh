python train_model.py --data_preprocessing True \
                    --add_entity_marker True \
                    --entity_marker_type entity_marker_punc \
                    --head_type more_dense \
                    --lr 2e-5 \
                    --use_wandb True \
                    --user_name KSY \
                    --eval_steps 1 \
                    --save_steps 1 \
                    --eval_bs 256 \
                    --head_type more_dense \
                    --loss_fn labelsmoothingloss \
                    --smoothing 0.2 \
                    --epochs 8 \
                    --train_bs 64 \
                    --eval_bs 256

#python kfold_train_test.py --model_name klue/roberta-large \
#                --use_wandb True \
#                --user_name KSY \
#                --exp_name end_point_experiment \
#                --eval_steps 600 \
#                --save_steps 600 \
#                --load_best_model_at_end True \
#                --epochs 9 \
#                --train_bs 64 \
#                --eval_bs 64 \
#                --loss_fn labelsmoothingloss \
#                --smoothing 0.2 \
#                --head_type more_dense \
#                --save_total_limit 4 \
#                --lr 2e-5 \
#                --train_data_dir /opt/ml/dataset/train/preprocess.csv \
#                --data_preprocessing True \
#                --add_entity_marker True \
#                --entity_marker_type entity_marker_punc

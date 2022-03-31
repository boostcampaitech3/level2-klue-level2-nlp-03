python kfold_train_test.py --model_name klue/roberta-large \
                --use_wandb True \
                --user_name SujeongIm \
                --exp_name file_ensemble \
                --eval_steps 300 \
                --save_steps 300 \
                --load_best_model_at_end True \
                --epochs 1 \
                --train_bs 32 \
                --eval_bs 32 \
                --loss_fn labelsmoothingloss \
                --smoothing 0.2 \
                --head_type modifiedBiLSTM \
                --train_data_dir /opt/ml/dataset/train/train.csv \
                --data_preprocessing True \
                --add_entity_marker True \
                --entity_marker_type entity_marker
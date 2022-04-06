python kfold_train_test_rbert.py --data_preprocessing True \
                    --add_entity_marker True \
                    --entity_marker_type entity_marker_punc \
                    --head_type more_dense \
                    --lr 2e-5 \
                    --use_wandb True \
                    --user_name KSY \
                    --eval_steps 600 \
                    --save_steps 600 \
                    --save_total_limit 5 \
                    --train_bs 64 \
                    --eval_bs 64 \
                    --head_type more_dense \
                    --loss_fn labelsmoothingloss \
                    --smoothing 0.2 \
                    --epochs 8 \
                    --exp_name rbert_no_entity_embedding \
                    --use_entity_embedding False \
                    --output_dir results_no_embedding \
                    --train_data_dir /opt/ml/dataset/train/train.csv

python kfold_train_test_rbert.py --data_preprocessing True \
                    --add_entity_marker True \
                    --entity_marker_type entity_marker_punc \
                    --head_type more_dense \
                    --lr 2e-5 \
                    --use_wandb True \
                    --user_name KSY \
                    --eval_steps 600 \
                    --save_steps 600 \
                    --save_total_limit 5 \
                    --train_bs 64 \
                    --eval_bs 64 \
                    --head_type more_dense \
                    --loss_fn labelsmoothingloss \
                    --smoothing 0.2 \
                    --epochs 8 \
                    --exp_name rbert_with_entity_embedding \
                    --use_entity_embedding True \
                    --output_dir results_with_embedding \
                    --train_data_dir /opt/ml/dataset/train/train.csv
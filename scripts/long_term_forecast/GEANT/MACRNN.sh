export CUDA_VISIBLE_DEVICES=1

model_name=MACRNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/GEANT/\
  --data_path  GEANT_full_spine.csv \
  --model_id GEANT_12_1 \
  --model $model_name \
  --data GEANT \
  --features M \
  --n_heads 1 \
  --dropout 0.5 \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 529 \
  --matrix_dim 23 \
  --seq_len 12 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 529 \
  --c_out 529 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1000 \
  --learning_rate 0.0001 \
  --patience 1000 \
  --lradj off \
  # --loss 'RMSE' \
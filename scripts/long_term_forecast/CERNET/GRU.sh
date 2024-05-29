export CUDA_VISIBLE_DEVICES=1

model_name=GRU

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/CERNET/\
  --data_path  CERNET-fixdate.csv \
  --model_id CERNET_12_1 \
  --model $model_name \
  --data CERNET \
  --features M \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 288 \
  --seq_len 12 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 196 \
  --c_out 196 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1000 \
  --learning_rate 0.0001 \
  --patience 1000 \
  # --loss 'RMSE' \
export CUDA_VISIBLE_DEVICES=1

model_name=Ours

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/TM/CERNET/\
#   --data_path  CERNET-fixdate.csv \
#   --model_id CERNET_5_1_1l \
#   --model $model_name \
#   --data CERNET \
#   --features M \
#   --freq t \
#   --seq_len 5 \
#   --label_len 1 \
#   --pred_len 1 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 196 \
#   --dec_in 196 \
#   --c_out 196 \
#   --batch_size 16 \
#   --d_model 512 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --lstm_layer_num 1 \
#   --lstm_hidden_dims 512 \
  # --loss 'RMSE' \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/CERNET/\
  --data_path  CERNET-fixdate.csv \
  --model_id CERNET_5_1_4l \
  --model $model_name \
  --data CERNET \
  --features M \
  --freq t \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 196 \
  --dec_in 196 \
  --c_out 196 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --lstm_layer_num 4 \
  --lstm_hidden_dims 512 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/CERNET/\
  --data_path  CERNET-fixdate.csv \
  --model_id CERNET_5_1_8l \
  --model $model_name \
  --data CERNET \
  --features M \
  --freq t \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 196 \
  --dec_in 196 \
  --c_out 196 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --lstm_layer_num 8 \
  --lstm_hidden_dims 512 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/CERNET/\
  --data_path  CERNET-fixdate.csv \
  --model_id CERNET_5_1_12l \
  --model $model_name \
  --data CERNET \
  --features M \
  --freq t \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 196 \
  --dec_in 196 \
  --c_out 196 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --lstm_layer_num 12 \
  --lstm_hidden_dims 512 \
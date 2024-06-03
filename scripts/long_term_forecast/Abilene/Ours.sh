export CUDA_VISIBLE_DEVICES=1

model_name=Ours

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/Abilene/\
  --data_path  AbileneTM-deoutlier.csv \
  --model_id Abilene_5_1 \
  --model $model_name \
  --data Abilene \
  --features M \
  --freq t \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 144 \
  --dec_in 144 \
  --c_out 144 \
  --batch_size 16 \
  --d_model 512 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 512 \
  # --loss 'RMSE' \
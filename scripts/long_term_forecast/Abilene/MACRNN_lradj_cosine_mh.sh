export CUDA_VISIBLE_DEVICES=1

model_name=MACRNN

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/Abilene/\
  --data_path  AbileneTM-deoutlier.csv \
  --model_id Abilene_12_1_deoutlier_cosineadj_mh_2 \
  --model $model_name \
  --data Abilene \
  --features M \
  --n_heads 2 \
  --dropout 0.5 \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 288 \
  --matrix_dim 12 \
  --seq_len 12 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 144 \
  --c_out 144 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1000 \
  --learning_rate 0.0001 \
  --patience 1000 \
  --lradj cosine \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/Abilene/\
  --data_path  AbileneTM-deoutlier.csv \
  --model_id Abilene_12_1_deoutlier_cosineadj_mh_4 \
  --model $model_name \
  --data Abilene \
  --features M \
  --n_heads 4 \
  --dropout 0.5 \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 288 \
  --matrix_dim 12 \
  --seq_len 12 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 144 \
  --c_out 144 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1000 \
  --learning_rate 0.0001 \
  --patience 1000 \
  --lradj cosine \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/Abilene/\
  --data_path  AbileneTM-deoutlier.csv \
  --model_id Abilene_12_1_deoutlier_cosineadj_mh_8 \
  --model $model_name \
  --data Abilene \
  --features M \
  --n_heads 8 \
  --dropout 0.5 \
  --lstm_layer_num 2 \
  --lstm_hidden_dims 288 \
  --matrix_dim 12 \
  --seq_len 12 \
  --label_len 0 \
  --pred_len 1 \
  --enc_in 144 \
  --c_out 144 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 1000 \
  --learning_rate 0.0001 \
  --patience 1000 \
  --lradj cosine \
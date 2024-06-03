export CUDA_VISIBLE_DEVICES=1

model_name=TimesNet

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/TM/GEANT/\
  --data_path  GEANT_full_spine.csv \
  --model_id GEANT_5_1 \
  --model $model_name \
  --data GEANT \
  --features M \
  --freq t \
  --seq_len 5 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 529 \
  --dec_in 529 \
  --c_out 529 \
  --batch_size 16 \
  --d_model 32 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Yearly' \
#   --model_id m4_Yearly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 32 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Quarterly' \
#   --model_id m4_Quarterly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 64 \
#   --d_ff 64 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id m4_Daily \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 16 \
#   --d_ff 16 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Weekly' \
#   --model_id m4_Weekly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Hourly' \
#   --model_id m4_Hourly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --batch_size 16 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

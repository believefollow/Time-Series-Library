export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 172 \
  --d_ff 172 \
  --e_layers 3 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 3 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 3 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 3 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 172 \
  --d_ff 172 \
  --e_layers 2 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 16 \
  --d_ff 16 \
  --e_layers 2 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 32 \
  --d_ff 32 \
  --e_layers 2 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model TimesNet \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 64 \
  --d_ff 64 \
  --e_layers 2 \
  --enc_in 172 \
  --c_out 172 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3
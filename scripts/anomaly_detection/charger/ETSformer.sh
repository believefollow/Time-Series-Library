export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger/ \
  --model_id MSL \
  --model ETSformer \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 100 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --d_layers 3 \
  --enc_in 172 \
  --c_out 172 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 10
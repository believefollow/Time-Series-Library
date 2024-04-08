export CUDA_VISIBLE_DEVICES=0
python --version
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/charger \
  --model_id CHG \
  --model Autoformer \
  --data CHG \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in 8 \
  --c_out 8 \
  --top_k 3 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --train_epochs 3 
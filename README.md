# SOEN-691
Fine-tuning

export CUDA_VISIBLE_DEVICES=0,1,2
python run.py \
 --output_dir=./saved_models_codes \
 --model_type=roberta \
 --config_name=microsoft/codebert-base \
 --model_name_or_path=microsoft/codebert-base \
 --tokenizer_name=roberta-base \
 --do_train \
 --do_eval \
 --train_data_file /dataset/train.jsonl \
 --query_data_file /dataset/Query.jsonl \
 --candidate_data_file  /dataset/Candidate.jsonl \
 --epoch 3 \
 --save_steps=50 \
 --block_size 512 \
 --train_batch_size 8 \
 --eval_batch_size 16 \
 --learning_rate 2e-5 \
 --max_grad_norm 1.0 \
 --evaluate_during_training \
 --seed 123456 

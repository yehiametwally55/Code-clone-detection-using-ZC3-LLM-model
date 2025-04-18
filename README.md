# ZC3 - Cross Code Clone Detection: A Replication Study

# Task definition
Given a code and a collection of candidates as the input, the task is to classify and retrieve codes with the same semantic from a collection of candidates. Models are evaluated by MAP score and F1-score. MAP is defined as the mean of average precision scores, which is evaluated for retrieving similar samples given a query. F1-score is a measure used to classify problems to evaluate a model's accuracy by balancing precision and recall.

# Structure
You can find the ZC3 model in the ZC3-main folder. The C4 model is in the C4.zip folder.

All the datasets (Atcoder, GPTClone-Bench, XLCost) we used are available and preprocessed.

We added all the preprocessing programs we used to format the data for the ZC3 model in the preprocessing folder.

# Dataset description
Given a file dataset/train.jsonl:

{"label": "65", "index": "0", "code": "function0", description: "Python"}

{"label": "65", "index": "1", "code": "function1", description: "java"}

{"label": "65", "index": "2", "code": "function2", description: "C"}

{"label": "66", "index": "3", "code": "function3", description: "Python"}

Where:

code: the source code

label: the identifier number that the same clones in different languages will have

index: the index of example

description: The programming language

This is the command for training and evaluating the model we used.

# Fine-tuning

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

# Zenodo
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15243978.svg)](https://doi.org/10.5281/zenodo.15243978)

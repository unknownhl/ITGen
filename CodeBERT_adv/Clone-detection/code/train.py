import os

os.system(f"CUDA_VISIBLE_DEVICES=2 python run.py \
        --output_dir=../saved_models \
        --model_type=roberta \
        --config_name=microsoft/codebert-base  \
        --model_name_or_path=microsoft/codebert-base  \
        --tokenizer_name=roberta-base \
        --do_train \
        --train_data_file=../../../dataset/Clone-detection/train_sampled.txt \
        --eval_data_file=../../../dataset/Clone-detection/valid_sampled.txt \
        --test_data_file=../../../dataset/Clone-detection/test_sampled.txt \
        --epoch 2 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --seed 123456  2>&1 | tee ../saved_models/train.log")
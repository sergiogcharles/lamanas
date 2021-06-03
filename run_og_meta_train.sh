#!/bin/bash

DATASET=omniglot
DATASET_DIR=/home/ubuntu/metanas_custom/omniglot
TRAIN_DIR=/home/ubuntu/metanas_custom/results
		
mkdir -p $TRAIN_DIR

args=(
    # Execution
    --name metatrain_og \
    --job_id 0 \
    --path ${TRAIN_DIR} \
    --data_path ${DATASET_DIR} \
    --dataset $DATASET
    --hp_setting 'og_metanas' \
    --use_hp_setting 1 \
    --workers 0 \
    --gpus 0 \
    --test_adapt_steps 1.0 \

    # few shot params
     # examples per class
    --n 5 \
    # number classes  
    --k 20 \
    # test examples per class
    --q 1 \

    --meta_model_prune_threshold 0.01 \
    --alpha_prune_threshold 0.01 \
    # Meta Learning
    --meta_model searchcnn \
    --meta_epochs 30 \
    --warm_up_epochs 0 \
    --use_pairwise_input_alphas \
    --eval_freq 1000 \
    --eval_epochs 5 \

    --normalizer softmax \
    --normalizer_temp_anneal_mode linear \
    --normalizer_t_min 0.05 \
    --normalizer_t_max 1.0 \
    --drop_path_prob 0.2 \

    # Architectures
    --init_channels 28 \
    --layers 4 \
    --reduction_layers 1 3 \
    --use_first_order_darts \
    --loss_nn loss_nn \
    --pretrained none \
    --use_torchmeta_loader \

)


python -u -m metanas.metanas_main "${args[@]}"


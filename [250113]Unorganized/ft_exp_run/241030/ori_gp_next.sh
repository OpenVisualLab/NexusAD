#!/bin/bash
# 微调示例，具体参数含义可参考: https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html

export CUDA_VISIBLE_DEVICES=1

# 执行swift命令
swift sft \
    --model_id_or_path "/data1/nemo/projects/pre_weights/LLaVa-NeXT-7B" \
    --model_type "llava1_6-vicuna-7b-instruct" \
    --system "You are a seasoned driver, skilled at handling corner cases." \
    --dataset "/data2/datasets/drivelm/data/qas/original/original_train_general_perception.jsonl" "/data2/datasets/drivelm/data/qas/original/original_val_general_perception.jsonl" \
    --lora_target_modules 'ALL' \
    --max_length 4096 \
    --batch_size "1" \
    --dataloader_num_workers "2" \
    --learning_rate "1e-4" \
    --num_train_epochs "4" \
    --gradient_accumulation_steps "16" \
    --eval_steps "1" \
    --save_steps "1" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_batch_size "1" \
    --dataset_test_ratio "1" \
    --save_total_limit -1 \
    --use_flash_attn True \
    --val_dataset "/data2/datasets/drivelm/data/qas/original/original_test_general_perception.jsonl" \
    --output_dir "/data2/datasets/drivelm/output3/241030_gp_ori_next" \
    --add_output_dir_suffix False

    # --device_map_config_path /data1/nemo/projects/spatial_ad/a_pin_jsons/device_maps/vl_26b_2.json \
    # --resume_from_checkpoint output/internvl2-26b/v7-20240730-020149/checkpoint-5790

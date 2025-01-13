#!/bin/bash
# 微调示例，具体参数含义可参考: https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html

export CUDA_VISIBLE_DEVICES=3

# 执行swift命令
swift sft \
    --model_id_or_path "/data1/nemo/projects/pre_weights/InternVL2-8B" \
    --resume_from_checkpoint "/data2/datasets/drivelm/output4/1112_ds_icl_ift/checkpoint-1737" \
    --model_type "internvl2-8b" \
    --system "You are a seasoned driver, skilled at handling corner cases." \
    --dataset "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft2/train_v_ds_v1_2.jsonl" "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft2/val_t_ds_v1_2.jsonl" \
    --lora_target_modules 'ALL' \
    --max_length 8192 \
    --batch_size "1" \
    --dataloader_num_workers "2" \
    --learning_rate "1e-4" \
    --num_train_epochs "6" \
    --gradient_accumulation_steps "16" \
    --eval_steps "1" \
    --save_steps "1" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --eval_batch_size "1" \
    --dataset_test_ratio "1" \
    --save_total_limit -1 \
    --use_flash_attn True \
    --val_dataset "/data2/datasets/drivelm/data/qas/new/new_mini_driving_suggestion.jsonl" \
    --output_dir "/data2/datasets/drivelm/output4/1112_ds_icl_ift" \
    --add_output_dir_suffix False
    # --device_map_config_path /data1/nemo/projects/spatial_ad/a_pin_jsons/device_maps/vl_26b_2.json \
    # --resume_from_checkpoint output/internvl2-26b/v7-20240730-020149/checkpoint-5790

#!/bin/bash

# 定义数据路径
data_root="/data2/datasets/drivelm/val_data"
orignal_path=""
extract_path=""
convert_path="$data_root/drivelm-nuscenes/v1_1_val_nus_q_only.json"
output_path="$data_root/drivelm-nuscenes/swift_v1_1_val_nus.jsonl"

# python3 extract_data.py "$orignal_path" "$extract_path"

# python3 convert_data.py "$extract_path" "$convert_path"

python3 /data1/nemo/projects/NexusAD/data_processing/convert2swift.py "$convert_path" "$output_path" "$data_root"
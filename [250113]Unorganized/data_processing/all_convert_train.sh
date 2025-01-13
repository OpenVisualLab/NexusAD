#!/bin/bash

# 定义数据路径
data_root="/data2/datasets/drivelm/data/"
orignal_path="$data_root/drivelm-nuscenes/v1_1_train_nus.json"
extract_path="$data_root/drivelm-nuscenes/temp_test.json"
convert_path="$data_root/drivelm-nuscenes/temp_test_eval.json"
output_path="$data_root/drivelm-nuscenes/swift_v1_1_train_nus.jsonl"

python3 extract_data.py "$orignal_path" "$extract_path"

python3 convert_data.py "$extract_path" "$convert_path"

python3 convert2swift.py "$convert_path" "$output_path" "$data_root"
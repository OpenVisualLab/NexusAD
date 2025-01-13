#!/bin/bash

# 定义数据路径
data_root="/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/data"
orignal_path="/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/data/train_sample.json"
extract_path="/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/data/test.json"
convert_path="$data_root/test_swift.json"
output_path="$data_root/test_swift_for_eval.jsonl"

python3 /data1/nemo/projects/NexusAD/data_processing/extract_data.py "$orignal_path" "$extract_path"

python3 /data1/nemo/projects/NexusAD/data_processing/convert_data.py "$extract_path" "$convert_path"

python3 /data1/nemo/projects/NexusAD/data_processing/convert2swift.py "$convert_path" "$output_path" "$data_root"
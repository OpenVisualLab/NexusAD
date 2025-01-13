import argparse
import json
import os
from tqdm import tqdm
from multiprocessing import Process, current_process

from vl2_8b import init_vl2_model, run_vl2_model

def load_jsonl_file(jsonl_path):
    with open(jsonl_path, 'r') as infile:
        return [json.loads(line) for line in infile]

def write_responses_to_file(out_json_path, results):
    with open(out_json_path, 'w', encoding='utf-8') as outf:
        json.dump(results, outf, indent=4, ensure_ascii=False)

def process_data_subset(data_subset, gpu_id, ckpt_dir, out_path):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    model, template = init_vl2_model(ckpt_dir=ckpt_dir, max_new_tokens=2048)

    results = []
    for info in tqdm(data_subset, desc=f"Process {current_process().name}", position=gpu_id):
        qa_id = info['id']
        image_paths = info['image']
        question = info['query']
        gt_answer = info['response']

        answer = run_vl2_model(model, template, question, image_paths)

        new_info = {'id': qa_id, 'image': image_paths, 'question': question, 'gt_answer': gt_answer, 'answer': answer}
        results.append(new_info)

    # 保存每个进程的结果到不同的临时文件
    with open(out_path, 'w', encoding='utf-8') as outf:
        json.dump(results, outf, indent=4, ensure_ascii=False)
    print(f'Process {current_process().name} finished writing to {out_path}')

def main(data_root, json_path, out_json_path, ckpt_dir=None, num_gpus=4):
    json_info = load_jsonl_file(json_path)
    
    # 将数据按 GPU 数量分割
    chunk_size = len(json_info) // num_gpus
    chunks = [json_info[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]
    if len(json_info) % num_gpus != 0:
        chunks[-1].extend(json_info[num_gpus * chunk_size:])
    
    processes = []
    temp_files = []

    for gpu_id in range(num_gpus):
        out_path = f"temp_output_gpu_{gpu_id}.json"
        temp_files.append(out_path)
        p = Process(target=process_data_subset, args=(chunks[gpu_id], gpu_id, ckpt_dir, out_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 合并所有进程的输出
    combined_results = []
    for temp_file in temp_files:
        with open(temp_file, 'r', encoding='utf-8') as infile:
            combined_results.extend(json.load(infile))
        os.remove(temp_file)

    # 保存最终的结果
    write_responses_to_file(out_json_path, combined_results)
    print(f'Finished writing combined results to {out_json_path}')

if __name__ == "__main__":
    
    os.environ['INPUT_SIZE'] = '448'
    os.environ['MAX_NUM'] = '12'
    
    parser = argparse.ArgumentParser(description='Process a subset and task of the CODA dataset using a specified model.')
    parser.add_argument('--data_root', type=str, default='/data1/nemo/projects/240922_AD/datasets/images', help='Root directory of datasets')
    parser.add_argument('--in_json_path', type=str, default='/data2/datasets/drivelm/val_data/swift_v1_1_val_nus.jsonl', help='Path to the input JSON file')
    parser.add_argument('--out_json_path', type=str, default='/data2/datasets/drivelm/output/val_pred_8b_ft_20_241022.json', help='Path to the output JSON file')
    parser.add_argument('--ckpt_dir', type=str, default='/data2/datasets/drivelm/output/internvl2-8b/v1-20241014-120522/checkpoint-36800', help='Directory of model weights')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs to use')

    args = parser.parse_args()
    main(args.data_root, args.in_json_path, args.out_json_path, args.ckpt_dir, args.num_gpus)

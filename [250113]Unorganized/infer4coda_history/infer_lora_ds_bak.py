import argparse
import json
from tqdm import tqdm
import os

from vl2_8b import init_vl2_model, run_vl2_model


def load_json_lines(json_path):
    with open(json_path, 'r') as f:
        return f.read().splitlines()


def find_start_index(lines, last_question_id):
    return next((i for i, line in enumerate(lines) if json.loads(line)['question_id'] == last_question_id), None)


def batch_lines(lines, batch_size):
    return [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]


def process_batch_lines(json_path, last_question_id=None, batch_size=100):
    lines = load_json_lines(json_path)
    if last_question_id:
        start_index = find_start_index(lines, last_question_id)
        if start_index is not None:
            lines = lines[start_index + 1:]
    return batch_lines(lines, batch_size)


def get_last_question_id(out_json_path):
    if not os.path.exists(out_json_path) or os.path.getsize(out_json_path) == 0:
        return None
    lines = load_json_lines(out_json_path)
    return json.loads(lines[-1])['question_id'] if lines else None


def prepare_batch_data(data_batch, data_root):
    # for swift2
    # if 'data1' in data_batch[0]['images'[0]]
    
    # questions = [data['question'] for data in data_batch]
    # full_image_paths = [os.path.join(data_root, data['image'][0]) for data in data_batch]
    
    questions = [data['query'] for data in data_batch]
    full_image_paths = [os.path.join(data_root, data['images'][0]) for data in data_batch]
    
    # # for coda-lm
    # questions = [data['question'] for data in data_batch]
    # full_image_paths = [os.path.join(data_root, data['image']) for data in data_batch]
    
    return questions, full_image_paths


def write_responses_to_file(outf, responses, data_batch):
    for response, data in zip(responses, data_batch):
        data['prediction'] = response
        outf.write(json.dumps(data) + '\n')


def main(data_root, model_dir, ckpt_dir, json_path, out_json_path, batch_size=1, tqdm_position=0): #TODO
    ckpt_dir = None if ckpt_dir=="" else ckpt_dir
        
    last_question_id = get_last_question_id(out_json_path)
    batched_lines = process_batch_lines(json_path, last_question_id, batch_size)
    
    model, template = init_vl2_model(model_path=model_dir,ckpt_dir=ckpt_dir)
    
    print(f'Processing: {json_path}')
    print(f'Saving in: {out_json_path}')
    
    total_samples = sum([len(batch) for batch in batched_lines])  # 计算总样本数

    # 使用 tqdm 显示样本进度条，并传递 position 参数
    with open(out_json_path, 'a') as outf, tqdm(total=total_samples, position=tqdm_position, leave=True, desc='Sample Progress') as pbar:
        for batch in batched_lines:
            data_batch = [json.loads(line) for line in batch]
            questions, full_image_paths = prepare_batch_data(data_batch, data_root)
            
            responses = [run_vl2_model(model, template, questions[0], full_image_paths[0])]

            write_responses_to_file(outf, responses, data_batch)
            
            # 更新进度条
            pbar.update(len(batch))
    
    print(f'Finished writing to {out_json_path}')


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    parser = argparse.ArgumentParser(description='Process a subset and task of the CODA dataset using a specified model.')
    parser.add_argument('--data_root', type=str, default='/data1/nemo/projects/240922_AD/datasets/images', help='Root directory of datasets')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--model_dir', type=str, default='/data1/nemo/projects/pre_weights/CogVLM2-19B', help='Directory of model weights')
    parser.add_argument('--ckpt_dir', type=str, default='', help='Directory of model weights')
    parser.add_argument('--json_path', type=str, default='/data2/datasets/drivelm/data/qas/original/original_test_region_perception.jsonl', help='Path to the input JSON file')
    parser.add_argument('--out_json_path', type=str, default='/data2/datasets/drivelm/output2/nexusad/241030_cogvlm/ori_test_rp_cogvlm.jsonl', help='Path to the output JSON file')
    parser.add_argument('--tqdm_position', type=int, default=0, help='Position of tqdm progress bar')

    args = parser.parse_args()
    main(args.data_root, args.model_dir, args.ckpt_dir, args.json_path, args.out_json_path, args.batch_size, args.tqdm_position)

import os
import json
from tqdm import tqdm
import language_evaluation
from pprint import PrettyPrinter

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def calculate_average(metrics_list):
    # Initialize a dictionary to store the sum of each metric
    sums = {}
    num_dicts = len(metrics_list)
    
    # Loop through each dictionary in the list
    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in sums:
                sums[key] = 0
            sums[key] += value
    
    # Calculate the average for each metric
    averages = {key: value / num_dicts for key, value in sums.items()}
    return averages

def split_list_into_chunks(input_list, chunk_size=16):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

# base_path = "/data2/datasets/drivelm/output4/res_1108_rp_icl_ft/ori_rp_icl_run_0.jsonl"
# base_path = "/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_rp_run.jsonl"

base_path = "/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/eval_test_region_perception.jsonl"

ours_path = "/data2/datasets/drivelm/output4/res_1108_rp_icl_ft/ft_rp_icl_run_4.jsonl"
gt_path = "/data2/datasets/drivelm/output2/nexusad/241025_ori_temp/ori_vl_8b_test_rp_ft_4.jsonl"

gt_res = read_jsonl(gt_path)
base = read_jsonl(base_path)
ours = read_jsonl(ours_path)

# print(ours[0].keys())

gt = [d['response'][:512] for d in base]

pred_base = [d['prediction'][:512] for d in base]
pred_ours = [d['prediction'][:512] for d in ours]



evaluator = language_evaluation.CocoEvaluator()

# rec_list = []
# for p,g in tqdm(zip(pred_base, gt), total=len(gt)):
#     rec_list.append(evaluator.run_evaluation(p, g))
    
# print(calculate_average(rec_list))


gt_c = split_list_into_chunks(gt)
pred_ours_c = split_list_into_chunks(pred_ours)
pred_base_c = split_list_into_chunks(pred_base)


# rec_list = []
# for p,g in tqdm(zip(pred_ours_c, gt_c), total=len(gt_c)):
#     rec_list.append(evaluator.run_evaluation(p, g))
    
# print(calculate_average(rec_list))

rec_list = []
for p,g in tqdm(zip(pred_base_c, gt_c), total=len(gt_c)):
    rec_list.append(evaluator.run_evaluation(p, g))
    
print(calculate_average(rec_list))


# pprint = PrettyPrinter().pprint
# pprint(results)

# ours = {'Bleu_1': 0.4121802835960635, 'Bleu_2': 0.2563510525984998, 'Bleu_3': 0.173402831443669, 'Bleu_4': 0.12131721395775023, 'METEOR': 0.18835445231006728, 'ROUGE_L': 0.29544214448526124, 'CIDEr': 0.3484922812185627, 'SPICE': 0.23394348274116272}

# internvl-8b(error): {'Bleu_1': 0.05506909756817279, 'Bleu_2': 0.00022994112952089675, 'Bleu_3': 3.715790328460483e-05, 'Bleu_4': 1.4947365927798772e-05, 'METEOR': 0.05478058416200274, 'ROUGE_L': 0.08175399102450417, 'CIDEr': 0.11278010183159219, 'SPICE': 0.015866009149287916}


# internvl-8b = {'Bleu_1': 0.32294573698725476, 'Bleu_2': 0.16860987015087595, 'Bleu_3': 0.0921608348954974, 'Bleu_4': 0.05170150184645904, 'METEOR': 0.14623892528780896, 'ROUGE_L': 0.22811402755664084, 'CIDEr': 0.13568406558732182, 'SPICE': 0.13944334233334837}


# gpt-4o = {'Bleu_1': 0.29326723278361144, 'Bleu_2': 0.1463249535547886, 'Bleu_3': 0.08084977313831784, 'Bleu_4': 0.04209328388428398, 'METEOR': 0.13935421516059113, 'ROUGE_L': 0.2037530346641165, 'CIDEr': 0.07994893125352802, 'SPICE': 0.1631079513736897}

# qwen-vl-max = {'Bleu_1': 0.30234286299201774, 'Bleu_2': 0.16537450280943303, 'Bleu_3': 0.09689411374145841, 'Bleu_4': 0.054314605350958424, 'METEOR': 0.14595699584066885, 'ROUGE_L': 0.22444662511676317, 'CIDEr': 0.07910815245879582, 'SPICE': 0.17869833065260426}


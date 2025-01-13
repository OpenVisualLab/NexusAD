


import ast
import re

import os
import json
from tqdm import tqdm

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')

def read_json(file_path):
    with open(file_path, 'r') as f:
     return json.load(f)

def read_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()


def extract_contribution_values(response):
    # 去除 ```python 和 ``` 标记
    cleaned_response = re.sub(r"```(?:python)?\n|```", "", response)

    # 定位到包含字典的部分
    start = cleaned_response.find("contribution_values = ")
    if start == -1:
        raise ValueError("No 'contribution_values' found in response.")
    
    # 提取出字典字符串并转换成 Python 字典
    dict_str = cleaned_response[start + len("contribution_values = "):]
    contribution_values = ast.literal_eval(dict_str)
    
    return contribution_values


def calculate_average_contributio_bak(contribution_list):
    # 定义标准化的类别名称
    standard_categories = {
        "vehicles": "vehicles",
        "vulnerable_road_users": "vulnerable_road_users",
        "traffic_signs": "traffic_signs",
        "traffic signs": "traffic_signs",
        "traffic_lights": "traffic_lights",
        "traffic lights": "traffic_lights",
        "traffic_cones": "traffic_cones",
        "traffic cones": "traffic_cones",
        "barriers": "barriers",
        "other_objects": "other_objects",
        "other objects": "other_objects"
    }
    
    # 初始化每个标准类别的累计贡献值
    total_contributions = {category: [0.0, 0.0] for category in set(standard_categories.values())}
    num_entries = len(contribution_list)
    
    # 累加每个标准类别的贡献值
    for contribution in contribution_list:
        for key, value in contribution.items():
            standardized_key = standard_categories.get(key, key)  # 将键名转换为标准名称
            if standardized_key in total_contributions:
                total_contributions[standardized_key][0] += value[0]
                total_contributions[standardized_key][1] += value[1]
    
    # 计算每个类别的平均贡献度
    average_contributions = {
        category: [total_contributions[category][0] / num_entries,
                   total_contributions[category][1] / num_entries]
        for category in total_contributions
    }
    
    return average_contributions


def collect_nonzero_contributions(contribution_list):
    # 定义标准化的类别名称
    standard_categories = {
        "vehicles": "vehicles",
        "vulnerable_road_users": "vulnerable_road_users",
        "traffic_signs": "traffic_signs",
        "traffic signs": "traffic_signs",
        "traffic_lights": "traffic_lights",
        "traffic lights": "traffic_lights",
        "traffic_cones": "traffic_cones",
        "traffic cones": "traffic_cones",
        "barriers": "barriers",
        "other_objects": "other_objects",
        "other objects": "other_objects"
    }
    
    # 初始化每个类别的列表，用于存储非零贡献值
    contribution_values = {category: [] for category in set(standard_categories.values())}
    
    # 遍历每个字典，处理各类别的贡献值
    for contribution in contribution_list:
        # 创建包含默认值的完整字典
        normalized_contribution = {category: [0.0, 0.0] for category in contribution_values}
        
        # 更新字典中的存在的类别值
        for key, value in contribution.items():
            standardized_key = standard_categories.get(key, key)
            if standardized_key in normalized_contribution:
                normalized_contribution[standardized_key] = value
        
        # 将非零贡献值添加到对应类别的列表中
        for category, value in normalized_contribution.items():
            if value != [0.0, 0.0]:  # 仅添加非零贡献值
                contribution_values[category].append(value)
                
    # 计算每个类别的平均贡献度
    average_contributions = {}
    for category, values in contribution_values.items():
        # print(category, len(values))
        if values:  # 如果列表非空
            description_avg = sum(v[0] for v in values) / (len(values))**2  # 考虑图像中的目标数量
            explanation_avg = sum(v[1] for v in values) / (len(values))**2
            average_contributions[category] = [description_avg, explanation_avg, len(values)]
        else:
            average_contributions[category] = [0.0, 0.0, 0]  # 若类别无非零值，设为0
    
    return average_contributions

def normalize_to_one(data):
    # Flatten all values and calculate the total sum
    all_values = [v for values in data.values() for v in values]
    total_sum = sum(all_values)
    
    # Normalizing each value
    normalized_data = {key: [v / total_sum for v in values] for key, values in data.items()}
    return normalized_data

score_info = read_jsonl("/data1/nemo/projects/NexusAD/1.nexus_retrieval/11111/svae_dir/gp_scores.jsonl")
# score_info = read_jsonl("/data1/nemo/projects/NexusAD/1.nexus_retrieval/11111/svae_dir/ds_scores.jsonl")


score_list = []
for info in tqdm(score_info):
    score = extract_contribution_values(info['scores'])
    score = normalize_to_one(score)
    score_list.append(score)

avg =  collect_nonzero_contributions(score_list)

print(avg)

# print(normalize_to_one(avg))

{'other_objects': [0.00054581814467776, 0.0006386798855737266, 181], 'traffic_signs': [0.0006285764864711771, 0.0007931302369169014, 151], 'vehicles': [0.0006195016889334009, 0.0006012173586662003, 465], 'traffic_cones': [0.0008083261738395045, 0.000957866237819896, 129], 'traffic_lights': [0.0012841599143850015, 0.0016193145378323697, 76], 'barriers': [0.0004614140649921071, 0.0005467892171554832, 228], 'vulnerable_road_users': [0.0005567609688740336, 0.0006636747213421017, 233]}


{'other_objects': [0.0006316534375317782, 0.0007504287080930263, 184], 'traffic_signs': [0.0007402974227390827, 0.0008497651670582599, 151], 'barriers': [0.0005488150034198396, 0.000598118288538979, 230], 'traffic_lights': [0.0015005011428759604, 0.0017769481179181954, 82], 'vulnerable_road_users': [0.0006548018407078118, 0.0007229108661674124, 233], 'vehicles': [0.0005153758878582254, 0.0005180836545143274, 465], 'traffic_cones': [0.0010322944920784119, 0.001064811451660405, 131]}



# 全局感知
# {'vehicles': [0.5755999999999993, 0.5589999999999995], 'traffic_lights': [0.03329999999999999, 0.043500000000000004], 'barriers': [0.10940000000000012, 0.13440000000000013], 'other_objects': [0.0793000000000001, 0.0971000000000001], 'vulnerable_road_users': [0.13699999999999998, 0.16979999999999992], 'traffic_cones': [0.06189999999999999, 0.07510000000000003], 'traffic_signs': [0.06740000000000004, 0.08840000000000003]}
# {'vehicles': [0.2579777698099675, 0.2505378271781999], 'traffic_lights': [0.014924704195051989, 0.019496235209752605], 'barriers': [0.04903191107923994, 0.06023664395840811], 'other_objects': [0.035541412692721465, 0.04351918250268919], 'vulnerable_road_users': [0.06140193617784153, 0.07610254571531014], 'traffic_cones': [0.027742918608820367, 0.03365901756902118], 'traffic_signs': [0.030207959842237387, 0.039619935460738644]}

gp = {'barriers': [0.04797229750909939, 0.05684858132922128], 'traffic_cones': [0.026902711717726388, 0.03187970412712177], 'vehicles': [0.26790350537924923, 0.2599964467551983], 'traffic_lights': [0.014834615330975538, 0.018706321541039538], 'vulnerable_road_users': [0.060451992478404826, 0.07206047389388272], 'other_objects': [0.03576309647557619, 0.04184758346256172], 'traffic_signs': [0.028664344936058614, 0.03616832506388454]}

# 驾驶建议
# {'other_objects': [0.12939999999999996, 0.16289999999999996], 'traffic_signs': [0.12519999999999995, 0.14620000000000005], 'traffic_lights': [0.06720000000000001, 0.0812], 'vehicles': [0.607799999999999, 0.6090000000000013], 'traffic_cones': [0.12330000000000003, 0.12990000000000002], 'barriers': [0.19569999999999996, 0.22], 'vulnerable_road_users': [0.23239999999999994, 0.2602000000000002]}

# {'traffic_cones': [0.039897747864354134, 0.042033393735438775], 'traffic_lights': [0.021744757960134607, 0.02627491586849598], 'traffic_signs': [0.04051255500906029, 0.04730779187160239], 'vehicles': [0.19667356976443143, 0.19706186901372028], 'vulnerable_road_users': [0.07520062127879883, 0.08419622055397365], 'other_objects': [0.04187160238156871, 0.052711623090862], 'barriers': [0.06332513590473722, 0.07118819570282163]}

ds = {'vulnerable_road_users': [0.07109707426037278, 0.07849221602672529], 'barriers': [0.05806462736181902, 0.06328091492742395], 'other_objects': [0.04277051756215176, 0.050813028682394984], 'vehicles': [0.2228743027042895, 0.22404527639472085], 'traffic_signs': [0.03375904307174764, 0.03875099114819076], 'traffic_lights': [0.020178739369395914, 0.023896398289763882], 'traffic_cones': [0.035430411557115245, 0.03654645864388841]}
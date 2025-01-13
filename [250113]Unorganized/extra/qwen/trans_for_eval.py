import os
import json
from tqdm import tqdm

import os
from openai import OpenAI
import dashscope
import json
from tqdm import tqdm

os.environ['https_proxy'] = ''   

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def qwen_max(query, image_path):
    messages=[{"role": "user",
           "content": [
        {
            "image": image_path
        },
        {"text": query},
        ]}]

    return dashscope.MultiModalConversation.call(
        api_key="sk-4703fcbe20ba4ce1b306a1f465e58c17",
        model='qwen-vl-max',
        messages=messages
        )

ds_lorem = """Stay Calm and Alert: In extreme scenarios, it is essential to stay composed. Focus fully on the road and be mindful of surroundings, including other vehicles, obstacles, and any unusual conditions.
	2.	Adjust Speed and Distance: Slow down to give yourself more time to react. Increase the following distance to the vehicle in front to allow for extended braking time, especially on slippery or uneven roads.
	3.	Adapt Steering and Braking: Avoid sharp turns and sudden braking, which can lead to loss of control, particularly in wet, icy, or rough conditions. Use smooth, gradual movements to steer, and gently tap the brakes if needed.
	4.	Use Appropriate Lighting: Turn on headlights in low-visibility situations (fog, rain, or darkness) so others can see you better. Avoid using high beams in fog or heavy rain, as they can reflect back and reduce visibility.
	5.	Mind the Terrain: If driving through mountainous, rough, or off-road areas, be aware of sudden changes in elevation, loose surfaces, and rocks. Drive carefully and avoid speeding, as these terrains can shift unexpectedly.
	6.	Watch for Hazards: In extreme weather, obstacles like fallen branches, debris, or flooding may appear. Stay vigilant, scan the road ahead, and take extra caution near hills, bends, and intersections.
	7.	Prepare for Emergencies: Carry essential equipment, such as a first-aid kit, flashlight, warning triangles, and an emergency blanket. In remote areas or during harsh weather, have extra fuel, water, and a charged phone for safety.
	8.	Check Vehicle Condition: Ensure your vehicle is suited for the conditions. Check tire pressure, brakes, and fluid levels. For winter, consider snow tires or chains; for off-road or rocky terrain, check for reinforced tires and adequate clearance."""  
    
# ori_json = "/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/original_test_general_perception.jsonl"
# out_json = os.path.join(os.path.dirname(ori_json), "eval_test_general_perception.jsonl")

ori_json = "/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/original_test_driving_suggestion.jsonl"
out_json = os.path.join(os.path.dirname(ori_json), "eval_test_driving_suggestion.jsonl")

ori_json = "/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/original_test_region_perception.jsonl"
out_json = os.path.join(os.path.dirname(ori_json), "eval_test_region_perception.jsonl")


ori_infos = read_jsonl(ori_json)

out_list = []
for ori in tqdm(ori_infos):
    if ori["res"]["status_code"] == 200:
        ori["prediction"] = ori["res"]["output"]["choices"][0]["message"]["content"][0]["text"]
    else:
        print(ori["images"])
        response = qwen_max(ori['query'], ori['images'][0])
        ori['res'] = response
        print(response) 
        if ori["res"]["status_code"] == 200:   
            ori["prediction"] = ori["res"]["output"]["choices"][0]["message"]["content"][0]["text"]
        else:
            ori["prediction"] = "Slow Down!"
        
    out_list.append(ori)

with open(out_json, 'w') as jsonl_file:
    for out in out_list:
        jsonl_file.write(json.dumps(out) + '\n')
    
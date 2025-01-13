import numpy as np
import json
import os


import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import re
import argparse

def stitch_images(image_paths, output_folder='.', resize_dim=(896, 448), border_thickness=1):
    labels = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    # Load and resize images
    images = []
    for label, color in zip(labels, colors):
        image_path = next((path for path in image_paths if f"__{label}__" in path), None)
        if image_path is None:
            raise FileNotFoundError(f"Missing image for label: {label}")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"图像无法读取，请检查路径是否正确: {image_path}")
        img = cv2.resize(img, resize_dim)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, border_thickness)
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, lineType=cv2.LINE_AA)
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        images.append(img)

    # Stitch images together
    final_image = np.vstack((np.hstack(images[:3]), np.hstack(images[3:])))

    # Save final stitched image
    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.basename(image_paths[0]).split('__')[0] + os.path.basename(image_paths[0]).split('__')[2]
    output_path = os.path.join(output_folder, base_name)
    cv2.imwrite(output_path, final_image)

    return output_path


def convert_coordinates(targets, resize_dim=(896, 448), orignal_dim=(1600,900), _normal=False):
    stitch_dim = (2688, 896)

    offsets = {
        "CAM_FRONT_LEFT": (0, 0),
        "CAM_FRONT": (resize_dim[0], 0),
        "CAM_FRONT_RIGHT": (2 * (resize_dim[0]), 0),
        "CAM_BACK_RIGHT": (0, resize_dim[1]),
        "CAM_BACK": (resize_dim[0], resize_dim[1]),
        "CAM_BACK_LEFT": (2 * (resize_dim[0]), resize_dim[1])
    }
    updated_targets = []
    for target in targets:
        # print(target)
        target_id, cam_label, x, y = target.split(',')
        if cam_label in offsets:
            offset_x, offset_y = offsets[cam_label]
            target_x = int(float(x) / orignal_dim[0] * resize_dim[0]) + offset_x
            target_y = int(float(y) / orignal_dim[1] * resize_dim[1]) + offset_y
            if _normal:
                target_x = int(target_x*1000./stitch_dim[0])
                target_y = int(target_y*1000./stitch_dim[0])
            updated_targets.append(f"<{target_id},{cam_label},{target_x},{target_y}>")
    # print(targets, updated_targets)
    # aa
    return updated_targets

def extract_targets(response):
    # return re.findall(r'<([^>]+)>', response)  # 这个会匹配到'<and>'导致后面需要写判断
    return re.findall(r'<([a-zA-Z0-9_]+,[A-Z_]+,[\d.]+,[\d.]+)>', response)

def convert_qa_coordinates(qa):
    targets = extract_targets(qa)
    if targets:
        updated_targets = convert_coordinates(targets)
        qa = re.sub(r'<([^>]+)>', lambda m: updated_targets.pop(0) if updated_targets else m.group(0), qa)
    return qa           
    

def convert2llama(root, dst, data_root):
    stitch_images_dir = os.path.join(data_root, 'nuscenes', 'stitched')
    with open(root, 'r') as f:
        test_file = json.load(f)

    output = []
    for scene_id in tqdm(test_file.keys()):
        scene_data = test_file[scene_id]['key_frames']

        for frame_id in scene_data.keys():
            image_paths = scene_data[frame_id]['image_paths']
            image_paths = [image_paths[key].replace("../", "") for key in image_paths.keys()]
            
            image_paths = [os.path.join(data_root, i_p) for i_p in image_paths]

            frame_data_qa = scene_data[frame_id]['QA']
            QA_pairs = frame_data_qa["perception"] + frame_data_qa["prediction"] + frame_data_qa["planning"] + frame_data_qa["behavior"]
            
            image_output_path = stitch_images(image_paths, output_folder=stitch_images_dir)
            
            for idx, qa in enumerate(QA_pairs):
                question = convert_qa_coordinates(qa['Q'])
                answer = convert_qa_coordinates(qa['A'])
                
                output.append(
                    {
                        "id": scene_id + "_" + frame_id + "_" + str(idx),
                        "image": [image_output_path],
                        "query": question,
                        "response": answer
                        }
                    
                )
    
    with open(dst, 'w') as f:
        for item in output:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    # data_root = "/data2/datasets/drivelm/data"
    
    # root = os.path.join(data_root, "v1_1_train_nus.json")
    # dst = os.path.join(data_root, "swift_train_nus.jsonl")
    # convert2llama(root, dst, data_root)
    
     # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Convert data to swift format")
    
    # 添加参数
    parser.add_argument("input_path", type=str, help="Path to the input JSON file")
    parser.add_argument("output_path", type=str, help="Path to save the converted JSONL file")
    parser.add_argument("data_root", type=str, help="Root path of the data directory")

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 convert2llama 函数
    convert2llama(args.input_path, args.output_path, args.data_root)

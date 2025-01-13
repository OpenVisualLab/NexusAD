class TaskConfig:
    DEFAULTS = {
        "model_dir": "/default/model_dir",
        "ckpt_dir": "/default/ckpt_dir",
        "json_path": "/default/json.json",
        "out_json_path": "/default/out_json.json",
        "gpu": [0],
    }

    def __init__(self, model_dir=None, ckpt_dir=None, json_path=None, out_json_path=None, gpu=None):
        self.model_dir = model_dir or self.DEFAULTS["model_dir"]
        self.ckpt_dir = ckpt_dir or self.DEFAULTS["ckpt_dir"]
        self.json_path = json_path or self.DEFAULTS["json_path"]
        self.out_json_path = out_json_path or self.DEFAULTS["out_json_path"]
        self.gpu = gpu or self.DEFAULTS["gpu"]

    def to_dict(self):
        return {
            "model_dir": self.model_dir,
            "ckpt_dir": self.ckpt_dir,
            "json_path": self.json_path,
            "out_json_path": self.out_json_path,
            "gpu": self.gpu,
        }

class Config:
    DEFAULTS = {
        "data_root": "/default/data_root",
        "batch_size": 1,
    }

    def __init__(self, data_root=None, batch_size=None, manual_tasks=None, structured_tasks=None):
        self.data_root = data_root or self.DEFAULTS["data_root"]
        self.batch_size = batch_size or self.DEFAULTS["batch_size"]
        self.tasks = self._combine_tasks(manual_tasks, structured_tasks)

    def _combine_tasks(self, manual_tasks, structured_tasks):
        manual_tasks = manual_tasks or []
        structured_tasks = structured_tasks or []
        return manual_tasks + structured_tasks

    def to_dict(self):
        return {
            "data_root": self.data_root,
            "batch_size": self.batch_size,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def generate_structured_tasks(ckpt_dirs, gpu_list=None):
    gpu_list = [1, 2, 3]
    tasks = []
    for idx, ckpt in enumerate(ckpt_dirs):
        tasks.append(TaskConfig(
            model_dir='/data1/nemo/projects/pre_weights/InternVL2-8B',
            ckpt_dir=ckpt,
            json_path="/data2/datasets/drivelm/val_data/swift_v1_1_val_nus.jsonl",
            out_json_path=f"/data1/nemo/projects/241009_DriveLM/241125_8b_20/out_json_{idx+1}.json",
            gpu=[gpu_list[idx%len(gpu_list)]]
        ))
    return tasks

import os, re
def extract_checkpoints(base_dir):
    checkpoints = []
    for folder in os.listdir(base_dir):
        match = re.match(r'checkpoint-(\d+)', folder)
        if match:
            checkpoints.append((int(match.group(1)), folder))
    # 根据编号排序
    checkpoints.sort(key=lambda x: x[0])
    return [os.path.join(base_dir, folder) for _, folder in checkpoints]

base_ckpt_dir = "/data2/datasets/drivelm/output/internvl2-8b/v7-20241121-225041"
structured_tasks = generate_structured_tasks(
    ckpt_dirs=extract_checkpoints(base_ckpt_dir),
    gpu_list=[1, 2, 3]
)

CONFIG = Config(
    data_root="/path/to/custom_data_root",
    batch_size=1,
    # manual_tasks=[
    #     TaskConfig(
    #         model_dir="/custom/manual_model_dir",
    #         ckpt_dir="/custom/manual_ckpt_dir",
    #         json_path="/custom/manual_json.json",
    #         out_json_path="/custom/manual_out.json",
    #         gpu=[2]
    #     )
    # ],
    structured_tasks=structured_tasks
)

if __name__ == "__main__":
    print(CONFIG.to_dict())

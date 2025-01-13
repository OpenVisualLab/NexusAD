import yaml
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def load_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"[Error] 加载配置文件失败: {e}")

def validate_yaml_structure(config):
    required_config_keys = ['config', 'tasks']
    required_task_keys = ['model_dir', 'ckpt_dir', 'json_path', 'out_json_path']

    for key in required_config_keys:
        if key not in config:
            raise ValueError(f"[Error] 配置文件中缺少必要的键: '{key}'")

    if not isinstance(config.get('tasks'), list) or not config['tasks']:
        raise ValueError("[Error] 配置文件中 'tasks' 必须是非空列表")

    for i, task in enumerate(config['tasks']):
        for key in required_task_keys:
            if key not in task:
                raise ValueError(f"[Error] 任务 {i} 中缺少必要的键: '{key}'")

def ensure_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

def create_gpu_locks(tasks):
    gpu_locks = {}
    for task in tasks:
        for gpu in task.get('gpu', [0]):
            if gpu not in gpu_locks:
                gpu_locks[gpu] = threading.Lock()
    return gpu_locks

def run_task(task, data_root, batch_size, task_path, gpu_locks, position, log_dir):
    model_dir = task['model_dir']
    ckpt_dir = task['ckpt_dir']
    json_path = task['json_path']
    out_json_path = task['out_json_path']
    gpu_list = task.get('gpu', [0])

    log_file = os.path.join(log_dir, f"task_gpu_{'_'.join(map(str, gpu_list))}_{os.path.basename(json_path)}.log")
    print(f"[Running] GPUs {gpu_list}: {os.path.basename(json_path)} - Logging to {log_file}")

    for gpu in gpu_list:
        gpu_locks[gpu].acquire()

    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_list))

        with open(log_file, 'w') as log:
            proc = subprocess.Popen([
                'python', task_path,
                # '--data_root', data_root,
                # '--batch_size', str(batch_size),
                # '--model_dir', model_dir,
                '--ckpt_dir', ckpt_dir,
                '--json_path', json_path,
                '--out_json_path', out_json_path,
                # '--tqdm_position', str(position)
            ], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1)
            proc.wait()

        if proc.returncode == 0:
            print(f"[Completed] GPUs {gpu_list}: {os.path.basename(json_path)}")
        else:
            print(f"[Error] GPUs {gpu_list}: {os.path.basename(json_path)} failed with return code {proc.returncode}")

    except Exception as e:
        print(f"[Exception] GPUs {gpu_list}: {os.path.basename(json_path)} - Exception occurred: {e}")

    finally:
        for gpu in gpu_list:
            gpu_locks[gpu].release()

def main(config_path, task_path, log_dir):
    ensure_log_dir(log_dir)

    try:
        config = load_yaml(config_path)
        validate_yaml_structure(config)
    except ValueError as e:
        print(e)
        return

    data_root = config['config'].get('data_root')
    batch_size = config['config'].get('batch_size')

    if not data_root or not batch_size:
        print("[Error] 配置文件缺少 'data_root' 或 'batch_size'")
        return

    tasks = config['tasks']
    gpu_locks = create_gpu_locks(tasks)

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_task, task, data_root, batch_size, task_path, gpu_locks, i, log_dir)
            for i, task in enumerate(tasks)
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[Exception] 任务执行中出现异常: {e}")

if __name__ == '__main__':
    infer_dir = '/data1/nemo/projects/NexusAD/infer4drivelm'
    main(
        config_path=os.path.join(infer_dir, 'yamls/temp2.yaml'), 
        task_path=os.path.join(infer_dir, 'infer_lora_ds.py'),
        log_dir=os.path.join(infer_dir, 'log')
    )

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
from config import CONFIG
import json


def ensure_log_dir(log_dir):
    os.makedirs(log_dir, exist_ok=True)


def create_gpu_locks(tasks):
    gpu_locks = {}
    for task in tasks:
        for gpu in task.get('gpu', []):
            if gpu not in gpu_locks:
                gpu_locks[gpu] = threading.Lock()
    return gpu_locks


def generate_log_filename(log_dir, task_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"task_{task_index}_{timestamp}.log"
    return os.path.join(log_dir, filename)


def run_task(task, data_root, batch_size, task_path, gpu_locks, task_index, log_dir):
    model_dir = task['model_dir']
    ckpt_dir = task['ckpt_dir']
    json_path = task['json_path']
    out_json_path = task['out_json_path']
    gpu_list = task.get('gpu', [])

    log_file = generate_log_filename(log_dir, task_index)
    print(f"[Running] Task {task_index} on GPUs {gpu_list} - Logging to {log_file}")

    for gpu in gpu_list:
        gpu_locks[gpu].acquire()

    try:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_list[0]) if len(gpu_list) == 1 else ','.join(map(str, gpu_list))

        with open(log_file, 'w') as log:
            log.write("[Task Config]\n")
            log.write(f"Data root: {data_root}\n")
            log.write(f"Batch size: {batch_size}\n")
            log.write("Task details:\n")
            log.write(json.dumps(task, indent=4))
            log.write("\n\n[Execution Log]\n")

        with open(log_file, 'a') as log:
            proc = subprocess.Popen([
                'python', task_path,
                '--model_dir', model_dir,
                '--ckpt_dir', ckpt_dir,
                '--json_path', json_path,
                '--out_json_path', out_json_path,
            ], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1)
            proc.wait()

        if proc.returncode == 0:
            print(f"[Completed] Task {task_index} on GPUs {gpu_list}")
        else:
            print(f"[Error] Task {task_index} on GPUs {gpu_list} failed with return code {proc.returncode}")

    except Exception as e:
        print(f"[Exception] Task {task_index} on GPUs {gpu_list} - Exception occurred: {e}")

    finally:
        for gpu in gpu_list:
            gpu_locks[gpu].release()


def main(task_path, log_dir):
    ensure_log_dir(log_dir)

    config = CONFIG.to_dict()
    data_root = config["data_root"]
    batch_size = config["batch_size"]
    tasks = config["tasks"]
    gpu_locks = create_gpu_locks(tasks)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_task, task, data_root, batch_size, task_path, gpu_locks, i, log_dir)
            for i, task in enumerate(tasks)
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[Exception] Task execution encountered an error: {e}")


if __name__ == '__main__':
    infer_dir = '/data1/nemo/projects/NexusAD/infer_4_vlm'
    main(
        task_path=os.path.join(infer_dir, 'infer_lora.py'),
        log_dir=os.path.join(infer_dir, 'log')
    )

import yaml
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, wait
import threading

def load_yaml(file_path):
    """加载 YAML 配置文件"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_log_dir(log_dir):
    """确保日志目录存在"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

def run_task(task, data_root, batch_size, task_path, gpu_locks, position, log_dir):
    """运行单个任务，并将输出重定向到日志文件"""
    model_dir = task['model_dir']
    # ckpt_dir = task['ckpt_dir']
    json_path = task['json_path']
    out_json_path = task['out_json_path']
    gpu = task.get('gpu', 0)  # 使用 'gpu' 参数，默认使用 GPU 0

    log_file = os.path.join(log_dir, f"task_gpu_{gpu}_{os.path.basename(json_path)}.log")
    print(f"[Running] GPU {gpu}: {os.path.basename(json_path)} - Logging to {log_file}")

    # 使用锁确保同一 GPU 的任务按顺序执行
    gpu_locks[gpu].acquire()

    try:
        # 构建独立的环境变量，并设置 GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)

        # 打开日志文件，捕获输出并实时写入日志文件
        with open(log_file, 'w') as log:
            # 启动子进程，捕获标准输出和错误输出，并实时写入日志文件
            with subprocess.Popen([
                'python', task_path,
                '--data_root', data_root,
                '--batch_size', str(batch_size),
                '--model_dir', model_dir,
                # '--ckpt_dir', ckpt_dir,
                '--json_path', json_path,
                '--out_json_path', out_json_path,
                '--tqdm_position', str(position)
            ], env=env, stdout=log, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:

                proc.wait()  # 等待子进程完成

        # 检查进程退出状态
        if proc.returncode == 0:
            print(f"[Completed] GPU {gpu}: {os.path.basename(json_path)}")
        else:
            print(f"[Error] GPU {gpu}: {os.path.basename(json_path)} failed with return code {proc.returncode}")

    finally:
        # 任务执行完毕，释放锁
        gpu_locks[gpu].release()

def main(config_path, task_path, log_dir):
    """主函数，按 GPU 分配任务"""
    ensure_log_dir(log_dir)  # 确保日志目录存在
    
    config = load_yaml(config_path)
    data_root = config['config']['data_root']
    batch_size = config['config']['batch_size']
    
    tasks = config['tasks']
    
    # 创建一个 GPU 锁字典，用于确保同一 GPU 上的任务按顺序执行
    gpu_locks = {}
    for task in tasks:
        gpu = task.get('gpu', 0)
        if gpu not in gpu_locks:
            gpu_locks[gpu] = threading.Lock()

    # 通过线程池并发处理不同 GPU 上的任务
    futures = []
    with ThreadPoolExecutor() as executor:
        for i, task in enumerate(tasks):
            gpu = task.get('gpu', 0)
            # 为每个任务分配一个线程，并根据 GPU 锁控制执行顺序
            futures.append(executor.submit(run_task, task, data_root, batch_size, task_path, gpu_locks, i, log_dir))
    
    # 等待所有任务完成
    wait(futures)

if __name__ == '__main__':
    infer_dir = '/data1/nemo/projects/NexusAD/3_in_context_learning/uvllm'
    main(
        config_path=os.path.join(infer_dir, 'yamls/241115/tasks_ori_vl_gp_ds_0123.yaml'), 
        task_path=os.path.join(infer_dir, 'infer_lora_ds.py'),
        log_dir=os.path.join(infer_dir, 'log')  # 日志目录
    )

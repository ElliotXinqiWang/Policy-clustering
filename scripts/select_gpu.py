import subprocess
import re

def get_gpu_memory_usage():
    try:
        # nvidia-smiGPU
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpu_info = output.strip().split('\n')
        memory_usage = []
        for line in gpu_info:
            used, total = map(int, line.split(', '))
            memory_usage.append(used)  # ：used / total
        return memory_usage
    except Exception as e:
        print(f"GPU: {e}")
        return []

def select_least_used_gpu():
    memory_usage = get_gpu_memory_usage()
    if not memory_usage:
        return 0  # GPU
    # GPU
    min_usage = min(memory_usage)
    selected_gpu = memory_usage.index(min_usage)
    return selected_gpu

if __name__ == "__main__":
    print(select_least_used_gpu())
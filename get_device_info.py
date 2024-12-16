import os
import torch
import multiprocessing as mp

print("cpu_count:", os.cpu_count())
print("gpu_count:", torch.cuda.device_count())
print("process_count:", torch.cuda.get_device_properties(torch.cuda.current_device))
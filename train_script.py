import subprocess
from os.path import basename
import time

CUDA_VISIBLE_DEVICES_list = [1, 2, 3, 4]
configs = [
    "./configs/unet_base_20_3.yaml",
]
for gpu_id, config in zip(CUDA_VISIBLE_DEVICES_list, configs):
    cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    nohup python train.py \
    --config {config} \
    >{basename(config).replace(".yaml",".out")} \
    2>&1 & \
"""
    subprocess.run(cmd, shell=True)
    time.sleep(5)

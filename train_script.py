import subprocess
from os.path import basename
import time

CUDA_VISIBLE_DEVICES_list = [2, 2, 3, 4]
configs = [
    "./configs/semi_eunet_base20.yaml",
]
for gpu_id, config in zip(CUDA_VISIBLE_DEVICES_list, configs):
    cmd = f"""
    CUDA_VISIBLE_DEVICES={gpu_id} \
    nohup python train.py \
    --config {config} \
    >{basename(config).replace(".yaml",".out")} \
    2>&1 & \
""".strip()
    subprocess.run(cmd, shell=True)
    time.sleep(5)
